"""Compositional analysis with SCENARIO primitives (not behaviors).

Composition (`composed_scenarios.scenic`):
    Main = Subscenario1 ; do choose { Subscenario2L, Subscenario2R, Subscenario2S }
    => 3 distinct execution paths

Each sub-scenario is a self-contained `scenario X():` block with its own
setup that creates an ego using `EgoBehavior(trajectory=...)` —
`FollowTrajectoryBehavior` over real lane-network maneuvers, not open-loop
`take` actions. This is the rich-scenario composition path: the ego
actually navigates the road graph rather than just steering blindly.

Trace generation goes through `generate_graph_scenarios` (the library
companion to `generate_graph_traces` that handles self-contained leaf
scenarios; see its docstring for the wrapper-builder incompatibility it
sidesteps). Same DFA spec, relabel, and `check_with_dfa_scenic` pipeline
as the behavior-based 4-way test.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[3] / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from verifai.monitor import automaton_specification
from verifai.compositional_analysis import (
    ScenarioBase,
    CompositionalAnalysisEngine,
    relabel_traces,
)
from verifai.scenic_composition_analysis import (
    analyze_scenic_composition,
    build_partner_format,
)
from verifai.scenic_parser import parse_scenic_spec, get_primitives
from verifai.generate_graph_traces import generate_graph_scenarios


SCENIC_FILE = HERE / "4_way_intersection_scenic" / "composed_scenarios.scenic"
SAVE_DIR    = HERE / "storage_scenarios"
N_TRACES    = 300  # MetaDrive traces per sub-scenario (parallel, ~4 procs)
MONO_N      = 300  # MetaDrive traces of MonolithicMain (single-process)
MAX_STEPS   = 85   # ticks per per-primitive trace BEFORE trim.
                   # Subscenario2*'s EgoBehaviorWithPrewarm prepends Uniform(0..25)
                   # ticks of cruise prewarm; we trim PREWARM_TRIM=25 rows from
                   # each Sub2 CSV post-generation so row 0 represents a warm
                   # state (varied speed at the intersection) matching where
                   # Sub1 ends, fixing the KDE boundary mismatch that was
                   # zeroing out segment-2 contributions to compositional rho.
                   # Sub1 has no prewarm and is not trimmed; trace stays 85 ticks.
                   # Monolithic uses 2*MAX_STEPS = 170 ticks per trace.
PREWARM_TRIM = 25  # rows trimmed from start of each Sub2 CSV
                   # (== max prewarm value in EgoBehaviorWithPrewarm)
SUB2_MAX_STEPS = MAX_STEPS + PREWARM_TRIM  # = 110. Sub2 needs PREWARM_TRIM extra raw
                   # ticks so that AFTER trimming the prewarm prefix, the remaining
                   # CSV has MAX_STEPS = 85 rows — matching the per-segment
                   # evaluation length monolithic seg-2 sees (85 ticks of
                   # `MonolithicMain`'s second FollowTrajectoryBehavior call).
                   # Without this bump, Sub2 post-trim has only 60 rows, and the
                   # spec's warmup mask further drops it to 35 evaluated ticks vs
                   # mono seg-2's 85 → Sub2 ρ overestimates "stays below MAX",
                   # which the multiplicative engine then over-multiplies.
SUB2_PRIMITIVES = {"Subscenario2L", "Subscenario2R", "Subscenario2S"}  # which primitives have prewarm

# DFA spec: SAFETY (absorbing-reject) — speed never exceeds MAX_SPEED post-warmup.
# Why safety, not liveness? `check_with_dfa` multiplies per-segment rhos with
# state conditioning on acceptance: rho = rho_step1 * rho_step2 where each
# rho_step is "fraction accepting given the segment starts from the
# previous-step's acceptance-conditioned q distribution". For absorbing-reject,
# this correctly equals P(survive whole composition). For absorbing-accept
# (liveness), it collapses: once Sub1 accepts, q_init for Sub2 = {accept: 1.0},
# Sub2 trivially stays in accept, rho_step_2 = 1.0, so compositional rho =
# Sub1's rho exactly. (That's the 0.31 we kept seeing.)
# Safety semantics: traces with UBER_SPEED close to 8 will exceed 5.5 -> reject;
# traces with UBER_SPEED close to 2 stay below -> accept. Graded rho across the
# UBER_SPEED=Range(2,8) distribution.
WARMUP_STEPS = 25
MAX_SPEED    = 5.5


# ---------------------------------------------------------------------------
# DFA spec
# ---------------------------------------------------------------------------

def make_spec():
    """Safety: speed stays at-or-below MAX_SPEED at every post-warmup step.
    Once exceeded, DFA enters `bad` and stays there (absorbing-reject)."""
    def transition(state, sym):
        if state == "bad":
            return "bad"
        return "bad" if sym == "high" else "ok"

    def label_row(row):
        if row["step"] < WARMUP_STEPS:
            return "low"  # masked during warmup
        return "high" if row["speed"] > MAX_SPEED else "low"

    return automaton_specification(
        start="ok",
        inputs={"high", "low"},
        transition=transition,
        label=lambda s: s == "ok",
        labeling_function=label_row,
    )


def hoeffding_eps(n, delta=0.05):
    return float(np.sqrt(np.log(2 / delta) / (2 * max(n, 1))))


def trim_prewarm(csv_path, n, step_offset=0):
    """Drop the first `n` rows of each trace and renumber `step` starting at
    `step_offset`. For Sub2 traces we set `step_offset=WARMUP_STEPS` so the
    spec's warmup mask (`row["step"] < WARMUP_STEPS`) doesn't re-fire on
    already-warm rows — Sub2 is post-prewarm at row 0, so it shouldn't be
    re-treated as a cold-start trace."""
    df = pd.read_csv(csv_path).sort_values(["trace_id", "step"])
    trimmed = []
    for tid, grp in df.groupby("trace_id"):
        kept = grp.iloc[n:].copy()
        kept["step"] = range(step_offset, step_offset + len(kept))
        trimmed.append(kept)
    pd.concat(trimmed, ignore_index=True).to_csv(csv_path, index=False)


def _max_steps_for(name):
    """Per-scenario raw simulation length. Sub2 needs the extra PREWARM_TRIM
    ticks so post-trim CSV length == MAX_STEPS (matches mono seg-2 length)."""
    return SUB2_MAX_STEPS if name in SUB2_PRIMITIVES else MAX_STEPS


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(reuse_traces=False):
    # 1. Parse Scenic -> partner -> Main composition.
    graph    = analyze_scenic_composition(SCENIC_FILE)
    partner  = build_partner_format(graph)
    paths    = parse_scenic_spec(partner)["Main"]
    primitives = get_primitives(paths)
    print(f"Source     : {SCENIC_FILE}")
    print(f"Primitives : {sorted(primitives)}")
    print(f"Composition: {paths}")

    # 2. Per-scenario trace generation. In --reuse_traces mode, regenerate
    #    only the per-primitive CSVs that are missing (so deleting just
    #    Subscenario2L/R/S CSVs lets us regen those without re-running Sub1
    #    or the long monolithic).
    def _post_process(name, csv_path):
        # Sub2 traces: trim prewarm prefix AND shift step values past the
        # warmup mask so the spec doesn't re-mask already-warm rows. Sub1
        # is a cold-start trace and gets the warmup mask normally.
        if name in SUB2_PRIMITIVES:
            trim_prewarm(csv_path, PREWARM_TRIM, step_offset=WARMUP_STEPS)

    if reuse_traces:
        logs = {}
        missing_for_regen = []
        for primitive in sorted(primitives):
            csv_path = SAVE_DIR / primitive / "traces.csv"
            if csv_path.exists():
                logs[primitive] = str(csv_path)
                print(f"[reuse] {primitive}: {csv_path}")
            else:
                print(f"[reuse] {primitive} CSV missing, will regenerate")
                missing_for_regen.append(primitive)
        if missing_for_regen:
            max_steps_map = {n: _max_steps_for(n) for n in missing_for_regen}
            regen_logs = generate_graph_scenarios(
                SCENIC_FILE, missing_for_regen,
                n=N_TRACES, save_dir=SAVE_DIR, max_steps=max_steps_map,
            )
            for name, csv_path in regen_logs.items():
                _post_process(name, csv_path)
                logs[name] = csv_path
    else:
        max_steps_map = {n: _max_steps_for(n) for n in primitives}
        logs = generate_graph_scenarios(
            SCENIC_FILE, sorted(primitives),
            n=N_TRACES, save_dir=SAVE_DIR, max_steps=max_steps_map,
        )
        for name, csv_path in logs.items():
            _post_process(name, csv_path)
    missing = primitives - logs.keys()
    if missing:
        raise RuntimeError(f"missing primitives: {sorted(missing)}")

    # 3. Apply DFA, compute per-primitive rho.
    spec = make_spec()
    print("\n=== Per-primitive rho ===")
    for name in sorted(primitives):
        rho = relabel_traces(logs[name], spec)
        print(f"  {name:18s} rho = {rho:.4f}  ({logs[name]})")

    # 4. Compositional analysis on the 2-step path.
    engine = CompositionalAnalysisEngine(ScenarioBase(logs))
    # Match boundary distributions on `speed` only — dropping (x, y) because
    # Sub1 ends at varying (x, y) along its trajectory (UBER_SPEED-dependent)
    # while Sub2 spawns at a near-fixed (x, y) 0-3m before the intersection.
    # That geometric mismatch makes the 2D position KDE collapse to ~0 weight
    # at the boundary even after prewarm, zeroing out segment-2's contribution
    # and pinning compositional ρ to Sub1's per-primitive ρ. Speed is the only
    # feature the DFA spec reads, so it's the one that matters for the boundary
    # state distribution.
    rho_comp, eps_comp = engine.check_with_dfa_scenic(
        paths, spec,
        features=["speed"], center_feat_idx=[],
    )

    # 5. Monolithic counterpart: simulate the full Subscenario1 -> turn chain
    #    end-to-end via MonolithicMain (uses MonolithicEgoBehavior to chain
    #    the two FollowTrajectoryBehaviors back-to-back). Each trace is
    #    2*MAX_STEPS = 170 ticks of continuous driving. Generated by the
    #    same library function (single-scenario invocation).
    MONO_NAME = "MonolithicMain"
    mono_csv_path = SAVE_DIR / MONO_NAME / "traces.csv"
    if reuse_traces and mono_csv_path.exists():
        mono_csv = str(mono_csv_path)
        print(f"[reuse] monolithic: {mono_csv}")
    else:
        if reuse_traces:
            print(f"[reuse] monolithic CSV missing at {mono_csv_path}, regenerating")
        mono_logs = generate_graph_scenarios(
            SCENIC_FILE, [MONO_NAME],
            n=MONO_N, save_dir=SAVE_DIR, max_steps=MAX_STEPS * 2,
        )
        mono_csv = mono_logs[MONO_NAME]
    rho_mono = relabel_traces(mono_csv, spec)
    n_mono = pd.read_csv(mono_csv)["trace_id"].nunique()
    eps_mono = hoeffding_eps(n_mono)

    print(f"\n=== Compositional vs Monolithic (n_mono={n_mono}) ===")
    print(f"  Compositional : rho = {rho_comp:.4f} +/- {eps_comp:.4f}")
    print(f"  Monolithic    : rho = {rho_mono:.4f} +/- {eps_mono:.4f}")
    print(f"  |diff|        : {abs(rho_comp - rho_mono):.4f}")
    print(f"  paths         : {paths}")


def test_4way_intersection_scenarios():
    main()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="4-way intersection scenarios test")
    parser.add_argument("--reuse_traces", action="store_true",
                        help="Skip trace generation and use existing CSVs in SAVE_DIR")
    args = parser.parse_args()
    main(reuse_traces=args.reuse_traces)
