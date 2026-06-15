"""Compositional analysis with SCENARIO primitives (not behaviors).

Composition (`composed_scenarios.scenic`):
    Main = Subscenario1 ; do choose { Subscenario2L, Subscenario2R, Subscenario2S }
    => 3 distinct execution paths

Each sub-scenario is a self-contained `scenario X():` block whose setup creates
an ego using `EgoBehavior(trajectory=...)` — `FollowTrajectoryBehavior` over
real lane-network maneuvers, not open-loop `take` actions.  This is the rich-
scenario composition path: the ego actually navigates the road graph.

Three DFA specs are evaluated:

  speed_safety        — absorbing-reject if speed > MAX_SPEED=5.5 post-warmup.
  no_near_stop_twice  — absorbing-reject if speed drops below NEAR_STOP=1.5 m/s
                        more than once (safety: at most one near-stop).
  tollgate_zone       — once speed drops below GATE_SLOW=3.0 m/s (slow zone),
                        absorbing-reject if speed later exceeds GATE_CAP=6.0 m/s
                        (post-gate speed cap, like a tollgate speed limit).

All three are absorbing-REJECT safety specs — `check_with_dfa` handles these
correctly.  Absorbing-accept (liveness) specs are NOT used here because they
collapse compositional rho to the first segment's rho (see memory note).
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
SUB2_MAX_STEPS = MAX_STEPS + PREWARM_TRIM  # = 110
SUB2_PRIMITIVES = {"Subscenario2L", "Subscenario2R", "Subscenario2S"}

# Spec thresholds
WARMUP_STEPS = 25
MAX_SPEED    = 5.5   # speed_safety cap
NEAR_STOP    = 1.5   # no_near_stop_twice threshold (m/s)
GATE_SLOW    = 3.0   # tollgate_zone: entering the slow zone below this speed
GATE_CAP     = 6.0   # tollgate_zone: must not exceed this after entering slow zone


# ---------------------------------------------------------------------------
# DFA specs (all absorbing-reject / safety)
# ---------------------------------------------------------------------------

def make_speed_safety_spec():
    """Speed never exceeds MAX_SPEED post-warmup (absorbing-reject)."""
    def transition(state, sym):
        if state == "bad":
            return "bad"
        return "bad" if sym == "high" else "ok"

    def label_row(row):
        if row["step"] < WARMUP_STEPS:
            return "low"
        return "high" if row["speed"] > MAX_SPEED else "low"

    return automaton_specification(
        start="ok",
        inputs={"high", "low"},
        transition=transition,
        label=lambda s: s == "ok",
        labeling_function=label_row,
    )


# Keep old name so external callers still work.
make_spec = make_speed_safety_spec


def make_no_near_stop_twice_spec():
    """At most one near-stop (speed < NEAR_STOP) post-warmup.

    DFA states
    ----------
    moving       : no near-stop seen yet
    once_stopped : exactly one near-stop seen; still OK
    twice_stopped: second near-stop seen → absorbing reject

    This is the same DFA family as test_check_with_dfa_metadrive_two_stops.py,
    adapted to the MetaDrive intersection scenario.  A near-stop can occur at
    spawn (before the ego accelerates) or mid-trajectory during a sharp turn.
    """
    def transition(state, sym):
        if state == "moving":
            return "once_stopped" if sym == "near_stop" else "moving"
        if state == "once_stopped":
            return "twice_stopped" if sym == "near_stop" else "once_stopped"
        return "twice_stopped"  # absorbing

    def label_row(row):
        if row["step"] < WARMUP_STEPS:
            return "moving"
        return "near_stop" if row["speed"] < NEAR_STOP else "moving"

    return automaton_specification(
        start="moving",
        inputs={"moving", "near_stop"},
        transition=transition,
        label=lambda s: s != "twice_stopped",
        labeling_function=label_row,
    )


def make_tollgate_zone_spec():
    """Tollgate-zone safety: once the car enters the slow zone (speed < GATE_SLOW),
    it must not re-accelerate beyond GATE_CAP.

    DFA states
    ----------
    normal  : car has not yet entered the slow zone
    gated   : car has passed through the slow zone; speed cap now active
    bad     : speed exceeded GATE_CAP while in gated state → absorbing reject

    Traces that never drop below GATE_SLOW stay in `normal` forever and always
    pass (no gate encountered, no constraint active).  Only traces that do slow
    down (e.g. at the intersection approach or during a turn) get the post-gate
    cap applied.

    Symbolically: "if you once went slow (≤ GATE_SLOW), you have passed the
    tollgate and must respect the GATE_CAP speed limit afterward."
    """
    def transition(state, sym):
        if state == "normal":
            return "gated" if sym == "slow" else "normal"
        if state == "gated":
            return "bad" if sym == "fast" else "gated"
        return "bad"  # absorbing

    def label_row(row):
        if row["step"] < WARMUP_STEPS:
            return "normal"
        spd = row["speed"]
        if spd < GATE_SLOW:
            return "slow"
        if spd > GATE_CAP:
            return "fast"
        return "normal"

    return automaton_specification(
        start="normal",
        inputs={"normal", "slow", "fast"},
        transition=transition,
        label=lambda s: s != "bad",
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

    # 2. Per-scenario trace generation (shared across all specs).
    def _post_process(name, csv_path):
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

    # 3. Monolithic ground-truth (generated once, reused across specs).
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
    n_mono = pd.read_csv(mono_csv)["trace_id"].nunique()
    eps_mono = hoeffding_eps(n_mono)

    # 4. Evaluate all specs.
    specs = {
        "speed_safety":       make_speed_safety_spec(),
        "no_near_stop_twice": make_no_near_stop_twice_spec(),
        "tollgate_zone":      make_tollgate_zone_spec(),
    }

    engine = CompositionalAnalysisEngine(ScenarioBase(logs))

    for spec_name, spec in specs.items():
        print(f"\n{'='*60}")
        print(f"Spec: {spec_name}")
        print(f"{'='*60}")

        print("  Per-primitive rho:")
        for name in sorted(primitives):
            rho = relabel_traces(logs[name], spec)
            print(f"    {name:18s} rho = {rho:.4f}")

        rho_comp, eps_comp = engine.check_with_dfa_scenic(
            paths, spec,
            features=["speed"], center_feat_idx=[],
        )
        rho_mono = relabel_traces(mono_csv, spec)

        print(f"  Compositional : rho = {rho_comp:.4f} +/- {eps_comp:.4f}")
        print(f"  Monolithic    : rho = {rho_mono:.4f} +/- {eps_mono:.4f}")
        print(f"  |diff|        : {abs(rho_comp - rho_mono):.4f}")


def test_4way_intersection_scenarios():
    main()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="4-way intersection scenarios test")
    parser.add_argument("--reuse_traces", action="store_true",
                        help="Skip trace generation and use existing CSVs in SAVE_DIR")
    args = parser.parse_args()
    main(reuse_traces=args.reuse_traces)
