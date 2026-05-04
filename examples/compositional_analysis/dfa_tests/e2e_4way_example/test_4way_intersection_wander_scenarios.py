"""Wander composition over SCENARIO primitives + multi-spec evaluation.

Composition (`wander_scenarios.scenic`):
    Main = 5x do choose { GoStraightScenario, TurnLeftScenario,
                          TurnRightScenario, BrakeScenario }
    => 4^5 = 1024 distinct execution paths

Each leaf is a self-contained `scenario X():` block with its own ego setup
(rather than a behavior attached to a Main-owned ego). Generalizes the
2-step `test_4way_intersection_scenarios` to 5 sequential decisions, and
adds the multi-spec post-hoc evaluation pattern from
`test_4way_intersection_wander.py`.

Trace generation goes through `generate_graph_scenarios` (the library
companion to `generate_graph_traces` for self-contained leaf scenarios;
see its docstring for the wrapper-builder incompatibility it sidesteps).

Boundary KDE features: SPEED ONLY. Each per-primitive scenario re-spawns at
the same `uberSpawnPoint` so position is fixed across primitives but varies
across primitives' end positions. Matching on (x, y) collapses the KDE
weights to ~0; speed is the only feature that meaningfully bridges segment
boundaries here. See memory: check_with_dfa_safety_only and the 2-step
scenarios test for the same lesson.
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


SCENIC_FILE = HERE / "4_way_intersection_scenic" / "wander_scenarios.scenic"
SAVE_DIR    = HERE / "storage_wander_scenarios"
N_TRACES    = 100  # MetaDrive traces per leaf scenario (parallel, ~4 procs)
MONO_N      = 100  # MetaDrive traces of MonolithicWander (single-process)
MAX_STEPS   = 75   # ticks per per-primitive trace BEFORE trim.
                   # Each leaf behavior prepends Uniform(0..35) ticks of
                   # cruise prewarm; we trim PREWARM_TRIM=35 rows post-
                   # generation so row 0 is a warm state with varied speed.
                   # Net useful per-primitive length = MAX_STEPS - PREWARM_TRIM
                   # = 40 ticks, matching WANDER_SEGMENT_LEN in the scenic.
                   # Monolithic uses 5*WANDER_SEGMENT_LEN = 200 ticks per trace.
PREWARM_TRIM = 35  # rows trimmed from start of each per-primitive CSV
                   # (== max prewarm value in the scenic file)


# Spec parameters — all SAFETY (absorbing-reject) DFAs.
# Liveness specs were removed because check_with_dfa's multiplicative formula
# with acceptance-conditioning collapses for absorbing-accept DFAs (compositional
# ρ pins to step-1's per-primitive ρ regardless of later segments). See the
# memory note `check_with_dfa_safety_only.md` for the algorithm bug.
WARMUP_STEPS         = 5
STOP_THRESHOLD       = 0.5    # m/s, "slow" symbol boundary
HIGH_SPEED           = 3.0    # m/s, "fast" boundary for the rise-then-fall pattern
LOW_SPEED            = 0.5    # m/s, "slow" boundary for the rise-then-fall pattern
FAST_K_THR           = 1.5    # m/s, "fast" boundary for k_consec_fast
MAX_CONSEC_SLOW      = 3      # K for k_consec_slow
MAX_CONSEC_FAST      = 3      # K for k_consec_fast
MAX_BRAKE_EPISODES_1 = 1      # K for at_most_one_brake_episode
MAX_BRAKE_EPISODES_2 = 2      # K for at_most_two_brake_episodes
TOTAL_SLOW_BUDGET    = 80     # max post-warmup slow-step count summed across
                              # the whole composition. Per-primitive BrakeScenario
                              # gives ~40 slow ticks; ~K=2 brake segments fit.


# ---------------------------------------------------------------------------
# DFA spec factories — ALL SAFETY (absorbing-reject)
# ---------------------------------------------------------------------------
# Pattern: every spec has one absorbing reject state ("bad" / "violated").
# `check_with_dfa`'s multiplicative formula `rho_step1 × rho_step2 × ...`
# with acceptance-conditioning correctly computes P(survive whole composition)
# for this DFA class — see the 2-step scenarios test for the validating run.
#
# The non-Markovian specs (k_consec_*, at_most_K_brake_*, no_rise_then_fall,
# bounded_total_slow) all encode their relevant accumulated state in the DFA
# itself. Conditioning on acceptance preserves that state across segment
# boundaries — e.g., `slow_2` after Sub1 means "two consecutive slow ticks
# at end of Sub1", and Sub2 starting from `slow_2` will violate on the next
# slow symbol. This is the cross-segment state propagation that compositional
# analysis is designed to exploit.

def make_spec_k_consec_slow():
    """SAFETY non-Markovian (correct): never have > MAX_CONSEC_SLOW
    consecutive slow steps post-warmup. Absorbing `bad` reject."""
    def transition(state, sym):
        if state == "ok_run":
            return "slow_1" if sym == "slow" else "ok_run"
        if state == "bad":
            return "bad"
        idx = int(state.split("_")[1])
        if sym == "fast":
            return "ok_run"
        return "bad" if idx >= MAX_CONSEC_SLOW else f"slow_{idx + 1}"

    def label_row(row):
        if row["step"] < WARMUP_STEPS:
            return "fast"
        return "slow" if row["speed"] < STOP_THRESHOLD else "fast"

    return automaton_specification(
        start="ok_run",
        inputs={"slow", "fast"},
        transition=transition,
        label=lambda s: s != "bad",
        labeling_function=label_row,
    )


def make_spec_k_consec_fast():
    """SAFETY non-Markovian (correct, mirror of k_consec_slow):
    never have > MAX_CONSEC_FAST consecutive fast steps post-warmup."""
    K = MAX_CONSEC_FAST

    def transition(state, sym):
        if state == "ok":
            return "fast_1" if sym == "fast" else "ok"
        if state == "bad":
            return "bad"
        idx = int(state.split("_")[1])
        if sym == "slow":
            return "ok"
        return "bad" if idx >= K else f"fast_{idx + 1}"

    def label_row(row):
        if row["step"] < WARMUP_STEPS:
            return "slow"
        return "fast" if row["speed"] >= FAST_K_THR else "slow"

    return automaton_specification(
        start="ok",
        inputs={"slow", "fast"},
        transition=transition,
        label=lambda s: s != "bad",
        labeling_function=label_row,
    )


def _make_spec_at_most_K_brake_episodes(K):
    """SAFETY counts: at most K distinct slow->fast transitions allowed.
    A 'brake episode' = a contiguous slow run that ends in a fast step.
    Debounced via the in-slow auxiliary state. Absorbing `violated` reject.
    Cross-segment: an in-slow state at end of one segment chains into the
    next segment — the slow run hasn't ended yet until a fast tick arrives,
    even if that fast tick is in a later segment."""
    def transition(state, sym):
        if state == "violated":
            return "violated"
        if state.endswith("_in_slow"):
            n = int(state[1:state.index("_")])
            if sym == "slow":
                return state
            n += 1
            return "violated" if n > K else f"q{n}"
        n = int(state[1:])
        return f"q{n}_in_slow" if sym == "slow" else state

    def label_row(row):
        if row["step"] < WARMUP_STEPS:
            return "fast"
        return "slow" if row["speed"] < STOP_THRESHOLD else "fast"

    return automaton_specification(
        start="q0",
        inputs={"slow", "fast"},
        transition=transition,
        label=lambda s: s != "violated",
        labeling_function=label_row,
    )


def make_spec_at_most_one_brake_episode():
    return _make_spec_at_most_K_brake_episodes(MAX_BRAKE_EPISODES_1)


def make_spec_at_most_two_brake_episodes():
    """K=2 variant — should be more permissive (~5/4 expected episodes
    across 5 segments, so most traces fit under K=2)."""
    return _make_spec_at_most_K_brake_episodes(MAX_BRAKE_EPISODES_2)


def make_spec_no_rise_then_fall():
    """SAFETY non-Markovian (correct): trace must NEVER complete the pattern
    `rose to HIGH, then dropped to LOW`. Once observed, DFA enters `bad`.
    This is the safety dual of the rise_then_fall liveness spec.

    Per-primitive: each leaf scenario completes only the rise phase (Cruise)
    or only the fall phase (Brake) on its own — no single 40-tick segment
    does both. So per-primitive ρ should be ≈ 1 across the board.

    Compositional: the `saw_fast` state propagates across segment boundaries.
    A Cruise segment ending in `saw_fast` followed by a Brake segment that
    emits `slow` triggers `bad`. So compositional ρ should be < 1 — exactly
    the cross-segment effect this test is designed to demonstrate."""
    def transition(state, sym):
        if state == "bad":
            return "bad"
        if state == "start":
            return "saw_fast" if sym == "fast" else "start"
        # state == "saw_fast"
        return "bad" if sym == "slow" else "saw_fast"

    def label_row(row):
        if row["step"] < WARMUP_STEPS:
            return "mid"
        if row["speed"] >= HIGH_SPEED:
            return "fast"
        if row["speed"] <= LOW_SPEED:
            return "slow"
        return "mid"

    return automaton_specification(
        start="start",
        inputs={"fast", "mid", "slow"},
        transition=transition,
        label=lambda s: s != "bad",
        labeling_function=label_row,
    )


def make_spec_bounded_total_slow():
    """SAFETY global counter (correct): total post-warmup slow-step count
    summed across the whole composition must stay <= TOTAL_SLOW_BUDGET.
    Once exceeded, DFA enters `bad`.

    DFA states: t0, t1, ..., t{BUDGET}, bad — encodes accumulated slow count.
    Per-primitive: BrakeScenario gives ~40 slow ticks; cruise primitives ~0.
    Per-primitive ρ depends on which q_init the DFA starts from — i.e.,
    how much budget the previous segment(s) used. The engine evaluates this
    correctly by iterating over q_init in `_dfa_labels`.

    Compositional: with BUDGET=80, ~2 BrakeScenario segments fit under the
    budget; ≥3 brakes violate. Mono ρ ≈ P(at most 2 of 5 random segments are
    Brake) = (3/4)^5 + 5(3/4)^4(1/4) + 10(3/4)^3(1/4)^2 ≈ 0.90."""
    BUDGET = TOTAL_SLOW_BUDGET

    def transition(state, sym):
        if state == "bad":
            return "bad"
        n = int(state[1:])  # state is "tN"
        if sym == "slow":
            n += 1
            return "bad" if n > BUDGET else f"t{n}"
        return state  # fast → no count change

    def label_row(row):
        if row["step"] < WARMUP_STEPS:
            return "fast"
        return "slow" if row["speed"] < STOP_THRESHOLD else "fast"

    return automaton_specification(
        start="t0",
        inputs={"slow", "fast"},
        transition=transition,
        label=lambda s: s != "bad",
        labeling_function=label_row,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hoeffding_eps(n, delta=0.05):
    return float(np.sqrt(np.log(2 / delta) / (2 * max(n, 1))))


def add_derived_columns(csv_path):
    """Add per-step heading derivative `dheading` so two_left_turns can read
    it. Idempotent: re-running just overwrites with the same values."""
    df = pd.read_csv(csv_path).sort_values(["trace_id", "step"])
    df["dheading"] = df.groupby("trace_id")["heading"].diff().fillna(0.0)
    df.to_csv(csv_path, index=False)


def trim_prewarm(csv_path, n, step_offset=0):
    """Drop the first `n` rows of each trace and renumber `step` starting at
    `step_offset`. Setting `step_offset=WARMUP_STEPS` for prewarmed primitives
    makes the spec's warmup mask (`row["step"] < WARMUP_STEPS`) NOT re-fire
    on already-warm rows — the prewarm prefix already serves as cold-start
    ramp, so per-primitive row 0 is post-warmup state, not cold-start.

    Without the shift, a per-primitive trace evaluates only
    (post-trim length - WARMUP_STEPS) ticks, while monolithic segments 2..N
    evaluate the full segment length (they don't get a fresh warmup mask
    mid-trace). This length asymmetry biases per-primitive ρ for safety
    specs (less time = fewer chances to violate), and the multiplicative
    composition formula then amplifies that bias N-fold across N segments.

    NOT idempotent — assumes CSV still has prewarm rows."""
    df = pd.read_csv(csv_path).sort_values(["trace_id", "step"])
    trimmed = []
    for tid, grp in df.groupby("trace_id"):
        kept = grp.iloc[n:].copy()
        kept["step"] = range(step_offset, step_offset + len(kept))
        trimmed.append(kept)
    pd.concat(trimmed, ignore_index=True).to_csv(csv_path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(reuse_traces=False):
    # 1. Parse Scenic -> partner -> 5-step composition.
    graph    = analyze_scenic_composition(SCENIC_FILE)
    partner  = build_partner_format(graph)
    paths    = parse_scenic_spec(partner)["Main"]
    primitives = get_primitives(paths)
    print(f"Source     : {SCENIC_FILE}")
    print(f"Primitives : {sorted(primitives)}")
    print(f"Wander len : {len(paths)} sequential `do choose` steps "
          f"({len(primitives)**len(paths)} distinct execution paths)")

    # 2. Per-leaf trace pool.
    if reuse_traces:
        logs = {}
        for primitive in primitives:
            csv_path = SAVE_DIR / primitive / "traces.csv"
            if csv_path.exists():
                logs[primitive] = str(csv_path)
                print(f"[reuse] {primitive}: {csv_path}")
            else:
                print(f"[ERROR] No existing traces for {primitive} at {csv_path}")
    else:
        logs = generate_graph_scenarios(
            SCENIC_FILE, sorted(primitives),
            n=N_TRACES, save_dir=SAVE_DIR, max_steps=MAX_STEPS,
        )
        # All 4 leaf scenarios have prewarm. Trim the prewarm prefix AND
        # shift step values past the warmup mask so the spec doesn't re-mask
        # already-warm rows (matches per-primitive evaluation length to
        # mono per-segment evaluation length, fixing the bias documented
        # in `trim_prewarm`'s docstring).
        for primitive, csv_path in logs.items():
            trim_prewarm(csv_path, PREWARM_TRIM, step_offset=WARMUP_STEPS)
        # Add dheading column (kept for any future turn-based safety spec
        # that reads it; current spec set doesn't use it).
        for primitive, csv_path in logs.items():
            add_derived_columns(csv_path)
    missing = primitives - logs.keys()
    if missing:
        raise RuntimeError(f"missing primitives: {sorted(missing)}")

    # 3. Per-spec evaluation against the shared pool. ALL safety specs.
    specs = {
        "k_consec_slow"      : make_spec_k_consec_slow(),
        "k_consec_fast"      : make_spec_k_consec_fast(),
        "at_most_one_brake"  : make_spec_at_most_one_brake_episode(),
        "at_most_two_brake"  : make_spec_at_most_two_brake_episodes(),
        "no_rise_then_fall"  : make_spec_no_rise_then_fall(),
        "bounded_total_slow" : make_spec_bounded_total_slow(),
    }
    engine = CompositionalAnalysisEngine(ScenarioBase(logs))

    print("\n=== Per-primitive rho per spec ===")
    prims = sorted(logs)
    header = f"  {'spec':<20s}  " + "  ".join(f"{p:>18s}" for p in prims)
    print(header)
    for sname, spec in specs.items():
        row = {p: relabel_traces(logs[p], spec) for p in prims}
        line = f"  {sname:<20s}  " + "  ".join(f"{row[p]:>18.4f}" for p in prims)
        print(line)

    print("\n=== Compositional rho per spec (single shared trace pool) ===")
    comp_results = {}
    for sname, spec in specs.items():
        rho_comp, eps_comp = engine.check_with_dfa_scenic(
            paths, spec,
            features=["speed"], center_feat_idx=[],
        )
        comp_results[sname] = (rho_comp, eps_comp)
        print(f"  {sname:<20s}: rho = {rho_comp:.4f} +/- {eps_comp:.4f}")

    # 4. Monolithic counterpart: ego runs WanderBehavior end-to-end
    #    (5 segments × WANDER_SEGMENT_LEN = 200 ticks). Same library
    #    function, single-scenario invocation.
    MONO_NAME = "MonolithicWander"
    mono_csv_path = SAVE_DIR / MONO_NAME / "traces.csv"
    if reuse_traces and mono_csv_path.exists():
        mono_csv = str(mono_csv_path)
        print(f"[reuse] monolithic: {mono_csv}")
    else:
        if reuse_traces:
            print(f"[reuse] monolithic CSV missing at {mono_csv_path}, regenerating")
        mono_logs = generate_graph_scenarios(
            SCENIC_FILE, [MONO_NAME],
            n=MONO_N, save_dir=SAVE_DIR, max_steps=200,
        )
        mono_csv = mono_logs[MONO_NAME]
    add_derived_columns(mono_csv)
    n_mono = pd.read_csv(mono_csv)["trace_id"].nunique()
    eps_mono = hoeffding_eps(n_mono)

    print(f"\n=== Compositional vs Monolithic (n_mono={n_mono}) ===")
    print(f"  {'spec':<20s}  {'compositional':>22s}  {'monolithic':>22s}  {'|diff|':>8s}")
    for sname, spec in specs.items():
        rho_mono = relabel_traces(mono_csv, spec)
        rho_comp, eps_comp = comp_results[sname]
        diff = abs(rho_comp - rho_mono)
        print(f"  {sname:<20s}  "
              f"{rho_comp:>10.4f} +/- {eps_comp:.4f}  "
              f"{rho_mono:>10.4f} +/- {eps_mono:.4f}  "
              f"{diff:>8.4f}")


def test_4way_intersection_wander_scenarios():
    main()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="4-way intersection wander scenarios test")
    parser.add_argument("--reuse_traces", action="store_true",
                        help="Skip trace generation and use existing CSVs in SAVE_DIR")
    args = parser.parse_args()
    main(reuse_traces=args.reuse_traces)
