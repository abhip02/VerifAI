"""Quick analysis of paper specs using whatever traces already exist in storage_paper/.

Reads CSVs without regenerating anything. Skips MonolithicShuffle if not ready.

Usage:
    python quick_paper_specs.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC  = HERE.parents[3] / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from verifai.monitor import automaton_specification
from verifai.compositional_analysis import ScenarioBase, CompositionalAnalysisEngine, relabel_traces
from verifai.scenic_composition_analysis import analyze_scenic_composition, build_partner_format
from verifai.scenic_parser import parse_scenic_spec

SCENIC_FILE = HERE / "4_way_intersection_scenic" / "composed_scenarios.scenic"
STORAGE_DIR = HERE / "storage_paper"
WARMUP_STEPS = 25
GATE_SLOW = 3.0
GATE_CAP  = 6.0
NEAR_STOP        = 1.5
NEAR_STOP_MEDIUM = 1.0   # middle threshold for two_stops
GATE_SLOW_TIGHT  = 2.0   # harder to enter gated
GATE_CAP_TIGHT   = 5.0   # lower cap from gated
LINGER_THRESH    = 1.0   # no_linger: near-stop threshold
LINGER_K         = 5     # no_linger: max consecutive slow steps before violation

PRIMITIVES = ["Subscenario1", "Subscenario2L", "Subscenario2R", "Subscenario2S"]


def make_tollgate_zone_spec():
    def transition(state, sym):
        if state == "normal":
            return "gated" if sym == "slow" else "normal"
        if state == "gated":
            return "bad" if sym == "fast" else "gated"
        return "bad"
    def label_row(row):
        if row["step"] < WARMUP_STEPS:
            return "normal"
        spd = float(row["speed"])
        if spd < GATE_SLOW:
            return "slow"
        if spd > GATE_CAP:
            return "fast"
        return "normal"
    return automaton_specification(
        start="normal", inputs={"normal", "slow", "fast"},
        transition=transition, label=lambda s: s != "bad",
        labeling_function=label_row,
    )


def make_tollgate_tight_spec():
    """tollgate_zone with tighter entry (GATE_SLOW_TIGHT) and lower cap (GATE_CAP_TIGHT).
    Fewer Sub1 traces end in gated; violations from gated are still non-trivial."""
    def transition(state, sym):
        if state == "normal":
            return "gated" if sym == "slow" else "normal"
        if state == "gated":
            return "bad" if sym == "fast" else "gated"
        return "bad"
    def label_row(row):
        if row["step"] < WARMUP_STEPS:
            return "normal"
        spd = float(row["speed"])
        if spd < GATE_SLOW_TIGHT:  return "slow"
        if spd > GATE_CAP_TIGHT:   return "fast"
        return "normal"
    return automaton_specification(
        start="normal", inputs={"normal", "slow", "fast"},
        transition=transition, label=lambda s: s != "bad",
        labeling_function=label_row,
    )


def make_no_linger_spec():
    """No more than LINGER_K consecutive steps below LINGER_THRESH.
    DFA resets to initial whenever speed is above threshold, so Sub1 rarely
    ends in a non-initial state and boundary propagation is minimal."""
    slow_states = [f"slow_{i}" for i in range(1, LINGER_K)]

    def transition(state, sym):
        if state == "violated":
            return "violated"
        if sym == "ok":
            return "ok"  # reset to initial
        # sym == "slow"
        if state == "ok":
            return "slow_1"
        idx = int(state.split("_")[1])
        if idx >= LINGER_K - 1:
            return "violated"
        return f"slow_{idx + 1}"

    def label_row(row):
        if row["step"] < WARMUP_STEPS:
            return "ok"
        return "slow" if float(row["speed"]) < LINGER_THRESH else "ok"

    return automaton_specification(
        start="ok",
        inputs={"slow", "ok"},
        transition=transition,
        label=lambda s: s != "violated",
        labeling_function=label_row,
    )


def make_two_stops_spec():
    def transition(state, sym):
        if state == "moving":
            return "stopped_once" if sym == "near_stop" else "moving"
        if state == "stopped_once":
            return "stopped_twice" if sym == "near_stop" else "stopped_once"
        return "stopped_twice"
    def label_row(row):
        if row["step"] < WARMUP_STEPS:
            return "moving"
        return "near_stop" if float(row["speed"]) < NEAR_STOP else "moving"
    return automaton_specification(
        start="moving", inputs={"moving", "near_stop"},
        transition=transition, label=lambda s: s != "stopped_twice",
        labeling_function=label_row,
    )


def make_two_stops_medium_spec():
    """two_stops with NEAR_STOP_MEDIUM threshold — fewer near-stops than original,
    weaker cross-segment correlation, hopefully non-trivial rho."""
    def transition(state, sym):
        if state == "moving":
            return "stopped_once" if sym == "near_stop" else "moving"
        if state == "stopped_once":
            return "stopped_twice" if sym == "near_stop" else "stopped_once"
        return "stopped_twice"
    def label_row(row):
        if row["step"] < WARMUP_STEPS:
            return "moving"
        return "near_stop" if float(row["speed"]) < NEAR_STOP_MEDIUM else "moving"
    return automaton_specification(
        start="moving", inputs={"moving", "near_stop"},
        transition=transition, label=lambda s: s != "stopped_twice",
        labeling_function=label_row,
    )


# SPECS = [
#     ("tollgate_zone",    make_tollgate_zone_spec),
#     ("tollgate_tight",   make_tollgate_tight_spec),
#     ("no_linger",        make_no_linger_spec),
#     ("two_stops",        make_two_stops_spec),
#     ("two_stops_medium", make_two_stops_medium_spec),
# ]

# ── V-shape constants ────────────────────────────────────────────────────────
VSHAPE_HIGH       = 5.0   # m/s — "fast"
VSHAPE_LOW        = 1.5   # m/s — "slow"
VSHAPE_HIGH_LOOSE = 4.0   # easier to reach "fast"
VSHAPE_LOW_LOOSE  = 2.0   # easier to reach "slow"


def make_vshape_cosafety_spec(high=VSHAPE_HIGH, low=VSHAPE_LOW):
    """Co-safety: eventually fast → slow → fast (absorbing-accept on completion).
    NOTE: check_with_dfa treats co-safety as first-segment-only; use this to
    measure and diagnose the gap caused by that limitation."""
    def transition(state, sym):
        if state == "done":
            return "done"
        if state == "ok":
            return "was_fast" if sym == "fast" else "ok"
        if state == "was_fast":
            if sym == "slow": return "was_slow"
            return "was_fast"
        # was_slow
        if sym == "fast": return "done"
        return "was_slow"
    def label_row(row):
        if row["step"] < WARMUP_STEPS:
            return "mid"
        spd = float(row["speed"])
        if spd >= high: return "fast"
        if spd <= low:  return "slow"
        return "mid"
    return automaton_specification(
        start="ok", inputs={"slow", "mid", "fast"},
        transition=transition, label=lambda s: s == "done",
        labeling_function=label_row,
    )


def make_vshape_cosafety_loose_spec():
    """V-shape co-safety with looser thresholds — easier to trigger the pattern."""
    return make_vshape_cosafety_spec(high=VSHAPE_HIGH_LOOSE, low=VSHAPE_LOW_LOOSE)


def make_vshape_safety_spec(high=VSHAPE_HIGH, low=VSHAPE_LOW):
    """Safety complement: never fast → slow → fast (absorbing-reject on violation).
    Included as control — check_with_dfa handles safety correctly."""
    def transition(state, sym):
        if state == "violated":
            return "violated"
        if state == "ok":
            return "was_fast" if sym == "fast" else "ok"
        if state == "was_fast":
            if sym == "slow": return "was_slow"
            return "was_fast"
        # was_slow
        return "violated" if sym == "fast" else "was_slow"
    def label_row(row):
        if row["step"] < WARMUP_STEPS:
            return "mid"
        spd = float(row["speed"])
        if spd >= high: return "fast"
        if spd <= low:  return "slow"
        return "mid"
    return automaton_specification(
        start="ok", inputs={"slow", "mid", "fast"},
        transition=transition, label=lambda s: s != "violated",
        labeling_function=label_row,
    )


def make_no_linger_fast_spec(high=4.0, k=8):
    """No more than k consecutive steps above `high` m/s.
    DFA resets whenever speed drops below high, so boundary propagation is
    minimal (same design principle as no_linger but for fast phases)."""
    def transition(state, sym):
        if state == "violated":
            return "violated"
        if sym != "fast":
            return "ok"  # reset
        if state == "ok":
            return "fast_1"
        idx = int(state.split("_")[1])
        if idx >= k - 1:
            return "violated"
        return f"fast_{idx + 1}"

    def label_row(row):
        if row["step"] < WARMUP_STEPS:
            return "ok"
        return "fast" if float(row["speed"]) > high else "ok"

    return automaton_specification(
        start="ok",
        inputs={"fast", "ok"},
        transition=transition,
        label=lambda s: s != "violated",
        labeling_function=label_row,
    )


# Goal: find vshape co-safety thresholds where per-segment Sub2 rho is non-trivial.
# Sub2 natural profile: starts moderate (3-4 m/s) → slows at turn → re-accelerates.
# For co-safety (fast→slow→fast) to complete WITHIN Sub2:
#   HIGH must be low enough that Sub2 initial speed > HIGH (enters was_fast early)
#   LOW must catch the intersection slow (Sub2 <1.5 m/s = 8-9% of steps)
#   HIGH must be low enough that exit speed > HIGH (completes pattern)
# Trying LOW thresholds (HIGH=2.0-3.0, LOW=0.5-1.5).
def make_vshape_safety_decay_spec(high=3.0, low=1.5, k_decay=40):
    """V-shape safety with was_fast decay.
    After k_decay consecutive non-slow steps in was_fast, the state resets to
    ok. This lets Sub1 (long fast approach) clear its state before the boundary,
    so Sub2 sees a fresh DFA — eliminating boundary propagation.
    Sub2 still completes fast->slow->fast violations from fresh ok start."""
    def transition(state, sym):
        if state == "violated":
            return "violated"
        if state == "was_slow":
            return "violated" if sym == "fast" else "was_slow"
        if state == "ok":
            return f"wf_1" if sym == "fast" else "ok"
        # state is wf_i
        idx = int(state.split("_")[1])
        if sym == "slow":
            return "was_slow"
        # fast or mid — advance decay counter
        if idx >= k_decay:
            return "ok"  # decay: too long in was_fast without seeing slow
        return f"wf_{idx + 1}"

    def label_row(row):
        if row["step"] < WARMUP_STEPS:
            return "mid"
        spd = float(row["speed"])
        if spd >= high: return "fast"
        if spd <= low:  return "slow"
        return "mid"

    return automaton_specification(
        start="ok",
        inputs={"slow", "mid", "fast"},
        transition=transition,
        label=lambda s: s != "violated",
        labeling_function=label_row,
    )


STEER_THRESH = 0.035   # rad/step
STEER_K      = 20     # consecutive sharp steps before violation


def make_steer_spec(thresh=STEER_THRESH, k=STEER_K):
    """Never sustain |dh| > thresh for k consecutive steps (safety DFA)."""
    def transition(state, sym):
        if state == "violated":
            return "violated"
        if sym == "gentle":
            return "ok"
        if state == "ok":
            return "sharp_1"
        idx = int(state.split("_")[1])
        if idx >= k:
            return "violated"
        return f"sharp_{idx + 1}"

    def label_row(row):
        if row["step"] < WARMUP_STEPS:
            return "gentle"
        return "sharp" if abs(float(row["dh"])) > thresh else "gentle"

    return automaton_specification(
        start="ok",
        inputs={"sharp", "gentle"},
        transition=transition,
        label=lambda s: s != "violated",
        labeling_function=label_row,
    )


SPECS = [
    ("no_linger",         make_no_linger_spec),
    ("two_stops_medium",  make_two_stops_medium_spec),
    ("vshape_safety_3p0", lambda: make_vshape_safety_spec(high=3.0, low=1.5)),
    ("steer_k20",         make_steer_spec),
]


def print_speed_stats(logs):
    """Print speed percentiles and threshold crossing rates for each primitive."""
    print("\n  Speed distribution across primitives (post-warmup steps only):")
    print(f"    {'primitive':22s}  {'p10':>5} {'p25':>5} {'p50':>5} {'p75':>5} {'p90':>5}  "
          f"{'<1.5':>5} {'<2.5':>5} {'>3.5':>5} {'>4.0':>5} {'>5.0':>5}")
    for name, csv in logs.items():
        df = pd.read_csv(csv)
        df = df[df["step"] >= WARMUP_STEPS]["speed"].astype(float)
        p = np.percentile(df, [10, 25, 50, 75, 90])
        print(f"    {name:22s}  {p[0]:5.2f} {p[1]:5.2f} {p[2]:5.2f} {p[3]:5.2f} {p[4]:5.2f}  "
              f"{(df<1.5).mean():5.2f} {(df<2.5).mean():5.2f} "
              f"{(df>3.5).mean():5.2f} {(df>4.0).mean():5.2f} {(df>5.0).mean():5.2f}")


def hoeffding_eps(n, delta=0.05):
    return float(np.sqrt(np.log(2 / delta) / (2 * max(n, 1))))


def diagnose_boundary(logs, spec, sub1_name, sub2_names):
    """Show how much of the comp/mono gap comes from DFA state propagation at the boundary.

    Prints:
      - DFA end-state distribution for Sub1
      - Sub2 rho from fresh start vs. from Sub1's end-state distribution
      The difference between these two columns is the boundary effect.
    """
    df1 = pd.read_csv(logs[sub1_name])
    grouped1 = {
        str(tid): grp.sort_values("step").to_dict("records")
        for tid, grp in df1.groupby("trace_id")
    }

    q_start = spec._dfa.start
    q_final_counts: dict = {}
    q_final_accepting: dict = {}
    n_accepting = 0

    for traj in grouped1.values():
        q_end = spec.advance_on_trace(traj, start=q_start)
        q_final_counts[q_end] = q_final_counts.get(q_end, 0) + 1
        if spec._dfa._label(q_end):
            n_accepting += 1
            q_final_accepting[q_end] = q_final_accepting.get(q_end, 0) + 1

    n_total = len(grouped1)
    print(f"\n  {sub1_name} DFA end-state distribution (n={n_total}):")
    for q, c in sorted(q_final_counts.items(), key=lambda x: -x[1]):
        marker = "accept" if spec._dfa._label(q) else "REJECT"
        print(f"    {str(q):20s}  {c:5d}  ({100*c/n_total:5.1f}%)  [{marker}]")

    if n_accepting == 0:
        print("  (all Sub1 traces rejected — cannot propagate)")
        return

    q_init_dist = {q: c / n_accepting for q, c in q_final_accepting.items()}
    print(f"  q_init_dist for Sub2 (conditioned on Sub1 acceptance):")
    for q, w in sorted(q_init_dist.items(), key=lambda x: -x[1]):
        print(f"    {str(q):20s}  {w:.4f}")

    print(f"\n  Sub2 rho: fresh-start vs from-Sub1-end-state")
    print(f"    {'primitive':22s}  {'fresh':>8}  {'from_sub1':>10}  {'delta':>7}")
    for name in sub2_names:
        if name not in logs:
            continue
        df2 = pd.read_csv(logs[name])
        trajs = [
            grp.sort_values("step").to_dict("records")
            for _, grp in df2.groupby("trace_id")
        ]

        fresh_labels = []
        conditioned_labels = []
        for traj in trajs:
            q_end_fresh = spec.advance_on_trace(traj, start=q_start)
            fresh_labels.append(1.0 if spec._dfa._label(q_end_fresh) else 0.0)

            label_cond = 0.0
            for q_s, w in q_init_dist.items():
                q_end = spec.advance_on_trace(traj, start=q_s)
                label_cond += w * (1.0 if spec._dfa._label(q_end) else 0.0)
            conditioned_labels.append(label_cond)

        rho_fresh = float(np.mean(fresh_labels))
        rho_cond  = float(np.mean(conditioned_labels))
        delta     = rho_cond - rho_fresh
        print(f"    {name:22s}  {rho_fresh:8.4f}  {rho_cond:10.4f}  {delta:+7.4f}")


def compute_stitched_shuffle_rho(logs, spec, sub1="Subscenario1",
                                  sub2s=None, n_samples=1000, seed=42):
    """Offline stitched shuffle ground truth.

    Samples n_samples traces by:
      1. Drawing a random Sub1 trace.
      2. Drawing a random permutation of sub2s and one trace from each.
      3. Chaining advance_on_trace across all segments (propagating DFA state).
      4. Returning rho = fraction of stitched traces where the DFA accepts.

    This is the correct independence-assumption ground truth for the shuffle
    operator — each segment is an independent draw from its primitive distribution.
    """
    if sub2s is None:
        sub2s = ["Subscenario2L", "Subscenario2R", "Subscenario2S"]

    rng = np.random.default_rng(seed)

    # Pre-load traces as lists of row dicts, grouped by trace_id
    def load_trajs(csv_path):
        df = pd.read_csv(csv_path, low_memory=False).sort_values("step")
        return [
            grp.to_dict("records")
            for _, grp in df.groupby("trace_id")
        ]

    sub1_trajs  = load_trajs(logs[sub1])
    sub2_trajs  = {s: load_trajs(logs[s]) for s in sub2s}

    q0      = spec._dfa.start
    labels  = []
    for _ in range(n_samples):
        # Sample one trace per segment independently
        traj1 = sub1_trajs[rng.integers(len(sub1_trajs))]
        perm  = rng.permutation(sub2s).tolist()
        trajs = [sub2_trajs[s][rng.integers(len(sub2_trajs[s]))] for s in perm]

        # Chain DFA state across all segments
        q = spec.advance_on_trace(traj1, start=q0)
        for traj in trajs:
            q = spec.advance_on_trace(traj, start=q)

        labels.append(1.0 if spec._dfa._label(q) else 0.0)

    rho = float(np.mean(labels))
    eps = hoeffding_eps(n_samples)
    return rho, eps


def main():
    # Collect available primitive traces
    logs = {}
    for p in PRIMITIVES:
        csv = STORAGE_DIR / p / "traces.csv"
        if csv.exists():
            logs[p] = str(csv)
        else:
            print(f"  [skip] {p} — traces.csv not found")

    if not logs:
        print("No primitive traces found in storage_paper/. Nothing to do.")
        return

    # Parse composition paths
    graph    = analyze_scenic_composition(SCENIC_FILE)
    partner  = build_partner_format(graph)
    all_paths = parse_scenic_spec(partner)
    choose_paths  = all_paths["Main"]
    shuffle_paths = all_paths["ShuffleMain"]

    mono_main_csv  = STORAGE_DIR / "MonolithicMain"    / "traces.csv"
    mono_shuf_csv  = STORAGE_DIR / "MonolithicShuffle" / "traces.csv"

    print_speed_stats(logs)

    engine = CompositionalAnalysisEngine(ScenarioBase(logs))

    for spec_name, spec_fn in SPECS:
        spec = spec_fn()
        print(f"\n{'='*64}")
        print(f"  Spec: {spec_name}")
        print(f"{'='*64}")

        print("  Per-primitive rho:")
        for p, csv in logs.items():
            rho = relabel_traces(csv, spec)
            n   = pd.read_csv(csv)["trace_id"].nunique()
            print(f"    {p:22s}  rho={rho:.4f}  (n={n})")

        # choose
        if len(logs) == len(PRIMITIVES):
            rho_comp_c, eps_comp_c = engine.check_with_dfa_scenic(
                choose_paths, spec, features=["speed"], center_feat_idx=[])
            if mono_main_csv.exists():
                rho_mono_c = relabel_traces(str(mono_main_csv), spec)
                n_mono     = pd.read_csv(mono_main_csv)["trace_id"].nunique()
                eps_mono_c = hoeffding_eps(n_mono)
                diff_c     = abs(rho_comp_c - rho_mono_c)
                tol_c      = 2.0 * (eps_comp_c + eps_mono_c) + 0.15
                status     = "OK" if diff_c <= tol_c else "FAIL"
                print(f"\n  [choose]  comp={rho_comp_c:.4f}+/-{eps_comp_c:.4f}  "
                      f"mono={rho_mono_c:.4f}+/-{eps_mono_c:.4f}  "
                      f"|diff|={diff_c:.4f}  tol={tol_c:.4f}  {status}")
            else:
                print(f"\n  [choose]  comp={rho_comp_c:.4f}+/-{eps_comp_c:.4f}  "
                      f"(MonolithicMain not found — no comparison)")
        else:
            missing = [p for p in PRIMITIVES if p not in logs]
            print(f"\n  [choose]  skipped — missing primitives: {missing}")

        # shuffle
        if len(logs) == len(PRIMITIVES):
            try:
                rho_comp_s, eps_comp_s = engine.check_with_dfa_scenic(
                    shuffle_paths, spec, features=["speed"], center_feat_idx=[])

                # Offline stitched ground truth (correct independence-assumption baseline)
                sub2_prims = [p for p in PRIMITIVES if p != "Subscenario1"]
                rho_stitch, eps_stitch = compute_stitched_shuffle_rho(
                    logs, spec, sub2s=sub2_prims)
                diff_stitch = abs(rho_comp_s - rho_stitch)
                tol_stitch  = 2.0 * (eps_comp_s + eps_stitch) + 0.05
                status_stitch = "OK" if diff_stitch <= tol_stitch else "FAIL"
                print(f"  [shuffle] comp={rho_comp_s:.4f}+/-{eps_comp_s:.4f}  "
                      f"stitched={rho_stitch:.4f}+/-{eps_stitch:.4f}  "
                      f"|diff|={diff_stitch:.4f}  tol={tol_stitch:.4f}  {status_stitch}")

                # Broken sim mono (kept for reference, flagged)
                if mono_shuf_csv.exists():
                    rho_mono_s = relabel_traces(str(mono_shuf_csv), spec)
                    n_mono_s   = pd.read_csv(mono_shuf_csv)["trace_id"].nunique()
                    eps_mono_s = hoeffding_eps(n_mono_s)
                    print(f"  [shuf-sim] mono={rho_mono_s:.4f}+/-{eps_mono_s:.4f}"
                          f"  (broken: trajectory mismatch in perm[1]/perm[2])")
            except Exception as e:
                print(f"  [shuffle] error: {e}")

        sub2s = [p for p in PRIMITIVES if p != "Subscenario1" and p in logs]
        if "Subscenario1" in logs and sub2s:
            diagnose_boundary(logs, spec, "Subscenario1", sub2s)


if __name__ == "__main__":
    main()
