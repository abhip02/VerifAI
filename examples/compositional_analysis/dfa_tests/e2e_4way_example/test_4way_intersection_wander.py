"""Wander composition + multi-spec post-hoc evaluation.

Composition (`composed_wander.scenic`):
    Main = 5x do choose { GoStraight, TurnLeft, TurnRight }
    => paths = [(1.0, [{GS:1/3, TL:1/3, TR:1/3}] * 5)]
    => 3^5 = 243 distinct execution paths

The compositional analysis engine evaluates these via importance sampling on
the same 3-primitive trace pool that the existing test uses (no expansion of
all 243 paths). Trace generation runs once; multiple DFA specs are then
evaluated against the same pool — one of the genuine wins of compositional
analysis (specs are decoupled from simulation).

Specs evaluated post-hoc:
    reach_high     : did speed exceed HIGH = 2.5 at any post-warmup step?
    safe_under_max : did speed stay <= MAX = 4.0 the whole post-warmup?
    rise_then_fall : reached >= HIGH, then later dropped to <= LOW (NM)
    k_consec_slow  : never had > K=3 consecutive slow steps post-warmup (NM)

Rerun cost per added spec: ~1 second of Python (no MetaDrive).
"""
import sys
from pathlib import Path

import csv
import os

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[3] / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from verifai.monitor import automaton_specification
from verifai.compositional_analysis import ScenarioBase, CompositionalAnalysisEngine
from verifai.generate_graph_traces import _trajectory_rows
from verifai.scenic_composition_analysis import (
    analyze_scenic_composition,
    build_partner_format,
)
from verifai.scenic_parser import parse_scenic_spec, get_primitives
from verifai.generate_graph_traces import generate_graph_traces


SCENIC_FILE = HERE / "4_way_intersection_scenic" / "composed_wander.scenic"
SAVE_DIR    = HERE / "storage_wander"
N_TRACES    = 300   # MetaDrive traces per primitive (parallel); bump for tighter eps
MONO_N      = 300   # MetaDrive traces of MonolithicWander (single-process, 5x longer per trace)
MAX_STEPS   = 40   # ticks per primitive trace -> 4s simulated each
                   # monolithic uses 5*MAX_STEPS = 200 ticks per trace

# Spec parameters (shared)
# Tuned so each spec lands somewhere graded for the actual speed distribution
# in this composition (post-warmup speeds typically 1-6 m/s, ramping):
#   HIGH=4.0  -> ~half traces reach this (graded reach_high / rise_then_fall)
#   MAX=5.5   -> top end; fast traces exceed (graded safe_under_max)
#   LOW=0.5   -> only Brake-tail or warmup-ramp segments dip below
WARMUP_STEPS    = 5
STOP_THRESHOLD  = 0.5    # m/s, "slow" boundary
HIGH_SPEED      = 4.0    # m/s, "fast" boundary for reach_high / rise_then_fall
MAX_SPEED       = 5.5    # m/s, safety upper bound
LOW_SPEED       = 0.5    # m/s, "slow" boundary for rise_then_fall
MAX_CONSEC_SLOW = 3      # K for k_consec_slow
TURN_THRESHOLD  = 0.05   # rad/step heading rate to register as turning
LEFT_TURNS_NEEDED = 2    # K for the "at least K left turns" spec


# ---------------------------------------------------------------------------
# DFA spec factories
# ---------------------------------------------------------------------------

def make_spec_reach_high():
    """Liveness: did speed exceed HIGH at any post-warmup step?"""
    def transition(state, sym):
        if state == "accept":
            return "accept"
        return "accept" if sym == "fast" else "start"

    def label_row(row):
        if row["step"] < WARMUP_STEPS:
            return "other"
        return "fast" if row["speed"] >= HIGH_SPEED else "other"

    return automaton_specification(
        start="start",
        inputs={"fast", "other"},
        transition=transition,
        label=lambda s: s == "accept",
        labeling_function=label_row,
    )


def make_spec_safe_under_max():
    """Safety: did speed stay <= MAX_SPEED for the whole post-warmup?"""
    def transition(state, sym):
        if state == "violated":
            return "violated"
        return "violated" if sym == "over" else "ok"

    def label_row(row):
        if row["step"] < WARMUP_STEPS:
            return "ok"
        return "over" if row["speed"] > MAX_SPEED else "ok"

    return automaton_specification(
        start="ok",
        inputs={"ok", "over"},
        transition=transition,
        label=lambda s: s != "violated",
        labeling_function=label_row,
    )


def make_spec_rise_then_fall():
    """Non-Markovian: trace reached >= HIGH at some post-warmup step AND
    later dropped to <= LOW. Encodes the "accelerated then slowed" pattern."""
    def transition(state, sym):
        if state == "accept":
            return "accept"
        if state == "start":
            return "saw_fast" if sym == "fast" else "start"
        # state == "saw_fast"
        return "accept" if sym == "slow" else "saw_fast"

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
        label=lambda s: s == "accept",
        labeling_function=label_row,
    )


def make_spec_two_left_turns():
    """Non-Markovian counts: car must complete >= LEFT_TURNS_NEEDED distinct
    left-turn episodes (debounced via the mid-turn state). A `_in_left` state
    soaks up consecutive `left` symbols so a single sustained turn = 1 episode."""
    K = LEFT_TURNS_NEEDED

    def transition(state, sym):
        if state == "done":
            return "done"
        if state.endswith("_in_left"):
            n = int(state[1:state.index("_")])
            if sym == "left":
                return state                      # still turning
            n += 1                                # turn just ended
            return "done" if n >= K else f"q{n}"
        # not currently in a left turn
        n = int(state[1:])
        return f"q{n}_in_left" if sym == "left" else state

    def label_row(row):
        if row["step"] < WARMUP_STEPS:
            return "straight"
        d = row.get("dheading", 0.0)
        if d <= -TURN_THRESHOLD: return "left"
        if d >=  TURN_THRESHOLD: return "right"
        return "straight"

    return automaton_specification(
        start="q0",
        inputs={"left", "right", "straight"},
        transition=transition,
        label=lambda s: s == "done",
        labeling_function=label_row,
    )


def make_spec_k_consec_slow():
    """Non-Markovian K-counter: > K=3 consecutive slow steps post-warmup => bad."""
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def relabel(csv_path, spec):
    df = pd.read_csv(csv_path).sort_values("step")
    labels = {tid: spec.evaluate(grp.to_dict("records")) > 0
              for tid, grp in df.groupby("trace_id")}
    df["label"] = df["trace_id"].map(labels)
    df.to_csv(csv_path, index=False)
    return df.groupby("trace_id")["label"].last().astype(float).mean()


def hoeffding_eps(n, delta=0.05):
    return float(np.sqrt(np.log(2 / delta) / (2 * max(n, 1))))


def add_derived_columns(csv_path):
    """Add per-step heading derivative `dheading` to a trace CSV.
    Idempotent: re-running just overwrites with the same values."""
    df = pd.read_csv(csv_path).sort_values(["trace_id", "step"])
    df["dheading"] = df.groupby("trace_id")["heading"].diff().fillna(0.0)
    df.to_csv(csv_path, index=False)


# ---------------------------------------------------------------------------
# Monolithic counterpart: a single MetaDrive simulation per trace, where the
# ego runs the `Wander` behavior (5 segments, each pre-sampled at scene
# creation, total 5*MAX_STEPS = 200 ticks). This is the literal continuous-
# drive analogue of the 5-step `do choose` composition.
# ---------------------------------------------------------------------------

def generate_monolithic_wander_traces(scenic_file, save_dir, n, max_steps,
                                      scenario_name="MonolithicWander"):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import scenic

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "traces.csv"

    sc = scenic.scenarioFromFile(
        str(scenic_file),
        scenario=scenario_name,
        mode2D=True,
        model="scenic.simulators.metadrive.model",
    )
    sim = sc.getSimulator()
    print(f"[mono-wander] generating {n} traces of `{scenario_name}` "
          f"(maxSteps={max_steps}) -> {csv_path}")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trace_id", "step", "x", "y", "heading", "speed",
                        "action", "reward", "label"],
        )
        writer.writeheader()
        trace_id = 0
        attempts = 0
        max_attempts = max(1000, n * 20)
        while trace_id < n and attempts < max_attempts:
            attempts += 1
            try:
                scene, _ = sc.generate(maxIterations=2000, verbosity=0)
                simulation = sim.simulate(scene, maxSteps=max_steps,
                                          verbosity=0, maxIterations=1)
            except Exception as exc:
                print(f"[mono-wander] attempt {attempts} failed: {exc}")
                continue
            if simulation is None:
                continue
            for row in _trajectory_rows(simulation, trace_id):
                writer.writerow(row)
            f.flush()
            if hasattr(simulation, "destroy"):
                try:
                    simulation.destroy()
                except Exception:
                    pass
            trace_id += 1
    return str(csv_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(reuse_traces=False):
    # 1. Parse Scenic -> partner -> N-step composition.
    # Explicitly select Main: the parser also finds Wander/MonolithicWander
    # as entrypoint candidates (used only for the monolithic counterpart),
    # and `scenic_to_check_input` would pick whichever is defined first.
    graph    = analyze_scenic_composition(SCENIC_FILE)
    partner  = build_partner_format(graph)
    paths    = parse_scenic_spec(partner)["Main"]
    primitives = get_primitives(paths)
    print(f"Source     : {SCENIC_FILE}")
    print(f"Primitives : {sorted(primitives)}")
    print(f"Wander len : {len(paths)} sequential `do choose` steps "
          f"({len(primitives)**len(paths)} distinct execution paths)")

    # 2. Per-primitive trace pool: regenerate or reuse existing CSVs.
    # `composed_wander.scenic` also contains the Wander behavior and
    # MonolithicWander scenario for the monolithic counterpart; the parser
    # picks them up as extra leaf containers and generate_graph_traces makes
    # wrappers for them. Filter the resulting logs to only the primitives
    # actually referenced in Main's composition.
    if reuse_traces:
        logs = {}
        for primitive in primitives:
            csv_path = SAVE_DIR / primitive / "traces.csv"
            if csv_path.exists():
                logs[primitive] = str(csv_path)
                print(f"[reuse] {primitive}: {csv_path}")
            else:
                print(f"[ERROR] No existing traces for primitive {primitive} at {csv_path}")
    else:
        raw_logs = generate_graph_traces(
            source=str(SCENIC_FILE),
            n=N_TRACES,
            save_dir=str(SAVE_DIR),
            backend="metadrive",
            max_steps=MAX_STEPS,
        )
        logs = {k: v for k, v in raw_logs.items() if k in primitives}
    missing = primitives - logs.keys()
    if missing:
        raise RuntimeError(f"missing primitives: {sorted(missing)}")

    # 3. Evaluate each spec against the shared pool (Python-only loop).
    specs = {
        "reach_high"    : make_spec_reach_high(),
        "safe_under_max": make_spec_safe_under_max(),
        "rise_then_fall": make_spec_rise_then_fall(),
        "k_consec_slow" : make_spec_k_consec_slow(),
    }
    engine = CompositionalAnalysisEngine(ScenarioBase(logs))

    print("\n=== Per-primitive rho per spec ===")
    prims = sorted(logs)
    header = f"  {'spec':<16s}  " + "  ".join(f"{p:>12s}" for p in prims)
    print(header)
    for sname, spec in specs.items():
        row = {p: relabel(logs[p], spec) for p in prims}
        line = f"  {sname:<16s}  " + "  ".join(f"{row[p]:>12.4f}" for p in prims)
        print(line)

    print("\n=== Compositional rho per spec (single shared trace pool) ===")
    comp_results = {}
    for sname, spec in specs.items():
        # The label column in each CSV is left as whatever the LAST relabel
        # call wrote, but check_with_dfa_scenic recomputes verdicts from the
        # spec's labeling function — so the order of specs doesn't matter.
        rho_comp, eps_comp = engine.check_with_dfa_scenic(
            paths, spec,
            features=["x", "y", "speed"], center_feat_idx=[0, 1],
        )
        comp_results[sname] = (rho_comp, eps_comp)
        print(f"  {sname:<16s}: rho = {rho_comp:.4f} +/- {eps_comp:.4f}")

    # 4. Monolithic counterpart: one MetaDrive sim per trace, ego runs the
    #    `Wander` behavior end-to-end (5 segments × MAX_STEPS = 200 ticks).
    if reuse_traces:
        mono_csv = str(SAVE_DIR / "monolithic" / "traces.csv")
        if not Path(mono_csv).exists():
            raise RuntimeError(f"No existing monolithic traces at {mono_csv}")
        print(f"[reuse] monolithic: {mono_csv}")
    else:
        mono_csv = generate_monolithic_wander_traces(
            SCENIC_FILE, SAVE_DIR / "monolithic", MONO_N,
            max_steps=MAX_STEPS * 5,
        )
    n_mono = pd.read_csv(mono_csv)["trace_id"].nunique()
    eps_mono = hoeffding_eps(n_mono)

    print(f"\n=== Compositional vs Monolithic (n_mono={n_mono}) ===")
    print(f"  {'spec':<16s}  {'compositional':>22s}  {'monolithic':>22s}  {'|diff|':>8s}")
    for sname, spec in specs.items():
        rho_mono = relabel(mono_csv, spec)
        rho_comp, eps_comp = comp_results[sname]
        diff = abs(rho_comp - rho_mono)
        print(f"  {sname:<16s}  "
              f"{rho_comp:>10.4f} +/- {eps_comp:.4f}  "
              f"{rho_mono:>10.4f} +/- {eps_mono:.4f}  "
              f"{diff:>8.4f}")


def test_4way_intersection_wander():
    main()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="4-way intersection wander test")
    parser.add_argument("--reuse_traces", action="store_true",
                        help="Skip trace generation and use existing CSVs in SAVE_DIR")
    args = parser.parse_args()
    main(reuse_traces=args.reuse_traces)
