"""Traversal-wander compositional analysis (split-primitive design).

4 primitive scenarios, each self-contained:
    ApproachScenario   — ego drives from far to the intersection entry
    TurnLeftScenario   — ego executes a left turn (with prewarm)
    TurnRightScenario  — ego executes a right turn (with prewarm)
    GoStraightScenario — ego drives straight through (with prewarm)

Compositional: 10-step Main = 5× (Approach ; do choose{L/R/S})
               4 trace pools sampled independently in parallel
               3^5 = 243 distinct execution paths

Monolithic:    Monolithic5 — ONE simulation, 5 chained traversals
               Each traversal = approach at speed sa_i + turn at independent
               speed st_i, mirroring the per-primitive independence assumption.

Spec: safe_under_max (speed <= MAX_SPEED at every post-warmup step).
"""
import sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC  = HERE.parents[3] / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from verifai.monitor import automaton_specification
from verifai.compositional_analysis import (
    ScenarioBase,
    CompositionalAnalysisEngine,
    relabel_traces,
)
from verifai.scenic_composition_analysis import analyze_scenic_composition, build_partner_format
from verifai.scenic_parser import parse_scenic_spec
from verifai.generate_graph_traces import generate_graph_scenarios

SCENIC_FILE = HERE / "4_way_intersection_scenic" / "traversal_wander.scenic"
SAVE_DIR    = HERE / "storage_traversal_wander"

N_TRACES    = 100   # per-primitive traces
MONO_N      = 100   # Monolithic5 traces
MAX_STEPS   = 100   # ticks per primitive (approach ~50-80; turn prewarm+turn ~60-90)
MONO_STEPS  = 700   # 5 traversals × ~140 ticks each (approach ~50 + turn ~90)

WARMUP_STEPS = 10   # skip initial acceleration from rest
MAX_SPEED    = 7.5  # P(target <= 7.5) = 0.92 with Range(2,8) → rho ≈ 0.42 over 10 segments


def make_spec():
    """Safety: speed stays <= MAX_SPEED at every post-warmup step."""
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


def hoeffding_eps(n, delta=0.05):
    return float(np.sqrt(np.log(2 / delta) / (2 * max(n, 1))))


def main(reuse_traces=False):
    # --- Parse 10-step composition from Scenic file ---
    graph   = analyze_scenic_composition(SCENIC_FILE)
    partner = build_partner_format(graph)
    paths   = parse_scenic_spec(partner)["Main"]
    print(f"Composition steps: {len(paths)}  (5× Approach + 5× do choose)")

    primitives = [
        "ApproachScenario",
        "TurnLeftScenario",
        "TurnRightScenario",
        "GoStraightScenario",
    ]

    # --- Per-primitive trace generation ---
    logs = {}
    if reuse_traces:
        missing = []
        for p in primitives:
            csv = SAVE_DIR / p / "traces.csv"
            if csv.exists():
                logs[p] = str(csv)
                print(f"[reuse] {p}: {csv}")
            else:
                print(f"[reuse] {p} missing, will regenerate")
                missing.append(p)
        if missing:
            new_logs = generate_graph_scenarios(
                SCENIC_FILE, missing,
                n=N_TRACES, save_dir=SAVE_DIR, max_steps=MAX_STEPS,
            )
            logs.update(new_logs)
    else:
        logs = generate_graph_scenarios(
            SCENIC_FILE, primitives,
            n=N_TRACES, save_dir=SAVE_DIR, max_steps=MAX_STEPS,
        )

    spec   = make_spec()
    engine = CompositionalAnalysisEngine(ScenarioBase(logs))

    print("\n=== Per-primitive rho ===")
    for p in primitives:
        rho = relabel_traces(logs[p], spec)
        print(f"  {p:22s} rho = {rho:.4f}")

    # --- Compositional rho ---
    rho_comp, eps_comp = engine.check_with_dfa_scenic(
        paths, spec, features=["speed"], center_feat_idx=[],
    )

    # --- Monolithic: Monolithic5 ---
    mono_csv = SAVE_DIR / "Monolithic5" / "traces.csv"
    if reuse_traces and mono_csv.exists():
        print(f"\n[reuse] Monolithic5: {mono_csv}")
    else:
        if reuse_traces:
            print(f"\n[reuse] Monolithic5 missing, regenerating")
        mono_logs = generate_graph_scenarios(
            SCENIC_FILE, ["Monolithic5"],
            n=MONO_N, save_dir=SAVE_DIR, max_steps=MONO_STEPS,
        )
        mono_csv = Path(mono_logs["Monolithic5"])

    rho_mono = relabel_traces(str(mono_csv), spec)
    n_mono   = pd.read_csv(mono_csv)["trace_id"].nunique()
    eps_mono = hoeffding_eps(n_mono)

    # --- Results ---
    print("\n=== Compositional vs Monolithic ===")
    print(f"  {'':40s}  {'rho':>10s}  {'eps':>8s}")
    print(f"  {'compositional (10-step: 5×Approach+Turn)':<40s}  {rho_comp:>10.4f}  {eps_comp:>8.4f}")
    print(f"  {'monolithic  (Monolithic5, 5 traversals)':<40s}  {rho_mono:>10.4f}  {eps_mono:>8.4f}")
    print(f"  {'|diff|':<40s}  {abs(rho_comp - rho_mono):>10.4f}")


def test_4way_intersection_traversal_wander():
    main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse_traces", action="store_true")
    args = parser.parse_args()
    main(reuse_traces=args.reuse_traces)
