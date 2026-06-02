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
NEAR_STOP = 1.5

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


SPECS = [
    ("tollgate_zone", make_tollgate_zone_spec),
    ("two_stops",     make_two_stops_spec),
]


def hoeffding_eps(n, delta=0.05):
    return float(np.sqrt(np.log(2 / delta) / (2 * max(n, 1))))


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

    mono_main_csv = STORAGE_DIR / "MonolithicMain" / "traces.csv"
    mono_shuf_csv = STORAGE_DIR / "MonolithicShuffle" / "traces.csv"

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
        if mono_shuf_csv.exists():
            if len(logs) == len(PRIMITIVES):
                rho_comp_s, eps_comp_s = engine.check_with_dfa_scenic(
                    shuffle_paths, spec, features=["speed"], center_feat_idx=[])
                rho_mono_s = relabel_traces(str(mono_shuf_csv), spec)
                n_mono_s   = pd.read_csv(mono_shuf_csv)["trace_id"].nunique()
                eps_mono_s = hoeffding_eps(n_mono_s)
                diff_s     = abs(rho_comp_s - rho_mono_s)
                tol_s      = 2.0 * (eps_comp_s + eps_mono_s) + 0.15
                status     = "OK" if diff_s <= tol_s else "FAIL"
                print(f"  [shuffle] comp={rho_comp_s:.4f}+/-{eps_comp_s:.4f}  "
                      f"mono={rho_mono_s:.4f}+/-{eps_mono_s:.4f}  "
                      f"|diff|={diff_s:.4f}  tol={tol_s:.4f}  {status}")
            else:
                print(f"  [shuffle] skipped — missing primitives: {missing}")
        else:
            print(f"  [shuffle] MonolithicShuffle not ready yet — skipped")


if __name__ == "__main__":
    main()
