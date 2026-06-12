"""Run all 4 paper specs (MetaDrive + Scenic) and print combined tables.

MetaDrive traces:
  - tollgate:   storage/tollgate/
  - two_stops, vshape_safety, steer: storage/vshape_speed_new/

Scenic traces:  e2e_4way_example/storage_paper/

Usage:
    cd examples/compositional_analysis/dfa_tests
    python run_all4_paper.py
"""

import ast
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE    = Path(__file__).resolve().parent
CA_DIR  = HERE.parent
SRC_DIR = CA_DIR.parent / "src"
E2E_DIR = HERE / "e2e_4way_example"

if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from verifai.monitor import automaton_specification
from verifai.compositional_analysis import (
    ScenarioBase, CompositionalAnalysisEngine, relabel_traces,
)
from verifai.scenic_composition_analysis import analyze_scenic_composition, build_partner_format
from verifai.scenic_parser import parse_scenic_spec

# ---------------------------------------------------------------------------
# Trace directories
# ---------------------------------------------------------------------------
TOLLGATE_DIR = HERE / "storage" / "tollgate"
VSHAPE_DIR   = HERE / "storage" / "vshape_speed_new"
STORAGE_PAPER = E2E_DIR / "storage_paper"
SCENIC_FILE   = E2E_DIR / "4_way_intersection_scenic" / "composed_scenarios.scenic"

COMBINATIONS = [
    ("SX",    ["S", "X"]),
    ("SXS",   ["S", "X", "S"]),
    ("SOC",   ["S", "O", "C"]),
    ("CSXS",  ["C", "S", "X", "S"]),
    ("CXSXC", ["C", "X", "S", "X", "C"]),
]
SCENIC_PRIMITIVES = ["Subscenario1", "Subscenario2L", "Subscenario2R", "Subscenario2S"]
WARMUP_STEPS = 25


def hoeffding_eps(n, delta=0.05):
    return float(np.sqrt(np.log(2 / delta) / (2 * max(n, 1))))


# ---------------------------------------------------------------------------
# Spec definitions
# ---------------------------------------------------------------------------

def make_tollgate_spec():
    K = 3
    wait_states = [f"wait_{i+1}" for i in range(K)]
    def transition(state, sym):
        if state == "moving":
            return "wait_1" if sym == "slow" else "moving"
        if state == "violated":
            return "violated"
        idx = wait_states.index(state)
        if sym == "slow":
            return "moving" if idx == K - 1 else wait_states[idx + 1]
        return "violated"
    return automaton_specification(
        start="moving", inputs={"slow", "fast"},
        transition=transition,
        label=lambda s: s != "violated",
        labeling_function=lambda row: "slow" if row["speed"] < 3.5 else "fast",
    )


def make_two_stops_spec():
    def transition(state, sym):
        if state == "moving":
            return "stopped_once" if sym == "near_stop" else "moving"
        if state == "stopped_once":
            return "stopped_twice" if sym == "near_stop" else "stopped_once"
        return "stopped_twice"
    return automaton_specification(
        start="moving", inputs={"moving", "near_stop"},
        transition=transition,
        label=lambda s: s != "stopped_twice",
        labeling_function=lambda row: "near_stop" if row["speed"] < 3.5 else "moving",
    )


def make_vshape_safety_spec():
    HIGH, LOW = 7.0, 3.5
    def transition(state, sym):
        if state == "violated":
            return "violated"
        if state == "was_slow":
            return "violated" if sym == "fast" else "was_slow"
        if state == "ok":
            return "was_fast" if sym == "fast" else "ok"
        # was_fast
        if sym == "slow":
            return "was_slow"
        return "was_fast"
    def sym(row):
        s = float(row["speed"])
        if s >= HIGH: return "fast"
        if s <= LOW:  return "slow"
        return "mid"
    return automaton_specification(
        start="ok", inputs={"fast", "slow", "mid"},
        transition=transition,
        label=lambda s: s != "violated",
        labeling_function=sym,
    )


def make_steer_metadrive_spec():
    STEER_THRESH = 0.20
    K = 3
    def transition(state, sym):
        if state == "violated":
            return "violated"
        if sym == "gentle":
            return "ok"
        if state == "ok":
            return "sharp_1"
        idx = int(state.split("_")[1])
        if idx >= K:
            return "violated"
        return f"sharp_{idx + 1}"
    def label_row(row):
        try:
            vals = ast.literal_eval(str(row["action"]))
            steer = abs(float(vals[0]))
        except Exception:
            steer = 0.0
        return "sharp" if steer > STEER_THRESH else "gentle"
    return automaton_specification(
        start="ok", inputs={"sharp", "gentle"},
        transition=transition,
        label=lambda s: s != "violated",
        labeling_function=label_row,
    )


def make_scenic_no_linger_spec():
    LINGER_THRESH = 1.0
    LINGER_K = 5
    slow_states = [f"slow_{i}" for i in range(1, LINGER_K)]
    def transition(state, sym):
        if state == "violated":
            return "violated"
        if sym == "ok":
            return "ok"
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
        start="ok", inputs={"ok", "slow"},
        transition=transition,
        label=lambda s: s != "violated",
        labeling_function=label_row,
    )


def make_scenic_two_stops_spec():
    NEAR_STOP_MEDIUM = 1.0
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
        transition=transition,
        label=lambda s: s != "stopped_twice",
        labeling_function=label_row,
    )


def make_scenic_vshape_spec():
    HIGH, LOW = 3.0, 1.5
    def transition(state, sym):
        if state == "violated":
            return "violated"
        if state == "was_slow":
            return "violated" if sym == "fast" else "was_slow"
        if state == "ok":
            return "was_fast" if sym == "fast" else "ok"
        if sym == "slow":
            return "was_slow"
        return "was_fast"
    def sym(row):
        if row["step"] < WARMUP_STEPS:
            return "mid"
        s = float(row["speed"])
        if s >= HIGH: return "fast"
        if s <= LOW:  return "slow"
        return "mid"
    return automaton_specification(
        start="ok", inputs={"fast", "slow", "mid"},
        transition=transition,
        label=lambda s: s != "violated",
        labeling_function=sym,
    )


def make_scenic_steer_spec():
    THRESH, K = 0.035, 20
    def transition(state, sym):
        if state == "violated":
            return "violated"
        if sym == "gentle":
            return "ok"
        if state == "ok":
            return "sharp_1"
        idx = int(state.split("_")[1])
        if idx >= K:
            return "violated"
        return f"sharp_{idx + 1}"
    def label_row(row):
        if row["step"] < WARMUP_STEPS:
            return "gentle"
        return "sharp" if abs(float(row["dh"])) > THRESH else "gentle"
    return automaton_specification(
        start="ok", inputs={"sharp", "gentle"},
        transition=transition,
        label=lambda s: s != "violated",
        labeling_function=label_row,
    )


# ---------------------------------------------------------------------------
# MetaDrive: run one spec
# ---------------------------------------------------------------------------

def run_metadrive_spec(spec_name, spec, trace_dir, features, center_feat_idx):
    print(f"\n{'='*64}")
    print(f"  MetaDrive spec: {spec_name}")
    print(f"  trace_dir: {trace_dir.name}")
    print(f"{'='*64}")

    # Load primitive paths
    prims = ["S", "X", "C", "O"]
    prim_paths = {}
    for p in prims:
        csv = trace_dir / p / "traces.csv"
        if not csv.exists():
            print(f"  [skip] missing {p}")
            return []
        prim_paths[p] = str(csv)
        rho = relabel_traces(str(csv), spec)
        n = pd.read_csv(str(csv))["trace_id"].nunique()
        print(f"  [prim] {p}: rho={rho:.4f}  (n={n})")

    engine = CompositionalAnalysisEngine(ScenarioBase(prim_paths))
    rows = []

    print(f"\n  {'combo':8s}  {'comp':>8}  {'eps_c':>7}  {'mono':>8}  {'eps_m':>7}  "
          f"{'|diff|':>7}  {'tol':>7}  {'OK?':>5}")
    for mono_name, comp_path in COMBINATIONS:
        mono_csv = trace_dir / mono_name / "traces.csv"
        if not mono_csv.exists():
            print(f"  {mono_name:8s}  (missing monolithic CSV)")
            continue

        relabel_traces(str(mono_csv), spec)
        df_mono = pd.read_csv(str(mono_csv))
        rho_mono = df_mono.groupby("trace_id")["label"].last().astype(float).mean()
        n_mono = df_mono["trace_id"].nunique()
        eps_mono = hoeffding_eps(n_mono)

        rho_comp, eps_comp = engine.check_with_dfa(
            comp_path, spec, features=features, center_feat_idx=center_feat_idx,
        )

        diff = abs(rho_comp - rho_mono)
        tol  = 2.0 * (eps_comp + eps_mono) + 0.15
        ok   = "OK" if diff <= tol else "FAIL"

        print(f"  {mono_name:8s}  {rho_comp:8.4f}  {eps_comp:7.4f}  "
              f"{rho_mono:8.4f}  {eps_mono:7.4f}  {diff:7.4f}  {tol:7.4f}  {ok:>5}")
        rows.append({
            "combo": mono_name,
            "comp": rho_comp, "eps_c": eps_comp,
            "mono": rho_mono, "eps_m": eps_mono,
            "diff": diff, "tol": tol, "ok": ok,
        })
    return rows


# ---------------------------------------------------------------------------
# Scenic: run one spec
# ---------------------------------------------------------------------------

def run_scenic_spec(spec_name, spec, choose_paths, shuffle_paths,
                    mono_main_csv, mono_shuf_csv):
    rows = []

    logs = {}
    for p in SCENIC_PRIMITIVES:
        csv = STORAGE_PAPER / p / "traces.csv"
        if not csv.exists():
            print(f"  [skip] missing {p}")
            return []
        logs[p] = str(csv)

    engine = CompositionalAnalysisEngine(ScenarioBase(logs))

    # choose
    rho_comp_c, eps_comp_c = engine.check_with_dfa_scenic(
        choose_paths, spec, features=["speed"], center_feat_idx=[])
    relabel_traces(str(mono_main_csv), spec)
    df_m = pd.read_csv(str(mono_main_csv))
    rho_mono_c = df_m.groupby("trace_id")["label"].last().astype(float).mean()
    eps_mono_c = hoeffding_eps(df_m["trace_id"].nunique())
    diff_c = abs(rho_comp_c - rho_mono_c)
    tol_c  = 2.0 * (eps_comp_c + eps_mono_c) + 0.15
    ok_c   = "OK" if diff_c <= tol_c else "FAIL"
    print(f"  {'choose':8s}  {rho_comp_c:8.4f}  {eps_comp_c:7.4f}  "
          f"{rho_mono_c:8.4f}  {eps_mono_c:7.4f}  {diff_c:7.4f}  {tol_c:7.4f}  {ok_c:>5}")
    rows.append({
        "combo": "choose", "comp": rho_comp_c, "eps_c": eps_comp_c,
        "mono": rho_mono_c, "eps_m": eps_mono_c,
        "diff": diff_c, "tol": tol_c, "ok": ok_c,
    })

    # shuffle
    try:
        rho_comp_s, eps_comp_s = engine.check_with_dfa_scenic(
            shuffle_paths, spec, features=["speed"], center_feat_idx=[])
        relabel_traces(str(mono_shuf_csv), spec)
        df_s = pd.read_csv(str(mono_shuf_csv))
        rho_mono_s = df_s.groupby("trace_id")["label"].last().astype(float).mean()
        eps_mono_s = hoeffding_eps(df_s["trace_id"].nunique())
        diff_s = abs(rho_comp_s - rho_mono_s)
        tol_s  = 2.0 * (eps_comp_s + eps_mono_s) + 0.15
        ok_s   = "OK" if diff_s <= tol_s else "FAIL"
        print(f"  {'shuffle':8s}  {rho_comp_s:8.4f}  {eps_comp_s:7.4f}  "
              f"{rho_mono_s:8.4f}  {eps_mono_s:7.4f}  {diff_s:7.4f}  {tol_s:7.4f}  {ok_s:>5}")
        rows.append({
            "combo": "shuffle", "comp": rho_comp_s, "eps_c": eps_comp_s,
            "mono": rho_mono_s, "eps_m": eps_mono_s,
            "diff": diff_s, "tol": tol_s, "ok": ok_s,
        })
    except Exception as e:
        print(f"  shuffle  error: {e}")

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Parse scenic composition paths once
    graph = analyze_scenic_composition(str(SCENIC_FILE))
    partner = build_partner_format(graph)
    all_paths = parse_scenic_spec(partner)
    choose_paths  = all_paths["Main"]
    shuffle_paths = all_paths["ShuffleMain"]
    mono_main_csv = STORAGE_PAPER / "MonolithicMain"    / "traces.csv"
    mono_shuf_csv = STORAGE_PAPER / "MonolithicShuffle" / "traces.csv"

    PAPER_SPECS = [
        # (display_name, metadrive_spec_fn, trace_dir, features, center_feat_idx, scenic_spec_fn)
        ("tollgate",
         make_tollgate_spec,
         TOLLGATE_DIR,
         ["x", "y", "speed"], [0, 1],
         make_scenic_no_linger_spec),

        ("two_stops",
         make_two_stops_spec,
         VSHAPE_DIR,
         ["x", "y", "speed"], [0, 1],
         make_scenic_two_stops_spec),

        ("vshape_safety",
         make_vshape_safety_spec,
         VSHAPE_DIR,
         ["speed"], [],
         make_scenic_vshape_spec),

        ("steer",
         make_steer_metadrive_spec,
         VSHAPE_DIR,
         ["x", "y", "speed"], [0, 1],
         make_scenic_steer_spec),
    ]

    all_results = {}

    for spec_name, md_spec_fn, trace_dir, feats, cfi, sc_spec_fn in PAPER_SPECS:
        md_spec = md_spec_fn()
        sc_spec = sc_spec_fn()

        # MetaDrive
        md_rows = run_metadrive_spec(spec_name, md_spec, trace_dir, feats, cfi)

        # Scenic
        print(f"\n  Scenic — {spec_name}:")
        print(f"  {'combo':8s}  {'comp':>8}  {'eps_c':>7}  {'mono':>8}  {'eps_m':>7}  "
              f"{'|diff|':>7}  {'tol':>7}  {'OK?':>5}")
        sc_rows = run_scenic_spec(spec_name, sc_spec, choose_paths, shuffle_paths,
                                  mono_main_csv, mono_shuf_csv)

        all_results[spec_name] = md_rows + sc_rows

    # -------------------------------------------------------------------
    # Combined summary tables
    # -------------------------------------------------------------------
    print(f"\n\n{'='*72}")
    print("  COMBINED RESULTS — all 4 specs × 7 rows")
    print(f"{'='*72}")
    hdr = f"  {'combo':8s}  {'comp':>8}  {'±eps_c':>7}  {'mono':>8}  {'±eps_m':>7}  {'|diff|':>7}  {'tol':>7}  {'OK?':>5}"
    for spec_name, rows in all_results.items():
        print(f"\n  ── {spec_name} ──")
        print(hdr)
        for r in rows:
            print(f"  {r['combo']:8s}  {r['comp']:8.4f}  {r['eps_c']:7.4f}  "
                  f"{r['mono']:8.4f}  {r['eps_m']:7.4f}  {r['diff']:7.4f}  "
                  f"{r['tol']:7.4f}  {r['ok']:>5}")


if __name__ == "__main__":
    main()
