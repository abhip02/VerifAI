"""Scenic 4-way intersection — paper specs: tollgate-zone and two-stops.

Two DFA safety specs (both absorbing-reject) evaluated over two scenic
composition operators:

  choose  : Sub1 ; do choose { Sub2L, Sub2R, Sub2S }   → MonolithicMain
  shuffle : Sub1 ; do shuffle { Sub2L, Sub2R, Sub2S }   → MonolithicShuffle

Specs
-----
tollgate_zone
    Once the ego enters the slow zone (speed < GATE_SLOW = 3.0 m/s) it must
    not re-accelerate above GATE_CAP = 6.0 m/s.  Triggered naturally by the
    intersection approach and turn; no obstacles needed.

two_stops
    At most one near-stop (speed < NEAR_STOP = 1.5 m/s) post-warmup.

Usage
-----
    pytest test_4way_paper_specs.py -s              # all 4 tests
    python test_4way_paper_specs.py                 # main() printout
    python test_4way_paper_specs.py --reuse         # skip generation

Traces stored in storage_paper/ (N=1000 each, separate from storage_scenarios/).
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
SRC  = HERE.parents[3] / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from verifai.monitor import automaton_specification
from verifai.compositional_analysis import (
    ScenarioBase, CompositionalAnalysisEngine, relabel_traces,
)
from verifai.scenic_composition_analysis import analyze_scenic_composition, build_partner_format
from verifai.scenic_parser import parse_scenic_spec
from verifai.generate_graph_traces import generate_graph_scenarios

SCENIC_FILE  = HERE / "4_way_intersection_scenic" / "composed_scenarios.scenic"
STORAGE_DIR  = HERE / "storage_paper"
N_TRACES     = 1000
MAX_STEPS    = 85
PREWARM_TRIM = 25          # rows trimmed from Sub2* traces (== max prewarm in scenic)
WARMUP_STEPS = 25          # step-offset assigned after trim; warmup mask in specs

SUB2_PRIMITIVES  = {"Subscenario2L", "Subscenario2R", "Subscenario2S"}
SUB2_MAX_STEPS   = MAX_STEPS + PREWARM_TRIM   # 110: extra headroom for prewarm

GATE_SLOW = 3.0   # m/s — entering slow zone
GATE_CAP  = 6.0   # m/s — speed cap after slow zone
NEAR_STOP = 1.5   # m/s — near-stop threshold


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

def make_tollgate_zone_spec():
    """Once speed < GATE_SLOW, must not exceed GATE_CAP (absorbing-reject)."""
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
        start="normal",
        inputs={"normal", "slow", "fast"},
        transition=transition,
        label=lambda s: s != "bad",
        labeling_function=label_row,
    )


def make_two_stops_spec():
    """At most one near-stop (speed < NEAR_STOP) post-warmup (absorbing-reject)."""
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
        start="moving",
        inputs={"moving", "near_stop"},
        transition=transition,
        label=lambda s: s != "stopped_twice",
        labeling_function=label_row,
    )


SPECS = [
    ("tollgate_zone", make_tollgate_zone_spec),
    ("two_stops",     make_two_stops_spec),
]


# ---------------------------------------------------------------------------
# Trace helpers
# ---------------------------------------------------------------------------

def _max_steps_for(name):
    return SUB2_MAX_STEPS if name in SUB2_PRIMITIVES else MAX_STEPS


def trim_prewarm(csv_path, n, step_offset):
    df = pd.read_csv(csv_path).sort_values(["trace_id", "step"])
    trimmed = []
    for _, grp in df.groupby("trace_id"):
        kept = grp.iloc[n:].copy()
        kept["step"] = range(step_offset, step_offset + len(kept))
        trimmed.append(kept)
    pd.concat(trimmed, ignore_index=True).to_csv(csv_path, index=False)


def hoeffding_eps(n, delta=0.05):
    return float(np.sqrt(np.log(2 / delta) / (2 * max(n, 1))))


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def setup(request):
    reuse = getattr(request, "param", False)
    return _build_setup(reuse=reuse)


def _build_setup(reuse=False):
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    graph     = analyze_scenic_composition(SCENIC_FILE)
    partner   = build_partner_format(graph)
    all_paths = parse_scenic_spec(partner)
    choose_paths  = all_paths["Main"]
    shuffle_paths = all_paths["ShuffleMain"]

    primitives = ["Subscenario1", "Subscenario2L", "Subscenario2R", "Subscenario2S"]

    # Per-primitive traces
    if reuse:
        logs = {}
        missing = []
        for p in primitives:
            csv = STORAGE_DIR / p / "traces.csv"
            if csv.exists():
                logs[p] = str(csv)
            else:
                missing.append(p)
        if missing:
            max_steps_map = {p: _max_steps_for(p) for p in missing}
            new_logs = generate_graph_scenarios(
                SCENIC_FILE, missing,
                n=N_TRACES, save_dir=STORAGE_DIR, max_steps=max_steps_map,
            )
            for name, csv_path in new_logs.items():
                if name in SUB2_PRIMITIVES:
                    trim_prewarm(csv_path, PREWARM_TRIM, step_offset=WARMUP_STEPS)
                logs[name] = csv_path
    else:
        max_steps_map = {p: _max_steps_for(p) for p in primitives}
        logs = generate_graph_scenarios(
            SCENIC_FILE, primitives,
            n=N_TRACES, save_dir=STORAGE_DIR, max_steps=max_steps_map,
        )
        for name, csv_path in logs.items():
            if name in SUB2_PRIMITIVES:
                trim_prewarm(csv_path, PREWARM_TRIM, step_offset=WARMUP_STEPS)

    # Monolithic choose (MonolithicMain)
    mono_main_csv = STORAGE_DIR / "MonolithicMain" / "traces.csv"
    if not (reuse and mono_main_csv.exists()):
        mono_logs = generate_graph_scenarios(
            SCENIC_FILE, ["MonolithicMain"],
            n=N_TRACES, save_dir=STORAGE_DIR, max_steps=MAX_STEPS * 2,
        )
        mono_main_csv = Path(mono_logs["MonolithicMain"])

    # Monolithic shuffle (MonolithicShuffle)
    mono_shuf_csv = STORAGE_DIR / "MonolithicShuffle" / "traces.csv"
    if not (reuse and mono_shuf_csv.exists()):
        shuf_logs = generate_graph_scenarios(
            SCENIC_FILE, ["MonolithicShuffle"],
            n=N_TRACES, save_dir=STORAGE_DIR, max_steps=MAX_STEPS * 4,
        )
        mono_shuf_csv = Path(shuf_logs["MonolithicShuffle"])

    return dict(
        logs=logs,
        mono_main_csv=str(mono_main_csv),
        mono_shuf_csv=str(mono_shuf_csv),
        choose_paths=choose_paths,
        shuffle_paths=shuffle_paths,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec_name,spec_fn", SPECS)
def test_choose_vs_monolithic(setup, spec_name, spec_fn):
    """Compositional rho (choose) should match MonolithicMain rho."""
    logs          = setup["logs"]
    mono_main_csv = setup["mono_main_csv"]
    choose_paths  = setup["choose_paths"]
    spec = spec_fn()

    print(f"\n  [{spec_name}] per-primitive rho:")
    for p, csv in logs.items():
        rho = relabel_traces(csv, spec)
        print(f"    {p:20s} rho={rho:.4f}")

    engine = CompositionalAnalysisEngine(ScenarioBase(logs))
    rho_comp, eps_comp = engine.check_with_dfa_scenic(
        choose_paths, spec, features=["speed"], center_feat_idx=[],
    )

    rho_mono = relabel_traces(mono_main_csv, spec)
    n_mono   = pd.read_csv(mono_main_csv)["trace_id"].nunique()
    eps_mono = hoeffding_eps(n_mono)

    diff      = abs(rho_comp - rho_mono)
    tolerance = 2.0 * (eps_comp + eps_mono) + 0.15
    print(f"\n  [{spec_name}] choose")
    print(f"    Compositional rho = {rho_comp:.4f} +/- {eps_comp:.4f}")
    print(f"    Monolithic    rho = {rho_mono:.4f} +/- {eps_mono:.4f}")
    print(f"    |diff| = {diff:.4f}  tolerance = {tolerance:.4f}")

    assert diff <= tolerance, (
        f"{spec_name} choose: |diff|={diff:.4f} > tolerance={tolerance:.4f} "
        f"(comp={rho_comp:.4f}, mono={rho_mono:.4f})"
    )


@pytest.mark.parametrize("spec_name,spec_fn", SPECS)
def test_shuffle_vs_monolithic(setup, spec_name, spec_fn):
    """Compositional rho (shuffle) should match MonolithicShuffle rho."""
    logs          = setup["logs"]
    mono_shuf_csv = setup["mono_shuf_csv"]
    shuffle_paths = setup["shuffle_paths"]
    spec = spec_fn()

    engine = CompositionalAnalysisEngine(ScenarioBase(logs))
    rho_comp, eps_comp = engine.check_with_dfa_scenic(
        shuffle_paths, spec, features=["speed"], center_feat_idx=[],
    )

    rho_mono = relabel_traces(mono_shuf_csv, spec)
    n_mono   = pd.read_csv(mono_shuf_csv)["trace_id"].nunique()
    eps_mono = hoeffding_eps(n_mono)

    diff      = abs(rho_comp - rho_mono)
    tolerance = 2.0 * (eps_comp + eps_mono) + 0.15
    print(f"\n  [{spec_name}] shuffle")
    print(f"    Compositional rho = {rho_comp:.4f} +/- {eps_comp:.4f}")
    print(f"    Monolithic    rho = {rho_mono:.4f} +/- {eps_mono:.4f}")
    print(f"    |diff| = {diff:.4f}  tolerance = {tolerance:.4f}")

    assert diff <= tolerance, (
        f"{spec_name} shuffle: |diff|={diff:.4f} > tolerance={tolerance:.4f} "
        f"(comp={rho_comp:.4f}, mono={rho_mono:.4f})"
    )


# ---------------------------------------------------------------------------
# main() for direct execution
# ---------------------------------------------------------------------------

def main(reuse=False):
    print(f"Source: {SCENIC_FILE}")
    print(f"Storage: {STORAGE_DIR}  (N={N_TRACES} per primitive)")

    data = _build_setup(reuse=reuse)
    logs          = data["logs"]
    mono_main_csv = data["mono_main_csv"]
    mono_shuf_csv = data["mono_shuf_csv"]
    choose_paths  = data["choose_paths"]
    shuffle_paths = data["shuffle_paths"]

    engine = CompositionalAnalysisEngine(ScenarioBase(logs))

    n_mono_main = pd.read_csv(mono_main_csv)["trace_id"].nunique()
    n_mono_shuf = pd.read_csv(mono_shuf_csv)["trace_id"].nunique()

    print(f"\n{'spec':<16} {'primitive':>20}  {'rho':>6}")
    print("-" * 46)

    for spec_name, spec_fn in SPECS:
        spec = spec_fn()
        for p, csv in logs.items():
            rho = relabel_traces(csv, spec)
            print(f"  {spec_name:<14} {p:>20}  {rho:.4f}")

        rho_c, eps_c = engine.check_with_dfa_scenic(
            choose_paths, spec, features=["speed"], center_feat_idx=[])
        rho_s, eps_s = engine.check_with_dfa_scenic(
            shuffle_paths, spec, features=["speed"], center_feat_idx=[])

        rho_mc = relabel_traces(mono_main_csv, spec)
        rho_ms = relabel_traces(mono_shuf_csv, spec)
        eps_mc = hoeffding_eps(n_mono_main)
        eps_ms = hoeffding_eps(n_mono_shuf)

        print(f"\n  [{spec_name}]")
        print(f"    choose  comp={rho_c:.4f}+/-{eps_c:.4f}  "
              f"mono={rho_mc:.4f}+/-{eps_mc:.4f}  |diff|={abs(rho_c-rho_mc):.4f}")
        print(f"    shuffle comp={rho_s:.4f}+/-{eps_s:.4f}  "
              f"mono={rho_ms:.4f}+/-{eps_ms:.4f}  |diff|={abs(rho_s-rho_ms):.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", action="store_true",
                        help="Reuse existing CSVs in storage_paper/ if present")
    args = parser.parse_args()
    main(reuse=args.reuse)
