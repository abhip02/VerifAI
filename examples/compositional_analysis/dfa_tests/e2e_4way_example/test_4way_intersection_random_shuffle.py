"""Random (do choose) and shuffle (do shuffle) composition operators.

Reuses per-primitive CSVs from test_4way_intersection_scenarios.py.
MonolithicShuffle traces are generated here (MetaDrive required for first run).

Both compositions are parsed directly from composed_scenarios.scenic:

  Main (do choose):
      Subscenario1 ; do choose { Sub2L:1, Sub2R:1, Sub2S:1 }
      2-step: Sub1 then one randomly-chosen Sub2 variant.
      Monolithic: MonolithicMain (already generated).

  ShuffleMain (do shuffle):
      Subscenario1 ; do shuffle { Sub2L:1, Sub2R:1, Sub2S:1 }
      4-step: Sub1 then ALL THREE Sub2 variants in a random ORDER.
      Engine averages rho over all 3! = 6 permutations.
      Monolithic: MonolithicShuffle (generated here, 340 ticks).

Spec: safe_under_max (speed <= 5.5 m/s post-warmup, absorbing-reject).
"""
import sys
from pathlib import Path
import argparse

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
from verifai.scenic_composition_analysis import analyze_scenic_composition, build_partner_format
from verifai.scenic_parser import parse_scenic_spec
from verifai.generate_graph_traces import generate_graph_scenarios

SCENIC_FILE      = HERE / "4_way_intersection_scenic" / "composed_scenarios.scenic"
STORAGE_DIR      = HERE / "storage_scenarios"
MAX_STEPS        = 85
MONO_SHUFFLE_N   = 300
MONO_SHUFFLE_STEPS = MAX_STEPS * 4   # 340: Sub1 + 3×Sub2

WARMUP_STEPS = 25
MAX_SPEED    = 5.5


def make_spec():
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
    # --- Parse compositions from Scenic file ---
    graph    = analyze_scenic_composition(SCENIC_FILE)
    partner  = build_partner_format(graph)
    all_paths = parse_scenic_spec(partner)

    random_paths  = all_paths["Main"]        # do choose  -> random pick
    shuffle_paths = all_paths["ShuffleMain"] # do shuffle -> all permutations

    print(f"Main       : {random_paths}")
    print(f"ShuffleMain: {shuffle_paths}")

    # --- Per-primitive CSVs (must already exist) ---
    primitives = ["Subscenario1", "Subscenario2L", "Subscenario2R", "Subscenario2S"]
    logs = {}
    for p in primitives:
        csv = STORAGE_DIR / p / "traces.csv"
        if not csv.exists():
            raise FileNotFoundError(
                f"Missing {csv} — run test_4way_intersection_scenarios.py first"
            )
        logs[p] = str(csv)

    spec   = make_spec()
    engine = CompositionalAnalysisEngine(ScenarioBase(logs))

    print("\n=== Per-primitive rho ===")
    for p in primitives:
        rho = relabel_traces(logs[p], spec)
        print(f"  {p:22s} rho = {rho:.4f}")

    # --- Compositional rho ---
    rho_rand, eps_rand = engine.check_with_dfa_scenic(
        random_paths, spec, features=["speed"], center_feat_idx=[],
    )
    rho_shuf, eps_shuf = engine.check_with_dfa_scenic(
        shuffle_paths, spec, features=["speed"], center_feat_idx=[],
    )

    # --- Monolithic: random (MonolithicMain, already generated) ---
    mono_main_csv = STORAGE_DIR / "MonolithicMain" / "traces.csv"
    if mono_main_csv.exists():
        rho_mono_rand = relabel_traces(str(mono_main_csv), spec)
        n_mono_rand   = pd.read_csv(mono_main_csv)["trace_id"].nunique()
        eps_mono_rand = hoeffding_eps(n_mono_rand)
    else:
        rho_mono_rand, eps_mono_rand = None, None

    # --- Monolithic: shuffle (MonolithicShuffle) ---
    mono_shuf_csv = STORAGE_DIR / "MonolithicShuffle" / "traces.csv"
    if reuse_traces and mono_shuf_csv.exists():
        print(f"\n[reuse] MonolithicShuffle: {mono_shuf_csv}")
    else:
        if reuse_traces:
            print(f"\n[reuse] MonolithicShuffle CSV missing, regenerating")
        shuf_logs = generate_graph_scenarios(
            SCENIC_FILE, ["MonolithicShuffle"],
            n=MONO_SHUFFLE_N, save_dir=STORAGE_DIR, max_steps=MONO_SHUFFLE_STEPS,
        )
        mono_shuf_csv = Path(shuf_logs["MonolithicShuffle"])

    rho_mono_shuf = relabel_traces(str(mono_shuf_csv), spec)
    n_mono_shuf   = pd.read_csv(mono_shuf_csv)["trace_id"].nunique()
    eps_mono_shuf = hoeffding_eps(n_mono_shuf)

    # --- Side-by-side table ---
    def _fmt(rho, eps):
        return f"{rho:.4f} +/- {eps:.4f}"

    print("\n=== Compositional vs Monolithic ===")
    print(f"  {'composition':<34s}  {'comp rho':>16s}  {'mono rho':>16s}  {'|diff|':>7s}")
    print("  " + "-" * 80)

    if rho_mono_rand is not None:
        print(f"  {'Main  (do choose, 2-step)':<34s}  "
              f"{_fmt(rho_rand, eps_rand):>16s}  "
              f"{_fmt(rho_mono_rand, eps_mono_rand):>16s}  "
              f"{abs(rho_rand - rho_mono_rand):>7.4f}")
    else:
        print(f"  {'Main  (do choose, 2-step)':<34s}  "
              f"{_fmt(rho_rand, eps_rand):>16s}  "
              f"{'(no MonolithicMain)':>16s}  {'—':>7s}")

    print(f"  {'ShuffleMain (do shuffle, 4-step)':<34s}  "
          f"{_fmt(rho_shuf, eps_shuf):>16s}  "
          f"{_fmt(rho_mono_shuf, eps_mono_shuf):>16s}  "
          f"{abs(rho_shuf - rho_mono_shuf):>7.4f}")

    print(f"\n  shuffle rho < random rho?  comp {rho_shuf:.4f} < {rho_rand:.4f} = "
          f"{rho_shuf < rho_rand}  (4 segments vs 2)")


def test_4way_intersection_random_shuffle():
    main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse_traces", action="store_true",
                        help="Skip MonolithicShuffle generation if CSV exists")
    args = parser.parse_args()
    main(reuse_traces=args.reuse_traces)
