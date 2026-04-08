"""
Test check_with_dfa with a tollgate-style "mandatory wait" DFA spec.

Spec: "If the vehicle slows below STOP_THRESHOLD, it must remain below
       that threshold for at least K consecutive timesteps before resuming.
       Resuming early is a violation."

This is genuinely non-Markovian: a single observation of speed > threshold
could be fine (never stopped) or a violation (stopped but didn't wait long
enough). You need the history to tell.

DFA (K=3):
    moving   (accepting) --slow--> wait_1
    moving               --fast--> moving
    wait_1   (accepting) --slow--> wait_2
    wait_1               --fast--> violated
    wait_2   (accepting) --slow--> wait_3
    wait_2               --fast--> violated
    wait_3   (accepting) --any---> moving    (completed wait, reset)
    violated (rejecting) --any---> violated  (absorbing)

Obstacles are added during trace generation to cause varied stopping behavior:
- Traffic lights (red), boxes, and cones
- 4 scenarios: many_close (3-5 obstacles at 10-40 units), few_far (1-3 at 50-100 units),
  mixed (2-4 at 15-70 units), no_stop (0 obstacles)
- 60% chance of obstacles per episode
- Initial speed varies (40-90 km/h) to create diverse stopping patterns

Usage: pytest test_check_with_dfa_tollgate.py -s
"""

import sys, os
from pathlib import Path
import numpy as np

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str((ROOT / ".." / ".." / ".." / "src").resolve()))
sys.path.insert(0, str((ROOT / "..").resolve()))

from verifai.monitor import automaton_specification
from verifai.compositional_analysis import ScenarioBase, CompositionalAnalysisEngine

STOP_THRESHOLD_MS = 3.5
REQUIRED_WAIT_STEPS = 3
N_EPISODES = 800
TRACE_DIR = os.path.join(os.path.dirname(__file__), "dfa_test_traces_tollgate")


def make_spec():
    """
    Tollgate-style mandatory wait: if you stop, you must stay stopped
    for REQUIRED_WAIT_STEPS consecutive steps before moving again.
    """
    wait_states = [f"wait_{i+1}" for i in range(REQUIRED_WAIT_STEPS)]

    def transition(state, sym):
        if state == "moving":
            return "wait_1" if sym == "slow" else "moving"
        if state == "violated":
            return "violated"
        wait_idx = wait_states.index(state)
        if sym == "slow":
            if wait_idx == REQUIRED_WAIT_STEPS - 1:
                return "moving"
            return wait_states[wait_idx + 1]
        else:
            return "violated"

    all_states = {"moving", "violated"} | set(wait_states)

    return automaton_specification(
        start="moving",
        inputs={"slow", "fast"},
        transition=transition,
        label=lambda s: s != "violated",
        labeling_function=lambda row: "slow" if row["speed"] < STOP_THRESHOLD_MS else "fast",
    )


def generate(scenario, seed=0):
    from utils import generate_traces as _generate_traces
    _generate_traces(seed=seed, save_dir=TRACE_DIR, model_path=None,
                     expert=True, n=N_EPISODES, scenario=scenario, gif=False,
                     extra_obstacles=True)
    return os.path.join(TRACE_DIR, scenario, "traces.csv")


def relabel(csv_path, spec):
    df = pd.read_csv(csv_path).sort_values("step")
    labels = {tid: spec.evaluate(grp.to_dict("records")) > 0
              for tid, grp in df.groupby("trace_id")}
    df["label"] = df["trace_id"].map(labels)
    df.to_csv(csv_path, index=False)


@pytest.fixture(scope="module")
def setup():
    os.makedirs(TRACE_DIR, exist_ok=True)
    spec = make_spec()
    paths = {}
    for name, seed in [("S", 0), ("X", 1), ("SX", 2)]:
        csv = generate(name, seed)
        relabel(csv, spec)
        paths[name] = csv
        rho = pd.read_csv(csv).groupby("trace_id")["label"].last().astype(float).mean()
        print(f"  {name}: rho={rho:.4f}")
    return paths, spec


def test_single_scenario_matches_empirical(setup):
    paths, spec = setup
    for name in ["S", "X"]:
        df = pd.read_csv(paths[name])
        rho_empirical = df.groupby("trace_id")["label"].last().astype(float).mean()
        engine = CompositionalAnalysisEngine(ScenarioBase({name: paths[name]}))
        rho_dfa, _ = engine.check_with_dfa(
            [name], spec, features=["x", "y", "speed"], center_feat_idx=[0, 1],
        )
        print(f"  {name}: empirical={rho_empirical:.4f}  dfa={rho_dfa:.4f}")
        assert abs(rho_dfa - rho_empirical) < 1e-6


def test_compositional_vs_monolithic(setup):
    paths, spec = setup

    sb_mono = ScenarioBase({"SX": paths["SX"]})
    rho_mono = sb_mono.get_success_prob("SX")
    eps_mono = sb_mono.get_success_prob_uncertainty("SX")

    engine = CompositionalAnalysisEngine(ScenarioBase({"S": paths["S"], "X": paths["X"]}))
    rho_comp, eps_comp = engine.check_with_dfa(
        ["S", "X"], spec, features=["x", "y", "speed"], center_feat_idx=[0, 1],
    )

    print(f"\n  Monolithic    rho(SX)  = {rho_mono:.4f} +/- {eps_mono:.4f}")
    print(f"  Compositional rho(S*X) = {rho_comp:.4f} +/- {eps_comp:.4f}")

    diff = abs(rho_comp - rho_mono)
    tolerance = 2.0 * (eps_mono + eps_comp) + 0.15
    assert diff <= tolerance, f"|diff| = {diff:.4f} > tolerance {tolerance:.4f}"


if __name__ == "__main__":
    os.makedirs(TRACE_DIR, exist_ok=True)
    spec = make_spec()
    paths = {}
    for name, seed in [("S", 0), ("X", 1), ("SX", 2)]:
        csv = generate(name, seed)
        relabel(csv, spec)
        paths[name] = csv
        rho = pd.read_csv(csv).groupby("trace_id")["label"].last().astype(float).mean()
        print(f"  {name}: rho={rho:.4f}")

    test_single_scenario_matches_empirical((paths, spec))
    test_compositional_vs_monolithic((paths, spec))
    print("\nAll tests passed.")
