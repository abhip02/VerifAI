"""
Test check_with_dfa with a "no more than one near-stop" DFA spec.

DFA: moving --near_stop--> stopped_once --near_stop--> stopped_twice (fail)

Usage: pytest test_check_with_dfa_metadrive_two_stops.py -s
"""

import os

import numpy as np
import pandas as pd
import pytest

from verifai.monitor import automaton_specification
from verifai.compositional_analysis import ScenarioBase, CompositionalAnalysisEngine

NEAR_STOP_MS = 3.5
N_EPISODES = 500
TRACE_DIR = os.path.join(os.path.dirname(__file__), "dfa_test_traces_two_stops")


def make_spec():
    def transition(state, sym):
        if state == "moving":
            return "stopped_once" if sym == "near_stop" else "moving"
        if state == "stopped_once":
            return "stopped_twice" if sym == "near_stop" else "stopped_once"
        return "stopped_twice"

    return automaton_specification(
        start="moving",
        inputs={"moving", "near_stop"},
        transition=transition,
        label=lambda s: s != "stopped_twice",
        labeling_function=lambda row: "near_stop" if row["speed"] < NEAR_STOP_MS else "moving",
    )


def generate(scenario, seed=0):
    from utils import generate_traces
    generate_traces(seed=seed, save_dir=TRACE_DIR, model_path=None,
                    expert=True, n=N_EPISODES, scenario=scenario, gif=False)
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


def test_compositional_vs_monolithic(setup):
    paths, spec = setup

    rho_mono = ScenarioBase({"SX": paths["SX"]}).get_success_prob("SX")
    eps_mono = ScenarioBase({"SX": paths["SX"]}).get_success_prob_uncertainty("SX")

    engine = CompositionalAnalysisEngine(ScenarioBase({"S": paths["S"], "X": paths["X"]}))
    rho_comp, eps_comp = engine.check_with_dfa(
        ["S", "X"], spec, features=["x", "y", "speed"], center_feat_idx=[0, 1],
    )

    print(f"\n  Monolithic    rho(SX)  = {rho_mono:.4f} +/- {eps_mono:.4f}")
    print(f"  Compositional rho(S*X) = {rho_comp:.4f} +/- {eps_comp:.4f}")

    diff = abs(rho_comp - rho_mono)
    tolerance = 2.0 * (eps_mono + eps_comp) + 0.15
    assert diff <= tolerance, f"|diff| = {diff:.4f} > tolerance {tolerance:.4f}"
    