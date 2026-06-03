"""
Test check_with_dfa with a "no more than one near-stop" DFA spec
across multiple scenario combinations.

DFA: moving --near_stop--> stopped_once --near_stop--> stopped_twice (fail)

Primitives:  S, X, C, O
Combinations:
    SX    : S → X              (2-step)
    SXS   : S → X → S          (3-step)
    SOC   : S → O → C          (3-step)
    CSXS  : C → S → X → S      (4-step)
    CXSXC : C → X → S → X → C  (5-step)

Usage: pytest test_check_with_dfa_metadrive_two_stops.py -s
"""

import os

import pandas as pd
import pytest

from verifai.monitor import automaton_specification
from verifai.compositional_analysis import ScenarioBase, CompositionalAnalysisEngine, relabel_traces

NEAR_STOP_MS = 3.5
N_EPISODES   = 1000
TRACE_DIR    = os.path.join(os.path.dirname(__file__), "dfa_test_traces_two_stops")

# Primitives: name → generation seed
PRIMITIVES = {"S": 0, "X": 1, "C": 2, "O": 8}

# Monolithic ground-truth scenarios: name → generation seed
MONOLITHICS = {"SX": 3, "SXS": 4, "SOC": 10, "CSXS": 11, "CXSXC": 12}

# (monolithic_name, compositional_path) pairs to test
COMBINATIONS = [
    ("SX",    ["S", "X"]),
    ("SXS",   ["S", "X", "S"]),
    ("SOC",   ["S", "O", "C"]),
    ("CSXS",  ["C", "S", "X", "S"]),
    ("CXSXC", ["C", "X", "S", "X", "C"]),
]


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


def generate(scenario, seed):
    from utils import generate_traces
    generate_traces(seed=seed, save_dir=TRACE_DIR, model_path=None,
                    expert=True, n=N_EPISODES, scenario=scenario, gif=False)
    return os.path.join(TRACE_DIR, scenario, "traces.csv")


@pytest.fixture(scope="module")
def setup():
    os.makedirs(TRACE_DIR, exist_ok=True)
    spec = make_spec()
    paths = {}

    for name, seed in PRIMITIVES.items():
        csv = generate(name, seed)
        relabel_traces(csv, spec)
        paths[name] = csv
        rho = pd.read_csv(csv).groupby("trace_id")["label"].last().astype(float).mean()
        print(f"  [primitive]  {name}: rho={rho:.4f}")

    for name, seed in MONOLITHICS.items():
        csv = generate(name, seed)
        relabel_traces(csv, spec)
        paths[name] = csv
        rho = pd.read_csv(csv).groupby("trace_id")["label"].last().astype(float).mean()
        print(f"  [monolithic] {name}: rho={rho:.4f}")

    return paths, spec


@pytest.mark.parametrize("mono_name,comp_path", COMBINATIONS)
def test_compositional_vs_monolithic(setup, mono_name, comp_path):
    paths, spec = setup

    mono_base = ScenarioBase({mono_name: paths[mono_name]})
    rho_mono  = mono_base.get_success_prob(mono_name)
    eps_mono  = mono_base.get_success_prob_uncertainty(mono_name)

    prim_paths = {p: paths[p] for p in set(comp_path)}
    engine = CompositionalAnalysisEngine(ScenarioBase(prim_paths))
    rho_comp, eps_comp = engine.check_with_dfa(
        comp_path, spec, features=["x", "y", "speed"], center_feat_idx=[0, 1],
    )

    label = f"{'→'.join(comp_path)}"
    print(f"\n  [{label}]")
    print(f"    Monolithic    rho({mono_name})  = {rho_mono:.4f} +/- {eps_mono:.4f}")
    print(f"    Compositional rho({label}) = {rho_comp:.4f} +/- {eps_comp:.4f}")

    diff = abs(rho_comp - rho_mono)
    tolerance = 2.0 * (eps_mono + eps_comp) + 0.15
    assert diff <= tolerance, (
        f"|diff| = {diff:.4f} > tolerance {tolerance:.4f}  "
        f"(mono={rho_mono:.4f}, comp={rho_comp:.4f})"
    )
