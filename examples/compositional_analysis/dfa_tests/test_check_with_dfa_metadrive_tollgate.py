"""
Tollgate test: non-Markovian "mandatory wait" spec verified with check_with_dfa
across multiple scenario combinations.

Spec: once the vehicle slows below STOP_THRESHOLD_MS, it must remain slow
for at least REQUIRED_WAIT_STEPS consecutive steps before speeding up again.
Resuming early is a violation.

DFA (K=3):
    moving   --slow--> wait_1
    wait_1   --slow--> wait_2
    wait_2   --slow--> wait_3
    wait_3   --any --> moving   (wait complete, back to normal)
    wait_*   --fast--> violated (absorbing)
    moving   --fast--> moving

Primitives:  S, X, C
Combinations:
    SX   : S → X          (2-step)
    SXS  : S → X → S      (3-step)
    SC   : S → C          (2-step)
    SCS  : S → C → S      (3-step)
    SCX  : S → C → X      (3-step)

Usage: pytest test_check_with_dfa_metadrive_tollgate.py -s
"""

import os
import pandas as pd
import pytest

from verifai.monitor import automaton_specification
from verifai.compositional_analysis import ScenarioBase, CompositionalAnalysisEngine, relabel_traces

STOP_THRESHOLD_MS   = 3.5
REQUIRED_WAIT_STEPS = 3
N_EPISODES          = 100
TRACE_DIR           = os.path.join(os.path.dirname(__file__), "storage", "tollgate")

# Primitives: name → generation seed
PRIMITIVES = {"S": 0, "X": 1, "C": 2}

# Monolithic ground-truth scenarios: name → generation seed
MONOLITHICS = {"SX": 3, "SXS": 4, "SC": 5, "SCS": 6, "SCX": 7}

# (monolithic_name, compositional_path) pairs to test
COMBINATIONS = [
    ("SX",  ["S", "X"]),
    ("SXS", ["S", "X", "S"]),
    ("SC",  ["S", "C"]),
    ("SCS", ["S", "C", "S"]),
    ("SCX", ["S", "C", "X"]),
]


def make_spec():
    K = REQUIRED_WAIT_STEPS
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
        start="moving",
        inputs={"slow", "fast"},
        transition=transition,
        label=lambda s: s != "violated",
        labeling_function=lambda row: "slow" if row["speed"] < STOP_THRESHOLD_MS else "fast",
    )


def generate(scenario, seed):
    from utils import generate_traces
    generate_traces(seed=seed, save_dir=TRACE_DIR, expert=True,
                    n=N_EPISODES, scenario=scenario, extra_obstacles=True)
    return os.path.join(TRACE_DIR, scenario, "traces.csv")


@pytest.fixture(scope="module")
def setup():
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

    label = "→".join(comp_path)
    print(f"\n  [{label}]")
    print(f"    Monolithic    rho({mono_name})  = {rho_mono:.4f} +/- {eps_mono:.4f}")
    print(f"    Compositional rho({label}) = {rho_comp:.4f} +/- {eps_comp:.4f}")

    diff = abs(rho_comp - rho_mono)
    tolerance = 2.0 * (eps_mono + eps_comp) + 0.15
    assert diff <= tolerance, (
        f"|diff| = {diff:.4f} > tolerance {tolerance:.4f}  "
        f"(mono={rho_mono:.4f}, comp={rho_comp:.4f})"
    )
