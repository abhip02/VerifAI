"""
Tollgate test: non-Markovian "mandatory wait" spec verified with check_with_dfa.

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

Usage: pytest test_check_with_dfa_metadrive_tollgate.py -s
"""

import os
import pandas as pd
import pytest

from verifai.monitor import automaton_specification
from verifai.compositional_analysis import ScenarioBase, CompositionalAnalysisEngine

STOP_THRESHOLD_MS = 3.5
REQUIRED_WAIT_STEPS = 3
N_EPISODES = 100
TRACE_DIR = os.path.join(os.path.dirname(__file__), "storage", "tollgate")


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


def generate(scenario, seed=0):
    from utils import generate_traces
    generate_traces(seed=seed, save_dir=TRACE_DIR, expert=True,
                    n=N_EPISODES, scenario=scenario, extra_obstacles=True)
    return os.path.join(TRACE_DIR, scenario, "traces.csv")


def relabel(csv_path, spec):
    df = pd.read_csv(csv_path).sort_values("step")
    labels = {tid: spec.evaluate(grp.to_dict("records")) > 0
              for tid, grp in df.groupby("trace_id")}
    df["label"] = df["trace_id"].map(labels)
    df.to_csv(csv_path, index=False)


@pytest.fixture(scope="module")
def setup():
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
