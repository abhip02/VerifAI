"""
Scenic composition primitives smoke-test using MetaDrive.

Four tests, one per Scenic composition primitive:
  1. Sequential        : ["S", "C"]
  2. Random choice     : ["S", {"C": 0.6, "X": 0.4}]
  3. Shuffle           : {"shuffle": {"S": 1, "C": 1}}  → [(0.5, ["S","C"]), (0.5, ["C","S"])]
  4. Multi-step branch : {Branch:[S→X], C}              → [(0.5, ["S","X"]), (0.5, ["C"])]

Each compositional rho is compared against the weighted monolithic rho.
The spec is the same tollgate mandatory-wait DFA (K=3, threshold=3.5 m/s).

MetaDrive map strings: S=Straight, C=Curve, X=Intersection.
Concatenating them gives the monolithic map (e.g. "SC" = straight then curve).

Usage:
    pytest test_scenic_primitives_metadrive.py -s
"""

import os
import numpy as np
import pandas as pd
import pytest

from verifai.monitor import automaton_specification
from verifai.compositional_analysis import ScenarioBase, CompositionalAnalysisEngine
from verifai.scenic_parser import scenic_to_check_input

STOP_THRESHOLD_MS = 3.5
REQUIRED_WAIT_STEPS = 3
N_EPISODES = 100
TRACE_DIR = os.path.join(os.path.dirname(__file__), "storage", "scenic_primitives")

FEATURES = ["x", "y", "speed"]
CENTER_FEAT_IDX = [0, 1]

# Cap obstacle distance so each primitive always completes its mandatory wait
# before the scenario ends — eliminates wait_* DFA states at handoff boundaries.
MAX_OBS_DIST = 15.0  # metres

# Choice weights for test 2
P_C = 0.6
P_X = 0.4


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate(scenario, seed=0):
    from utils import generate_traces
    generate_traces(seed=seed, save_dir=TRACE_DIR, expert=True,
                    n=N_EPISODES, scenario=scenario, extra_obstacles=True,
                    max_obstacle_distance=MAX_OBS_DIST)
    return os.path.join(TRACE_DIR, scenario, "traces.csv")


def relabel(csv_path, spec):
    df = pd.read_csv(csv_path).sort_values("step")
    labels = {tid: spec.evaluate(grp.to_dict("records")) > 0
              for tid, grp in df.groupby("trace_id")}
    df["label"] = df["trace_id"].map(labels)
    df.to_csv(csv_path, index=False)


def rho_of(csv_path):
    return pd.read_csv(csv_path).groupby("trace_id")["label"].last().astype(float).mean()


def eps_of(csv_path, delta=0.05):
    n = pd.read_csv(csv_path)["trace_id"].nunique()
    return np.sqrt(np.log(2 / delta) / (2 * n))


def tolerance(eps_mono, eps_comp):
    return 2.0 * (eps_mono + eps_comp) + 0.15


# ---------------------------------------------------------------------------
# Fixture: generate + relabel all traces once per test session
#
# Primitives : S, C, X  (single-segment MetaDrive maps)
# Monoliths  : SC, SX, CS  (concatenated maps for ground-truth comparison)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def setup():
    spec = make_spec()

    scenarios = {
        # primitives
        "S":  ("S",  0),
        "C":  ("C",  1),
        "X":  ("X",  2),
        # monoliths
        "SC": ("SC", 3),   # sequential S→C
        "SX": ("SX", 4),   # sequential S→X  /  multi-step branch leg
        "CS": ("CS", 5),   # shuffle C,S order
    }

    paths = {}
    print()
    for name, (scenario_str, seed) in scenarios.items():
        csv = generate(scenario_str, seed)
        relabel(csv, spec)
        paths[name] = csv
        print(f"  {name:3s}: rho={rho_of(csv):.4f}")

    return paths, spec


# ---------------------------------------------------------------------------
# Test 1: Sequential  S → C
#
# Compositional: check_with_dfa(["S", "C"], ...)
# Monolithic:    rho("SC")
# ---------------------------------------------------------------------------

def test_sequential(setup):
    paths, spec = setup

    engine = CompositionalAnalysisEngine(
        ScenarioBase({"S": paths["S"], "C": paths["C"]})
    )
    rho_comp, eps_comp = engine.check_with_dfa(
        ["S", "C"], spec,
        features=FEATURES, center_feat_idx=CENTER_FEAT_IDX,
    )

    rho_mono = rho_of(paths["SC"])
    eps_mono = eps_of(paths["SC"])

    print(f"\n[sequential]  mono={rho_mono:.4f}±{eps_mono:.4f}  "
          f"comp={rho_comp:.4f}±{eps_comp:.4f}")

    assert abs(rho_comp - rho_mono) <= tolerance(eps_mono, eps_comp), (
        f"|diff|={abs(rho_comp-rho_mono):.4f} > tol={tolerance(eps_mono,eps_comp):.4f}"
    )


# ---------------------------------------------------------------------------
# Test 2: Random choice  S → {C: 0.6, X: 0.4}
#
# Compositional: check_with_dfa(["S", {"C": 0.6, "X": 0.4}], ...)
# Monolithic:    0.6·rho("SC") + 0.4·rho("SX")
# ---------------------------------------------------------------------------

def test_random_choice(setup):
    paths, spec = setup

    engine = CompositionalAnalysisEngine(
        ScenarioBase({"S": paths["S"], "C": paths["C"], "X": paths["X"]})
    )
    rho_comp, eps_comp = engine.check_with_dfa(
        ["S", {"C": P_C, "X": P_X}], spec,
        features=FEATURES, center_feat_idx=CENTER_FEAT_IDX,
    )

    rho_mono = P_C * rho_of(paths["SC"]) + P_X * rho_of(paths["SX"])
    eps_mono = np.sqrt((P_C * eps_of(paths["SC"])) ** 2 +
                       (P_X * eps_of(paths["SX"])) ** 2)

    print(f"\n[random_choice]  mono={rho_mono:.4f}±{eps_mono:.4f}  "
          f"comp={rho_comp:.4f}±{eps_comp:.4f}")

    assert abs(rho_comp - rho_mono) <= tolerance(eps_mono, eps_comp), (
        f"|diff|={abs(rho_comp-rho_mono):.4f} > tol={tolerance(eps_mono,eps_comp):.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3: Shuffle  {S, C}  (each executed once, order uniformly random)
#
# Scenic graph:  {"shuffle": {"S": 1, "C": 1}}
# Expands to:    [(0.5, ["S","C"]), (0.5, ["C","S"])]
#
# Compositional: check_with_dfa_scenic(paths, ...)
# Monolithic:    0.5·rho("SC") + 0.5·rho("CS")
# ---------------------------------------------------------------------------

def test_shuffle(setup):
    paths, spec = setup

    shuffle_graph = {
        "entrypoints": ["Main"],
        "containers": {
            "Main": {"kind": "scenario",
                     "steps": [{"shuffle": {"S": 1, "C": 1}}]},
            "S": {"kind": "behavior", "steps": []},
            "C": {"kind": "behavior", "steps": []},
        },
    }
    scenic_paths = scenic_to_check_input(shuffle_graph)

    engine = CompositionalAnalysisEngine(
        ScenarioBase({"S": paths["S"], "C": paths["C"]})
    )
    rho_comp, eps_comp = engine.check_with_dfa_scenic(
        scenic_paths, spec,
        features=FEATURES, center_feat_idx=CENTER_FEAT_IDX,
    )

    rho_mono = 0.5 * rho_of(paths["SC"]) + 0.5 * rho_of(paths["CS"])
    eps_mono = np.sqrt((0.5 * eps_of(paths["SC"])) ** 2 +
                       (0.5 * eps_of(paths["CS"])) ** 2)

    print(f"\n[shuffle]  mono={rho_mono:.4f}±{eps_mono:.4f}  "
          f"comp={rho_comp:.4f}±{eps_comp:.4f}")
    print(f"  scenic_paths: {scenic_paths}")

    assert abs(rho_comp - rho_mono) <= tolerance(eps_mono, eps_comp), (
        f"|diff|={abs(rho_comp-rho_mono):.4f} > tol={tolerance(eps_mono,eps_comp):.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4: Multi-step branch  {Branch: 0.5, C: 0.5}
#                             where Branch = S → X  (a two-step sub-scenario)
#
# Scenic graph:  random choice between the multi-step "Branch" and bare "C"
# Expands to:    [(0.5, ["S","X"]), (0.5, ["C"])]
#
# Compositional: check_with_dfa_scenic(paths, ...)
# Monolithic:    0.5·rho("SX") + 0.5·rho("C")
# ---------------------------------------------------------------------------

def test_multi_step_branch(setup):
    paths, spec = setup

    multi_branch_graph = {
        "entrypoints": ["Main"],
        "containers": {
            "Main": {
                "kind": "scenario",
                "steps": [{"Branch": 1, "C": 1}],
            },
            "Branch": {"kind": "scenario", "steps": ["S", "X"]},
            "S": {"kind": "behavior", "steps": []},
            "X": {"kind": "behavior", "steps": []},
            "C": {"kind": "behavior", "steps": []},
        },
    }
    scenic_paths = scenic_to_check_input(multi_branch_graph)

    engine = CompositionalAnalysisEngine(
        ScenarioBase({"S": paths["S"], "X": paths["X"], "C": paths["C"]})
    )
    rho_comp, eps_comp = engine.check_with_dfa_scenic(
        scenic_paths, spec,
        features=FEATURES, center_feat_idx=CENTER_FEAT_IDX,
    )

    rho_mono = 0.5 * rho_of(paths["SX"]) + 0.5 * rho_of(paths["C"])
    eps_mono = np.sqrt((0.5 * eps_of(paths["SX"])) ** 2 +
                       (0.5 * eps_of(paths["C"])) ** 2)

    print(f"\n[multi_step_branch]  mono={rho_mono:.4f}±{eps_mono:.4f}  "
          f"comp={rho_comp:.4f}±{eps_comp:.4f}")
    print(f"  scenic_paths: {scenic_paths}")

    assert abs(rho_comp - rho_mono) <= tolerance(eps_mono, eps_comp), (
        f"|diff|={abs(rho_comp-rho_mono):.4f} > tol={tolerance(eps_mono,eps_comp):.4f}"
    )
