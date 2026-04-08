"""
Integration test for check_with_dfa using real MetaDrive trace generation.

Spec (non-Markovian, safety):
    "Once the vehicle exceeds 15 m/s, it must NEVER drop below 6.5 m/s
     for the remainder of the scenario."

    Violation is absorbing — once the DFA enters "violated", the trace fails.

DFA:
    cruising (accepting)  --speeding--> monitoring (accepting)
    cruising              --else------> cruising
    monitoring (accepting)--too_slow--> violated   (rejecting, absorbing)
    monitoring            --else------> monitoring
    violated   (rejecting)--any-------> violated

Scenarios:
    S  (Straight road)
    X  (Intersection)
    SX (Straight + Intersection, monolithic ground truth)

Usage:
    pytest test_check_with_dfa_metadrive.py -s
    python test_check_with_dfa_metadrive.py
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / ".." / ".." / ".." / "src"
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))

UTILS_DIR = PROJECT_ROOT / ".."
if UTILS_DIR.is_dir():
    sys.path.insert(0, str(UTILS_DIR.resolve()))

from verifai.monitor import automaton_specification
from verifai.compositional_analysis import ScenarioBase, CompositionalAnalysisEngine

SPEED_LIMIT_MS = 15.0
SLOW_THRESHOLD_MS = 6.5
N_EPISODES = 10000
DEFAULT_TRACE_DIR = os.path.join(os.path.dirname(__file__), "dfa_test_traces")


def make_safety_spec():
    """
    Non-Markovian safety property:
        "Once the vehicle exceeds SPEED_LIMIT_MS, it must never drop
         below SLOW_THRESHOLD_MS for the rest of the scenario."
    """
    def transition(state, sym):
        if state == "cruising":
            return "monitoring" if sym == "speeding" else "cruising"
        if state == "monitoring":
            return "violated" if sym == "too_slow" else "monitoring"
        return "violated"

    return automaton_specification(
        start="cruising",
        inputs={"normal", "speeding", "too_slow"},
        transition=transition,
        label=lambda s: s != "violated",
        labeling_function=lambda row: (
            "speeding" if row["speed"] > SPEED_LIMIT_MS
            else "too_slow" if row["speed"] < SLOW_THRESHOLD_MS
            else "normal"
        ),
    )


def _generate_traces_for_scenario(
    scenario: str,
    save_dir: str,
    n_episodes: int = N_EPISODES,
    seed: int = 0,
) -> str:
    from utils import generate_traces

    generate_traces(
        seed=seed,
        save_dir=save_dir,
        model_path=None,
        expert=True,
        n=n_episodes,
        scenario=scenario,
        gif=False,
    )

    csv_path = os.path.join(save_dir, scenario, "traces.csv")
    assert os.path.exists(csv_path), f"Trace CSV not created at {csv_path}"
    return csv_path


def _relabel_traces_with_dfa(csv_path: str, spec: automaton_specification) -> str:
    """
    Re-evaluate every trace against the DFA and overwrite the 'label'
    column with DFA acceptance.
    """
    df = pd.read_csv(csv_path)
    df["trace_id"] = df["trace_id"].astype(str)

    new_labels = {}
    for tid, group in df.sort_values("step").groupby("trace_id"):
        traj = group.to_dict("records")
        word = [spec.L(row) for row in traj]
        new_labels[tid] = spec._dfa.label(word)

    df["label"] = df["trace_id"].map(new_labels)
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="module")
def trace_dir():
    d = DEFAULT_TRACE_DIR
    os.makedirs(d, exist_ok=True)
    print(f"\nTraces saved to: {os.path.abspath(d)}")
    yield d


@pytest.fixture(scope="module")
def trace_paths_and_spec(trace_dir):
    """
    Generate MetaDrive traces for S, X, and SX, build spec, and relabel.
    Returns (paths_dict, spec).
    """
    csv_s = _generate_traces_for_scenario("S", trace_dir, N_EPISODES, seed=0)
    csv_x = _generate_traces_for_scenario("X", trace_dir, N_EPISODES, seed=1)
    csv_sx = _generate_traces_for_scenario("SX", trace_dir, N_EPISODES, seed=2)

    spec = make_safety_spec()

    _relabel_traces_with_dfa(csv_s, spec)
    _relabel_traces_with_dfa(csv_x, spec)
    _relabel_traces_with_dfa(csv_sx, spec)

    paths = {"S": csv_s, "X": csv_x, "SX": csv_sx}

    for name, path in paths.items():
        df = pd.read_csv(path)
        n = df["trace_id"].nunique()
        rho = df.groupby("trace_id")["label"].last().astype(float).mean()
        print(f"  {name}: {n} traces, rho={rho:.4f}")

    return paths, spec


@pytest.fixture(scope="module")
def trace_paths(trace_paths_and_spec):
    return trace_paths_and_spec[0]


@pytest.fixture(scope="module")
def spec(trace_paths_and_spec):
    return trace_paths_and_spec[1]


def test_traces_have_required_columns(trace_paths):
    for name, path in trace_paths.items():
        df = pd.read_csv(path)
        missing = {"trace_id", "step", "label"} - set(df.columns)
        assert not missing, f"{name} missing columns: {missing}"


def test_traces_have_feature_columns(trace_paths):
    for name, path in trace_paths.items():
        df = pd.read_csv(path)
        missing = {"x", "y", "speed"} - set(df.columns)
        assert not missing, f"{name} missing feature columns: {missing}"


def test_each_scenario_has_enough_traces(trace_paths):
    for name, path in trace_paths.items():
        df = pd.read_csv(path)
        n = df["trace_id"].nunique()
        assert n >= 2, f"{name} has only {n} trace(s); need >=2 for KDE"


def test_rho_in_valid_range(trace_paths):
    for name, path in trace_paths.items():
        df = pd.read_csv(path)
        rho = df.groupby("trace_id")["label"].last().astype(float).mean()
        assert 0.0 <= rho <= 1.0, f"{name}: rho={rho} out of [0,1]"


def test_check_with_dfa_single_scenario(trace_paths, spec):
    """
    For a single scenario starting from q0, check_with_dfa should exactly
    match the empirical DFA-relabeled success rate.
    """
    for name in ["S", "X"]:
        df = pd.read_csv(trace_paths[name])
        rho_empirical = (
            df.groupby("trace_id")["label"].last().astype(float).mean()
        )

        sb = ScenarioBase({name: trace_paths[name]})
        engine = CompositionalAnalysisEngine(sb)
        rho_dfa, eps_dfa = engine.check_with_dfa(
            [name],
            spec,
            features=["x", "y", "speed"],
            center_feat_idx=[0, 1],
        )

        print(f"\n  Single-scenario {name}:")
        print(f"    empirical rho = {rho_empirical:.4f}")
        print(f"    DFA       rho = {rho_dfa:.4f} +/- {eps_dfa:.4f}")

        assert abs(rho_dfa - rho_empirical) < 1e-6, (
            f"{name}: DFA rho ({rho_dfa:.6f}) != empirical ({rho_empirical:.6f})"
        )


def test_check_with_dfa_returns_valid_uncertainty(trace_paths, spec):
    sb = ScenarioBase({"S": trace_paths["S"], "X": trace_paths["X"]})
    engine = CompositionalAnalysisEngine(sb)
    rho, eps = engine.check_with_dfa(
        ["S", "X"],
        spec,
        features=["x", "y", "speed"],
        center_feat_idx=[0, 1],
    )
    assert np.isfinite(rho), f"rho is not finite: {rho}"
    assert np.isfinite(eps), f"uncertainty is not finite: {eps}"
    assert eps >= 0.0, f"uncertainty is negative: {eps}"


def test_check_with_dfa_compositional_vs_monolithic(trace_paths, spec):
    """
    Compositional rho from check_with_dfa([S, X]) should be consistent
    with monolithic rho from the directly-generated SX traces, within tolerance.
    """
    sb_mono = ScenarioBase({"SX": trace_paths["SX"]})
    rho_mono = sb_mono.get_success_prob("SX")
    eps_mono = sb_mono.get_success_prob_uncertainty("SX")

    sb_comp = ScenarioBase({"S": trace_paths["S"], "X": trace_paths["X"]})
    engine = CompositionalAnalysisEngine(sb_comp)
    rho_comp, eps_comp = engine.check_with_dfa(
        ["S", "X"],
        spec,
        features=["x", "y", "speed"],
        center_feat_idx=[0, 1],
    )

    print("\n  check_with_dfa -- MetaDrive integration")
    for name in ["S", "X", "SX"]:
        df = pd.read_csv(trace_paths[name])
        rho_i = df.groupby("trace_id")["label"].last().astype(float).mean()
        n_i = df["trace_id"].nunique()
        print(f"  {name:>3s}: n={n_i:4d}  rho={rho_i:.4f}")

    print(f"\n  Monolithic    rho(SX)  = {rho_mono:.4f} +/- {eps_mono:.4f}")
    print(f"  Compositional rho(S*X) = {rho_comp:.4f} +/- {eps_comp:.4f}")

    tolerance = 2.0 * (eps_mono + eps_comp) + 0.15
    diff = abs(rho_comp - rho_mono)

    print(f"\n  |diff| = {diff:.4f},  tolerance = {tolerance:.4f}")
    assert diff <= tolerance, (
        f"Estimates diverge: |{rho_comp:.4f} - {rho_mono:.4f}| = "
        f"{diff:.4f} > {tolerance:.4f}"
    )
    print("  PASS: Consistent.\n")


def test_check_with_dfa_rho_nonzero_when_traces_pass(trace_paths, spec):
    """
    If at least some traces satisfy the spec in both S and X, the
    compositional rho should be strictly positive.
    """
    has_passing_s = (
        pd.read_csv(trace_paths["S"])
        .groupby("trace_id")["label"].last().astype(float).sum() > 0
    )
    has_passing_x = (
        pd.read_csv(trace_paths["X"])
        .groupby("trace_id")["label"].last().astype(float).sum() > 0
    )

    if has_passing_s and has_passing_x:
        sb = ScenarioBase({"S": trace_paths["S"], "X": trace_paths["X"]})
        engine = CompositionalAnalysisEngine(sb)
        rho, _ = engine.check_with_dfa(
            ["S", "X"],
            spec,
            features=["x", "y", "speed"],
            center_feat_idx=[0, 1],
        )
        assert rho > 0.0, "Expected rho > 0 when both scenarios have passing traces"
    else:
        pytest.skip("Not enough passing traces to test positivity")


if __name__ == "__main__":
    print("Generating MetaDrive traces (this may take a minute) ...\n")

    trace_dir = DEFAULT_TRACE_DIR
    os.makedirs(trace_dir, exist_ok=True)
    print(f"Traces will be saved to: {os.path.abspath(trace_dir)}\n")

    csv_s = _generate_traces_for_scenario("S", trace_dir, N_EPISODES, seed=0)
    csv_x = _generate_traces_for_scenario("X", trace_dir, N_EPISODES, seed=1)
    csv_sx = _generate_traces_for_scenario("SX", trace_dir, N_EPISODES, seed=2)

    _spec = make_safety_spec()

    _relabel_traces_with_dfa(csv_s, _spec)
    _relabel_traces_with_dfa(csv_x, _spec)
    _relabel_traces_with_dfa(csv_sx, _spec)

    paths = {"S": csv_s, "X": csv_x, "SX": csv_sx}

    print(f"\nTrace CSVs:")
    for name, path in paths.items():
        df = pd.read_csv(path)
        n = df["trace_id"].nunique()
        rho = df.groupby("trace_id")["label"].last().astype(float).mean()
        print(f"  {name}: {path}  ({n} traces, rho={rho:.4f})")

    print()
    test_traces_have_required_columns(paths)
    test_traces_have_feature_columns(paths)
    test_each_scenario_has_enough_traces(paths)
    test_rho_in_valid_range(paths)
    test_check_with_dfa_single_scenario(paths, _spec)
    test_check_with_dfa_returns_valid_uncertainty(paths, _spec)
    test_check_with_dfa_compositional_vs_monolithic(paths, _spec)
    test_check_with_dfa_rho_nonzero_when_traces_pass(paths, _spec)

    print("\nAll tests passed.")
    print(f"Traces retained at: {os.path.abspath(trace_dir)}")