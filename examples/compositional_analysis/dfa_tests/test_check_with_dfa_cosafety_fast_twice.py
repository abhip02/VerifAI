"""
Co-safety "V-shape speed" test: non-Markovian 4-state DFA requiring the vehicle
to first reach HIGH_SPEED, then fall to LOW_SPEED, then reach HIGH_SPEED again.

Co-safety spec ("eventually: fast → slow → fast  (V-shape)"):
    DFA with an absorbing-accept sink.  Requires witnessing all three phases
    in order; any individual step is insufficient.

        init        --fast-->  was_fast
        init        --slow-->  init
        was_fast    --slow-->  was_slow
        was_fast    --fast-->  was_fast     (sustain fast, still waiting for dip)
        was_slow    --fast-->  achieved     (absorbing-accept)
        was_slow    --slow-->  was_slow     (deepen the dip, still waiting)
        achieved    --any -->  achieved

    label(s) = s == "achieved"

Safety complement ("never: fast → slow → fast  (no V-shape)"):
    Flip the absorbing sink.  Once the vehicle has been fast, dipped slow,
    and gone fast again, the property is permanently violated.

        ok          --fast-->  was_fast
        ok          --slow-->  ok
        was_fast    --slow-->  was_slow
        was_fast    --fast-->  was_fast
        was_slow    --fast-->  violated     (absorbing-reject)
        was_slow    --slow-->  was_slow
        violated    --any -->  violated

    label(s) = s != "violated"

The co-safety test is marked xfail because check_with_dfa uses a multiplicative
formula that is only correct for safety / absorbing-reject DFAs.  For a
co-safety DFA the conditioning on Sub1 success collapses Sub2 q_init to the
absorbing-accept state, so compositional rho collapses to Sub1 rho.

The safety-complement test behaves correctly and should pass.

Primitives:  S, X, C, O
Combinations:
    SX    : S → X              (2-step)
    SXS   : S → X → S          (3-step)
    SOC   : S → O → C          (3-step)
    CSXS  : C → S → X → S      (4-step)
    CXSXC : C → X → S → X → C  (5-step)

Usage: pytest test_check_with_dfa_cosafety_fast_twice.py -s
"""

import os

import pandas as pd
import pytest

from verifai.monitor import automaton_specification
from verifai.compositional_analysis import (
    ScenarioBase,
    CompositionalAnalysisEngine,
    relabel_traces,
)

HIGH_SPEED_MS = (
    7.0  # m/s  (~25 km/h) — "fast" (low bar so re-acceleration after stop is likely)
)
LOW_SPEED_MS = 3.5  # m/s  (~13 km/h) — "slow" (same threshold as tollgate / two_stops)
N_EPISODES = 1000
TRACE_DIR = os.path.join(os.path.dirname(__file__), "storage", "vshape_speed")

PRIMITIVES = {"S": 0, "X": 1, "C": 2, "O": 8}
MONOLITHICS = {"SX": 3, "SXS": 4, "SOC": 10, "CSXS": 11, "CXSXC": 12}
COMBINATIONS = [
    ("SX", ["S", "X"]),
    ("SXS", ["S", "X", "S"]),
    ("SOC", ["S", "O", "C"]),
    ("CSXS", ["C", "S", "X", "S"]),
    ("CXSXC", ["C", "X", "S", "X", "C"]),
]


def _sym(row):
    if row["speed"] >= HIGH_SPEED_MS:
        return "fast"
    if row["speed"] <= LOW_SPEED_MS:
        return "slow"
    return "mid"


def make_cosafety_spec():
    """Co-safety: 'eventually fast → slow → fast (V-shape)' — absorbing-accept."""

    def transition(state, sym):
        if state == "achieved":
            return "achieved"
        if state == "init":
            return "was_fast" if sym == "fast" else "init"
        if state == "was_fast":
            return "was_slow" if sym == "slow" else "was_fast"
        # state == "was_slow"
        return "achieved" if sym == "fast" else "was_slow"

    return automaton_specification(
        start="init",
        inputs={"slow", "mid", "fast"},
        transition=transition,
        label=lambda s: s == "achieved",
        labeling_function=_sym,
    )


def make_safety_spec():
    """Safety complement: 'never fast → slow → fast' — absorbing-reject."""

    def transition(state, sym):
        if state == "violated":
            return "violated"
        if state == "ok":
            return "was_fast" if sym == "fast" else "ok"
        if state == "was_fast":
            return "was_slow" if sym == "slow" else "was_fast"
        # state == "was_slow"
        return "violated" if sym == "fast" else "was_slow"

    return automaton_specification(
        start="ok",
        inputs={"slow", "mid", "fast"},
        transition=transition,
        label=lambda s: s != "violated",
        labeling_function=_sym,
    )


def generate(scenario, seed):
    from utils import generate_traces

    csv = os.path.join(TRACE_DIR, scenario, "traces.csv")
    if not os.path.exists(csv):
        # No extra_obstacles: vehicles follow natural speed profiles so S endings
        # and X/C starts share the same speed regime (11-25 m/s), giving the KDE
        # good overlap for importance sampling.
        generate_traces(
            seed=seed, save_dir=TRACE_DIR, expert=True, n=N_EPISODES, scenario=scenario
        )
    return csv


@pytest.fixture(scope="module")
def setup():
    os.makedirs(TRACE_DIR, exist_ok=True)
    cosafety_spec = make_cosafety_spec()
    safety_spec = make_safety_spec()
    paths = {}

    for name, seed in PRIMITIVES.items():
        paths[name] = generate(name, seed)
    for name, seed in MONOLITHICS.items():
        paths[name] = generate(name, seed)

    for spec_label, spec in [("co-safety", cosafety_spec), ("safety", safety_spec)]:
        print(f"\n  === {spec_label} spec ===")
        for name in list(PRIMITIVES) + list(MONOLITHICS):
            relabel_traces(paths[name], spec)
            rho = (
                pd.read_csv(paths[name])
                .groupby("trace_id")["label"]
                .last()
                .astype(float)
                .mean()
            )
            kind = "primitive" if name in PRIMITIVES else "monolithic"
            print(f"    [{kind}] {name}: rho={rho:.4f}")

    return paths, cosafety_spec, safety_spec


# ---------------------------------------------------------------------------
# Safety-complement tests — correct for check_with_dfa, should pass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mono_name,comp_path", COMBINATIONS)
def test_safety_complement_vs_monolithic(setup, mono_name, comp_path):
    """'Never fast→slow→fast' — safety spec, compositional should match monolithic."""
    paths, _, safety_spec = setup

    relabel_traces(paths[mono_name], safety_spec)
    mono_base = ScenarioBase({mono_name: paths[mono_name]})
    rho_mono = mono_base.get_success_prob(mono_name)
    eps_mono = mono_base.get_success_prob_uncertainty(mono_name)

    prim_paths = {p: paths[p] for p in set(comp_path)}
    for p in prim_paths:
        relabel_traces(prim_paths[p], safety_spec)
    engine = CompositionalAnalysisEngine(ScenarioBase(prim_paths))
    rho_comp, eps_comp = engine.check_with_dfa(
        comp_path,
        safety_spec,
        features=["speed"],
        center_feat_idx=[],
    )

    label = "→".join(comp_path)
    print(f"\n  [safety {label}]")
    print(f"    Monolithic    rho({mono_name})  = {rho_mono:.4f} +/- {eps_mono:.4f}")
    print(f"    Compositional rho({label}) = {rho_comp:.4f} +/- {eps_comp:.4f}")
    print(
        f"    => co-safety (1 - rho): mono = {1 - rho_mono:.4f},  comp = {1 - rho_comp:.4f}"
    )

    diff = abs(rho_comp - rho_mono)
    tolerance = 2.0 * (eps_mono + eps_comp) + 0.15
    assert diff <= tolerance, (
        f"|diff| = {diff:.4f} > tolerance {tolerance:.4f}  "
        f"(mono={rho_mono:.4f}, comp={rho_comp:.4f})"
    )


# ---------------------------------------------------------------------------
# Co-safety tests — xfail: absorbing-accept collapses compositional rho
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "check_with_dfa is incorrect for co-safety / absorbing-accept DFAs. "
        "Conditioning on Sub1 success collapses Sub2 q_init to the absorbing-accept "
        "state (trivially absorbing), so compositional rho = Sub1 rho and Sub2 "
        "is silently ignored."
    ),
    strict=False,
)
@pytest.mark.parametrize("mono_name,comp_path", COMBINATIONS)
def test_cosafety_vs_monolithic(setup, mono_name, comp_path):
    """'Eventually fast→slow→fast' — co-safety, expected to produce wrong compositional rho."""
    paths, cosafety_spec, _ = setup

    relabel_traces(paths[mono_name], cosafety_spec)
    mono_base = ScenarioBase({mono_name: paths[mono_name]})
    rho_mono = mono_base.get_success_prob(mono_name)
    eps_mono = mono_base.get_success_prob_uncertainty(mono_name)

    prim_paths = {p: paths[p] for p in set(comp_path)}
    for p in prim_paths:
        relabel_traces(prim_paths[p], cosafety_spec)
    engine = CompositionalAnalysisEngine(ScenarioBase(prim_paths))
    rho_comp, eps_comp = engine.check_with_dfa(
        comp_path,
        cosafety_spec,
        features=["speed"],
        center_feat_idx=[],
    )

    label = "→".join(comp_path)
    print(f"\n  [co-safety {label}]")
    print(f"    Monolithic    rho({mono_name})  = {rho_mono:.4f} +/- {eps_mono:.4f}")
    print(f"    Compositional rho({label}) = {rho_comp:.4f} +/- {eps_comp:.4f}")
    print(
        f"    (collapse: rho_comp should equal rho(first primitive) if bug is present)"
    )

    diff = abs(rho_comp - rho_mono)
    tolerance = 2.0 * (eps_mono + eps_comp) + 0.15
    assert diff <= tolerance, (
        f"|diff| = {diff:.4f} > tolerance {tolerance:.4f}  "
        f"(mono={rho_mono:.4f}, comp={rho_comp:.4f})"
    )
