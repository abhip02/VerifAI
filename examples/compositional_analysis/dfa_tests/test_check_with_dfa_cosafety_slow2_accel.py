"""
Co-safety "slow-2-then-accel" test: non-Markovian 4-state DFA requiring the vehicle
to slow for at least 2 consecutive steps and then immediately accelerate to fast.

The mandatory two-step slow "windup" makes the property history-dependent:
seeing a single slow→fast transition is not enough.

Co-safety spec ("eventually: slow×2+ then immediately fast"):

        moving      --slow-->  slow_1
        moving      --fast-->  moving
        slow_1      --slow-->  slow_2           (second consecutive slow)
        slow_1      --fast-->  moving           (only one slow — reset)
        slow_2      --fast-->  achieved         (absorbing-accept)
        slow_2      --slow-->  slow_2           (keep accumulating slow)
        achieved    --any -->  achieved

    label(s) = s == "achieved"

Safety complement ("never: slow×2+ then immediately fast"):
    Once a vehicle has been slow for ≥2 steps, going fast is a violation.

        ok          --slow-->  slow_1
        ok          --fast-->  ok
        slow_1      --slow-->  slow_2
        slow_1      --fast-->  ok
        slow_2      --fast-->  violated         (absorbing-reject)
        slow_2      --slow-->  slow_2
        violated    --any -->  violated

    label(s) = s != "violated"

The co-safety test is marked xfail because check_with_dfa uses a multiplicative
formula that is only correct for safety / absorbing-reject DFAs.  For a
co-safety DFA the conditioning on Sub1 success collapses Sub2 q_init to the
absorbing-accept state, so compositional rho collapses to Sub1 rho.

The safety-complement test behaves correctly and should pass.

Primitives:  S, X, C
Combinations:
    SX   : S → X          (2-step)
    SXS  : S → X → S      (3-step)
    SC   : S → C          (2-step)
    SCS  : S → C → S      (3-step)
    SCX  : S → C → X      (3-step)

Usage: pytest test_check_with_dfa_cosafety_slow2_accel.py -s
"""

import os

import pandas as pd
import pytest

from verifai.monitor import automaton_specification
from verifai.compositional_analysis import ScenarioBase, CompositionalAnalysisEngine, relabel_traces

SLOW_MS       = 3.5    # m/s — below this is "slow" (same threshold as tollgate / two_stops)
FAST_MS       = 8.0    # m/s — at or above this is "fast" (after the slow windup; 8 m/s reachable after a stop)
N_EPISODES    = 1000
TRACE_DIR     = os.path.join(os.path.dirname(__file__), "storage", "slow2_accel")

PRIMITIVES   = {"S": 0, "X": 1, "C": 2}
MONOLITHICS  = {"SX": 3, "SXS": 4, "SC": 5, "SCS": 6, "SCX": 7}
COMBINATIONS = [
    ("SX",  ["S", "X"]),
    ("SXS", ["S", "X", "S"]),
    ("SC",  ["S", "C"]),
    ("SCS", ["S", "C", "S"]),
    ("SCX", ["S", "C", "X"]),
]


def _sym(row):
    spd = row["speed"]
    if spd < SLOW_MS:
        return "slow"
    if spd >= FAST_MS:
        return "fast"
    return "mid"


def make_cosafety_spec():
    """Co-safety: 'eventually slow≥2 steps then immediately fast' — absorbing-accept."""
    def transition(state, sym):
        if state == "achieved":
            return "achieved"
        if state == "moving":
            return "slow_1" if sym == "slow" else "moving"
        if state == "slow_1":
            if sym == "slow":
                return "slow_2"
            if sym == "fast":
                return "moving"
            return "slow_1"   # mid: stay in slow_1
        # state == "slow_2"
        if sym == "fast":
            return "achieved"
        if sym == "slow":
            return "slow_2"
        return "slow_2"   # mid: stay in slow_2

    return automaton_specification(
        start="moving",
        inputs={"slow", "mid", "fast"},
        transition=transition,
        label=lambda s: s == "achieved",
        labeling_function=_sym,
    )


def make_safety_spec():
    """Safety complement: 'never slow≥2 steps then immediately fast' — absorbing-reject."""
    def transition(state, sym):
        if state == "violated":
            return "violated"
        if state == "ok":
            return "slow_1" if sym == "slow" else "ok"
        if state == "slow_1":
            if sym == "slow":
                return "slow_2"
            if sym == "fast":
                return "ok"
            return "slow_1"   # mid
        # state == "slow_2"
        if sym == "fast":
            return "violated"
        if sym == "slow":
            return "slow_2"
        return "slow_2"   # mid

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
        # extra_obstacles=True ensures the vehicle definitely stops for ≥2 consecutive steps
        # (static barriers), giving the DFA a clear slow_1→slow_2 transition.
        generate_traces(seed=seed, save_dir=TRACE_DIR, expert=True,
                        n=N_EPISODES, scenario=scenario, extra_obstacles=True)
    return csv


@pytest.fixture(scope="module")
def setup():
    os.makedirs(TRACE_DIR, exist_ok=True)
    cosafety_spec = make_cosafety_spec()
    safety_spec   = make_safety_spec()
    paths = {}

    for name, seed in PRIMITIVES.items():
        paths[name] = generate(name, seed)
    for name, seed in MONOLITHICS.items():
        paths[name] = generate(name, seed)

    for spec_label, spec in [("co-safety", cosafety_spec), ("safety", safety_spec)]:
        print(f"\n  === {spec_label} spec ===")
        for name in list(PRIMITIVES) + list(MONOLITHICS):
            relabel_traces(paths[name], spec)
            rho = pd.read_csv(paths[name]).groupby("trace_id")["label"].last().astype(float).mean()
            kind = "primitive" if name in PRIMITIVES else "monolithic"
            print(f"    [{kind}] {name}: rho={rho:.4f}")

    return paths, cosafety_spec, safety_spec


# ---------------------------------------------------------------------------
# Safety-complement tests — correct for check_with_dfa, should pass
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mono_name,comp_path", COMBINATIONS)
def test_safety_complement_vs_monolithic(setup, mono_name, comp_path):
    """'Never slow×2 then fast' — safety spec, compositional should match monolithic."""
    paths, _, safety_spec = setup

    relabel_traces(paths[mono_name], safety_spec)
    mono_base = ScenarioBase({mono_name: paths[mono_name]})
    rho_mono  = mono_base.get_success_prob(mono_name)
    eps_mono  = mono_base.get_success_prob_uncertainty(mono_name)

    prim_paths = {p: paths[p] for p in set(comp_path)}
    for p in prim_paths:
        relabel_traces(prim_paths[p], safety_spec)
    engine = CompositionalAnalysisEngine(ScenarioBase(prim_paths))
    rho_comp, eps_comp = engine.check_with_dfa(
        comp_path, safety_spec, features=["speed"], center_feat_idx=[],
    )

    label = "→".join(comp_path)
    print(f"\n  [safety {label}]")
    print(f"    Monolithic    rho({mono_name})  = {rho_mono:.4f} +/- {eps_mono:.4f}")
    print(f"    Compositional rho({label}) = {rho_comp:.4f} +/- {eps_comp:.4f}")
    print(f"    => co-safety (1 - rho): mono = {1 - rho_mono:.4f},  comp = {1 - rho_comp:.4f}")

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
    """'Eventually slow×2 then fast' — co-safety, expected to produce wrong compositional rho."""
    paths, cosafety_spec, _ = setup

    relabel_traces(paths[mono_name], cosafety_spec)
    mono_base = ScenarioBase({mono_name: paths[mono_name]})
    rho_mono  = mono_base.get_success_prob(mono_name)
    eps_mono  = mono_base.get_success_prob_uncertainty(mono_name)

    prim_paths = {p: paths[p] for p in set(comp_path)}
    for p in prim_paths:
        relabel_traces(prim_paths[p], cosafety_spec)
    engine = CompositionalAnalysisEngine(ScenarioBase(prim_paths))
    rho_comp, eps_comp = engine.check_with_dfa(
        comp_path, cosafety_spec, features=["speed"], center_feat_idx=[],
    )

    label = "→".join(comp_path)
    print(f"\n  [co-safety {label}]")
    print(f"    Monolithic    rho({mono_name})  = {rho_mono:.4f} +/- {eps_mono:.4f}")
    print(f"    Compositional rho({label}) = {rho_comp:.4f} +/- {eps_comp:.4f}")
    print(f"    (collapse: rho_comp should equal rho(first primitive) if bug is present)")

    diff = abs(rho_comp - rho_mono)
    tolerance = 2.0 * (eps_mono + eps_comp) + 0.15
    assert diff <= tolerance, (
        f"|diff| = {diff:.4f} > tolerance {tolerance:.4f}  "
        f"(mono={rho_mono:.4f}, comp={rho_comp:.4f})"
    )
