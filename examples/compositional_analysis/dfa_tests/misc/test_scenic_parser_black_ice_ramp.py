"""
Unit tests for scenic_parser using the black_ice_ramp scenario structure.

scenic_to_check_input returns List[Tuple[float, List[CompositionStep]]]:
a list of (probability, composition) paths that together cover the full
stochastic execution tree of the spec. All probabilities sum to 1.0.

Supported constructs exercised here:

  Shuffle  — BlackIceRampCleanup: {"shuffle": {ExitLeft: 0.5, ExitRight: 0.5}}
    → 2! = 2 paths, each with prob 0.5, one per ordering.

  Multi-step random branch — BlackIceRampEnvironment:
    {"PathA": 0.5, "PathB": 0.5} where PathA → [Stabilize, Fallback, TimedRecovery]
    → Expand into 2 paths (one per branch) weighted by branch probability.

  Full spec (Main) — 2 Environment × 4 Incident × 2 Cleanup = 16 paths.

Usage: pytest test_scenic_parser_black_ice_ramp.py -v
"""

import pytest
from verifai.scenic_parser import scenic_to_check_input, get_primitives

# ---------------------------------------------------------------------------
# Full spec exactly as produced by scenic_composition_analysis.py
# ---------------------------------------------------------------------------

BLACK_ICE_SPEC = {
    "entrypoints": ["Main"],
    "containers": {
        "Main": {
            "kind": "scenario",
            "steps": ["BlackIceRampEnvironment", "BlackIceRampTraffic"],
        },
        "BlackIceRampEnvironment": {
            "kind": "scenario",
            "steps": [
                "BlackIceRampHazardPrelude",
                {"BlackIceRampHazardPathA": 0.5, "BlackIceRampHazardPathB": 0.5},
            ],
        },
        "BlackIceRampHazardPrelude": {"kind": "behavior", "steps": []},
        "BlackIceRampHazardPathA": {
            "kind": "behavior",
            "steps": ["BlackIceRampPrimaryResponse"],
        },
        "BlackIceRampHazardPathB": {
            "kind": "behavior",
            "steps": ["BlackIceRampSecondaryResponse"],
        },
        "BlackIceRampIncident": {
            "kind": "scenario",
            "steps": [
                {"BlackIceRampPrimaryResponse": 0.75, "BlackIceRampSecondaryResponse": 0.25},
                {"shuffle": {"BlackIceRampAftershockA": 0.6666666666666666,
                             "BlackIceRampAftershockB": 0.3333333333333333}},
            ],
        },
        "BlackIceRampPrimaryResponse": {
            "kind": "behavior",
            "steps": [
                "BlackIceRampStabilize",
                "BlackIceRampFallback",
                "BlackIceRampTimedRecovery",
            ],
        },
        "BlackIceRampSecondaryResponse": {
            "kind": "behavior",
            "steps": [{"BlackIceRampFallback": 0.5, "BlackIceRampTimedRecovery": 0.5}],
        },
        "BlackIceRampStabilize":      {"kind": "behavior", "steps": []},
        "BlackIceRampFallback":        {"kind": "behavior", "steps": []},
        "BlackIceRampTimedRecovery":   {"kind": "behavior", "steps": []},
        "BlackIceRampAftershockA":     {"kind": "behavior", "steps": []},
        "BlackIceRampAftershockB":     {"kind": "behavior", "steps": []},
        "BlackIceRampCleanup": {
            "kind": "behavior",
            "steps": [
                {"shuffle": {"BlackIceRampExitLeft": 0.5, "BlackIceRampExitRight": 0.5}}
            ],
        },
        "BlackIceRampExitLeft":  {"kind": "behavior", "steps": []},
        "BlackIceRampExitRight": {"kind": "behavior", "steps": []},
        "BlackIceRampTraffic": {
            "kind": "scenario",
            "steps": ["BlackIceRampIncident", "BlackIceRampCleanup"],
        },
    },
}

# ---------------------------------------------------------------------------
# Simplified spec: every non-trivial container treated as a leaf.
# No shuffle, no multi-step branches → single path, prob 1.0.
# ---------------------------------------------------------------------------

SIMPLIFIED_SPEC = {
    "entrypoints": ["Main"],
    "containers": {
        "Main": {
            "kind": "scenario",
            "steps": ["BlackIceRampEnvironment", "BlackIceRampTraffic"],
        },
        "BlackIceRampEnvironment": {
            "kind": "scenario",
            "steps": [
                "BlackIceRampHazardPrelude",
                {"BlackIceRampHazardPathA": 0.5, "BlackIceRampHazardPathB": 0.5},
            ],
        },
        "BlackIceRampHazardPrelude": {"kind": "behavior", "steps": []},
        "BlackIceRampHazardPathA":   {"kind": "behavior", "steps": []},
        "BlackIceRampHazardPathB":   {"kind": "behavior", "steps": []},
        "BlackIceRampTraffic": {
            "kind": "scenario",
            "steps": ["BlackIceRampIncident", "BlackIceRampCleanup"],
        },
        "BlackIceRampIncident": {"kind": "behavior", "steps": []},
        "BlackIceRampCleanup":  {"kind": "behavior", "steps": []},
    },
}


# ---------------------------------------------------------------------------
# Tests: shuffle
# ---------------------------------------------------------------------------

def test_shuffle_expands_to_orderings():
    """
    Cleanup has shuffle([ExitLeft, ExitRight]).
    Should expand to 2 paths of equal probability, one per ordering.
    """
    cleanup_spec = {
        "entrypoints": ["BlackIceRampCleanup"],
        "containers": BLACK_ICE_SPEC["containers"],
    }
    paths = scenic_to_check_input(cleanup_spec)

    assert len(paths) == 2
    probs = [p for (p, _) in paths]
    assert abs(sum(probs) - 1.0) < 1e-9
    assert all(abs(p - 0.5) < 1e-9 for p in probs)

    names_per_path = [tuple(c) for (_, c) in paths]
    assert set(names_per_path) == {
        ("BlackIceRampExitLeft", "BlackIceRampExitRight"),
        ("BlackIceRampExitRight", "BlackIceRampExitLeft"),
    }


# ---------------------------------------------------------------------------
# Tests: multi-step random branch
# ---------------------------------------------------------------------------

def test_multi_step_branch_expands_to_paths():
    """
    Environment has {PathA: 0.5, PathB: 0.5}.
    PathA → [Stabilize, Fallback, TimedRecovery] (multi-step).
    PathB → [{Fallback: 0.5, TimedRecovery: 0.5}] (nested random choice).
    Should expand to 2 paths, each with prob 0.5.
    """
    env_spec = {
        "entrypoints": ["BlackIceRampEnvironment"],
        "containers": BLACK_ICE_SPEC["containers"],
    }
    paths = scenic_to_check_input(env_spec)

    assert len(paths) == 2
    probs = [p for (p, _) in paths]
    assert abs(sum(probs) - 1.0) < 1e-9
    assert all(abs(p - 0.5) < 1e-9 for p in probs)

    # PathA branch: [Prelude, Stabilize, Fallback, TimedRecovery] — 4 steps
    # PathB branch: [Prelude, {Fallback: 0.5, TimedRecovery: 0.5}] — 2 steps
    step_counts = sorted(len(c) for (_, c) in paths)
    assert step_counts == [2, 4]


# ---------------------------------------------------------------------------
# Tests: full spec
# ---------------------------------------------------------------------------

def test_full_spec_parses():
    """
    Main parses without error into 16 paths.
    2 Environment × (2 Incident-branch × 2 Aftershock-orderings) × 2 Cleanup-orderings
    = 2 × 4 × 2 = 16
    """
    paths = scenic_to_check_input(BLACK_ICE_SPEC)
    assert len(paths) == 16
    assert abs(sum(p for (p, _) in paths) - 1.0) < 1e-9


def test_full_spec_primitives():
    """get_primitives collects every leaf scenario name across all paths."""
    paths = scenic_to_check_input(BLACK_ICE_SPEC)
    assert get_primitives(paths) == {
        "BlackIceRampHazardPrelude",
        "BlackIceRampStabilize",
        "BlackIceRampFallback",
        "BlackIceRampTimedRecovery",
        "BlackIceRampAftershockA",
        "BlackIceRampAftershockB",
        "BlackIceRampExitLeft",
        "BlackIceRampExitRight",
    }


# ---------------------------------------------------------------------------
# Tests: simplified spec (no expansion needed)
# ---------------------------------------------------------------------------

def test_simplified_composition():
    """Simplified spec has no branching — returns a single path with prob 1.0."""
    paths = scenic_to_check_input(SIMPLIFIED_SPEC)
    assert len(paths) == 1
    prob, composition = paths[0]
    assert abs(prob - 1.0) < 1e-9
    assert composition == [
        "BlackIceRampHazardPrelude",
        {"BlackIceRampHazardPathA": 0.5, "BlackIceRampHazardPathB": 0.5},
        "BlackIceRampIncident",
        "BlackIceRampCleanup",
    ]


def test_simplified_primitives():
    paths = scenic_to_check_input(SIMPLIFIED_SPEC)
    assert get_primitives(paths) == {
        "BlackIceRampHazardPrelude",
        "BlackIceRampHazardPathA",
        "BlackIceRampHazardPathB",
        "BlackIceRampIncident",
        "BlackIceRampCleanup",
    }


# ---------------------------------------------------------------------------
# Tests: error cases
# ---------------------------------------------------------------------------

def test_cycle_detection():
    """A container that references itself (directly or indirectly) raises ValueError."""
    cyclic_spec = {
        "entrypoints": ["A"],
        "containers": {
            "A": {"kind": "scenario", "steps": ["B"]},
            "B": {"kind": "scenario", "steps": ["A"]},
        },
    }
    with pytest.raises(ValueError, match="[Cc]ycle"):
        scenic_to_check_input(cyclic_spec)
