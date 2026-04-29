"""
Parser for Scenic composition analysis output.

Converts the JSON produced by scenic_composition_analysis.py into the
List[CompositionStep] format expected by check_with_dfa.

Actual Scenic format:
    - steps is a list where each element is either:
        str              → reference to another container (sequential)
        dict[str, float] → random choice among containers with weights
        dict["shuffle", dict[str, float]] → shuffle (not yet supported)
    - Containers with empty steps are leaf primitives (the scenarios
      you actually generate traces for in MetaDrive)

Example:
    {
        "entrypoints": ["Main"],
        "containers": {
            "Main": {"kind": "scenario", "steps": ["A", {"B": 0.6, "C": 0.4}]},
            "A": {"kind": "behavior", "steps": []},
            "B": {"kind": "behavior", "steps": []},
            "C": {"kind": "behavior", "steps": []}
        }
    }
    → ["A", {"B": 0.6, "C": 0.4}]
"""

import json
from typing import Dict, List, Set, Tuple, Union

CompositionStep = Union[str, Dict[str, float]]


def parse_scenic_spec(spec: dict) -> Dict[str, List[CompositionStep]]:
    """
    Parse a Scenic spec and return a dict mapping each entrypoint
    to its flattened List[CompositionStep].
    """
    containers = spec["containers"]
    results = {}
    for entry in spec["entrypoints"]:
        results[entry] = _resolve_container(entry, containers, visited=set())
    return results


def scenic_to_check_input(spec: dict) -> List[CompositionStep]:
    """
    Parse a Scenic spec and return the composition for the first entrypoint.
    """
    results = parse_scenic_spec(spec)
    return results[spec["entrypoints"][0]]


def get_primitives(composition: List[CompositionStep]) -> Set[str]:
    """
    Extract the set of primitive scenario names from a parsed composition.
    These are the scenarios you need to generate traces for.
    """
    primitives = set()
    for step in composition:
        if isinstance(step, str):
            primitives.add(step)
        elif isinstance(step, dict):
            primitives.update(step.keys())
    return primitives


def _is_leaf(name: str, containers: dict) -> bool:
    """A container is a leaf if it has no steps (or isn't in containers at all)."""
    if name not in containers:
        return True
    return len(containers[name].get("steps", [])) == 0


def _is_shuffle(step: dict) -> bool:
    """Check if a dict step is a shuffle operation."""
    return "shuffle" in step and len(step) == 1


def _is_random_choice(step: dict) -> bool:
    """Check if a dict step is a random choice (all values are floats)."""
    return all(isinstance(v, (int, float)) for v in step.values())


def _resolve_container(
    name: str,
    containers: dict,
    visited: set,
) -> List[CompositionStep]:
    """Recursively resolve a container into a flat list of composition steps."""
    if _is_leaf(name, containers):
        return [name]

    if name in visited:
        raise ValueError(f"Cycle detected: container '{name}' references itself")

    visited = visited | {name}
    container = containers[name]
    steps = container.get("steps", [])

    result: List[CompositionStep] = []

    for step in steps:
        if isinstance(step, str):
            # Sequential reference to another container
            expanded = _resolve_container(step, containers, visited)
            result.extend(expanded)

        elif isinstance(step, dict) and _is_shuffle(step):
            # Shuffle: execute ALL branches sequentially in weight-descending order.
            # Scenic's `do shuffle` always runs every branch but in a random order;
            # we approximate this for check_with_dfa as sequential (highest weight
            # first) so that all primitives get traces and can be composed.
            shuffle_dict = step["shuffle"]
            ordered = sorted(shuffle_dict.items(), key=lambda kv: kv[1], reverse=True)
            for branch_name, _ in ordered:
                expanded = _resolve_container(branch_name, containers, visited)
                result.extend(expanded)

        elif isinstance(step, dict) and _is_random_choice(step):
            # Random choice: {"ContainerA": 0.5, "ContainerB": 0.5}
            # Resolve each branch. If all branches are single primitives,
            # emit one dict step (merging weights for duplicate primitives).
            # If any branch expands to multiple steps, raise NotImplementedError
            # because check_with_dfa requires a flat structure.
            merged: Dict[str, float] = {}
            multi_step_branches = {}

            for branch_name, weight in step.items():
                expanded = _resolve_container(branch_name, containers, visited)

                if len(expanded) == 1 and isinstance(expanded[0], str):
                    prim = expanded[0]
                    merged[prim] = merged.get(prim, 0.0) + weight
                else:
                    multi_step_branches[branch_name] = (expanded, weight)

            if multi_step_branches:
                branch_details = {
                    name: f"{exp} (weight={w})"
                    for name, (exp, w) in multi_step_branches.items()
                }
                raise NotImplementedError(
                    f"Random branches that expand to multi-step sequences "
                    f"are not yet supported by check_with_dfa.\n"
                    f"Branches: {branch_details}\n"
                    f"Each random branch must resolve to a single primitive scenario."
                )

            result.append(merged)

        else:
            raise ValueError(f"Unexpected step format: {step}")

    return result


if __name__ == "__main__":
    # Test with the actual black_ice_ramp output
    spec = {
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
                    {
                        "BlackIceRampPrimaryResponse": 0.75,
                        "BlackIceRampSecondaryResponse": 0.25,
                    },
                    {
                        "shuffle": {
                            "BlackIceRampAftershockA": 0.6666,
                            "BlackIceRampAftershockB": 0.3333,
                        }
                    },
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
                "steps": [
                    {"BlackIceRampFallback": 0.5, "BlackIceRampTimedRecovery": 0.5}
                ],
            },
            "BlackIceRampStabilize": {"kind": "behavior", "steps": []},
            "BlackIceRampFallback": {"kind": "behavior", "steps": []},
            "BlackIceRampTimedRecovery": {"kind": "behavior", "steps": []},
            "BlackIceRampAftershockA": {"kind": "behavior", "steps": []},
            "BlackIceRampAftershockB": {"kind": "behavior", "steps": []},
            "BlackIceRampCleanup": {
                "kind": "behavior",
                "steps": [
                    {
                        "shuffle": {
                            "BlackIceRampExitLeft": 0.5,
                            "BlackIceRampExitRight": 0.5,
                        }
                    }
                ],
            },
            "BlackIceRampExitLeft": {"kind": "behavior", "steps": []},
            "BlackIceRampExitRight": {"kind": "behavior", "steps": []},
            "BlackIceRampTraffic": {
                "kind": "scenario",
                "steps": ["BlackIceRampIncident", "BlackIceRampCleanup"],
            },
        },
    }

    print("Attempting to parse full black_ice_ramp spec...\n")

    # This will hit NotImplementedError because:
    # - BlackIceRampHazardPathA expands to 3 sequential steps (multi-step branch)
    # - BlackIceRampIncident has a shuffle step
    try:
        result = scenic_to_check_input(spec)
        print(f"Result: {result}")
        print(f"Primitives: {get_primitives(result)}")
    except NotImplementedError as e:
        print(f"NotImplementedError: {e}")

    # Test with a simpler spec that DOES work
    print("\n--- Simpler spec (all branches are single primitives) ---\n")

    simple_spec = {
        "entrypoints": ["Main"],
        "containers": {
            "Main": {
                "kind": "scenario",
                "steps": ["Straight", {"IntBlock": 0.6, "RoundBlock": 0.4}],
            },
            "Straight": {"kind": "behavior", "steps": []},
            "IntBlock": {"kind": "behavior", "steps": ["Intersection"]},
            "Intersection": {"kind": "behavior", "steps": []},
            "RoundBlock": {"kind": "behavior", "steps": ["Roundabout"]},
            "Roundabout": {"kind": "behavior", "steps": []},
        },
    }

    result = scenic_to_check_input(simple_spec)
    print(f"Result: {result}")
    print(f"Primitives: {get_primitives(result)}")
