"""
Parser for Scenic composition analysis output.

Converts the JSON produced by scenic_composition_analysis.py into a list of
(probability, composition) paths, where each composition is a
List[CompositionStep] as expected by check_with_dfa.

Supported Scenic constructs:
  - Sequential containers  (resolved recursively)
  - Random choice          {"A": 0.6, "B": 0.4}
  - Shuffle                {"shuffle": {"A": p1, "B": p2, ...}}
    → All items are executed once, in uniformly random order.
      Expanded into one path per permutation, each with equal probability.
  - Multi-step branches    Random-choice branch that resolves to a sequence
    → Each branch becomes a separate path weighted by its branch probability.

Return type of scenic_to_check_input / parse_scenic_spec:
    List[Tuple[float, List[CompositionStep]]]
    Each tuple is (probability, composition) where the composition is a
    flat list of CompositionSteps that check_with_dfa can consume directly.
    All probabilities in the returned list sum to 1.0.

Example (simple, no branching):
    {"entrypoints": ["Main"],
     "containers": {
         "Main": {"kind": "scenario", "steps": ["A", {"B": 0.6, "C": 0.4}]},
         "A": {"kind": "behavior", "steps": []},
         "B": {"kind": "behavior", "steps": []},
         "C": {"kind": "behavior", "steps": []}
     }}
    → [(1.0, ["A", {"B": 0.6, "C": 0.4}])]
"""

from itertools import permutations
from typing import Dict, List, Set, Tuple, Union

CompositionStep = Union[str, Dict[str, float]]
Path = Tuple[float, List[CompositionStep]]  # (probability, composition)


def parse_scenic_spec(spec: dict) -> Dict[str, List[Path]]:
    """
    Parse a Scenic spec and return a dict mapping each entrypoint
    to its list of (probability, composition) paths.
    """
    containers = spec["containers"]
    return {
        entry: _resolve_container(entry, containers, visited=set())
        for entry in spec["entrypoints"]
    }


def scenic_to_check_input(spec: dict) -> List[Path]:
    """
    Parse a Scenic spec and return paths for the first entrypoint.
    Each path is (probability, List[CompositionStep]).
    All probabilities sum to 1.0.
    """
    results = parse_scenic_spec(spec)
    return results[spec["entrypoints"][0]]


def get_primitives(paths: List[Path]) -> Set[str]:
    """
    Extract the set of primitive scenario names from a list of paths.
    These are the scenarios you need to generate traces for.
    """
    primitives = set()
    for _, composition in paths:
        for step in composition:
            if isinstance(step, str):
                primitives.add(step)
            elif isinstance(step, dict):
                primitives.update(step.keys())
    return primitives


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_leaf(name: str, containers: dict) -> bool:
    if name not in containers:
        return True
    return len(containers[name].get("steps", [])) == 0


def _is_shuffle(step: dict) -> bool:
    return "shuffle" in step and len(step) == 1


def _is_random_choice(step: dict) -> bool:
    return all(isinstance(v, (int, float)) for v in step.values())


def _cross_product(paths_a: List[Path], paths_b: List[Path]) -> List[Path]:
    """Combine two path lists: multiply probabilities and concatenate steps."""
    return [
        (p_a * p_b, steps_a + steps_b)
        for (p_a, steps_a) in paths_a
        for (p_b, steps_b) in paths_b
    ]


def _resolve_container(name: str, containers: dict, visited: set) -> List[Path]:
    """Recursively resolve a container into a list of (probability, composition) paths."""
    if _is_leaf(name, containers):
        return [(1.0, [name])]

    if name in visited:
        raise ValueError(f"Cycle detected: container '{name}' references itself")

    visited = visited | {name}
    steps = containers[name].get("steps", [])

    result_paths: List[Path] = [(1.0, [])]

    for step in steps:
        if isinstance(step, str):
            sub_paths = _resolve_container(step, containers, visited)
            result_paths = _cross_product(result_paths, sub_paths)

        elif isinstance(step, dict) and _is_shuffle(step):
            sub_paths = _expand_shuffle(step["shuffle"], containers, visited)
            result_paths = _cross_product(result_paths, sub_paths)

        elif isinstance(step, dict) and _is_random_choice(step):
            sub_paths = _expand_random_choice(step, containers, visited)
            result_paths = _cross_product(result_paths, sub_paths)

        else:
            raise ValueError(f"Unexpected step format: {step}")

    return result_paths


def _expand_shuffle(weights_dict: dict, containers: dict, visited: set) -> List[Path]:
    """
    Expand a shuffle step: execute all items once, in uniformly random order.
    Returns one path per permutation, each with probability 1 / n!.
    """
    names = list(weights_dict.keys())
    perms = list(permutations(names))
    perm_prob = 1.0 / len(perms)

    result: List[Path] = []
    for perm in perms:
        perm_paths: List[Path] = [(1.0, [])]
        for item in perm:
            perm_paths = _cross_product(perm_paths, _resolve_container(item, containers, visited))
        result.extend([(perm_prob * p, steps) for (p, steps) in perm_paths])
    return result


def _expand_random_choice(step: dict, containers: dict, visited: set) -> List[Path]:
    """
    Expand a random-choice step.

    If every branch resolves to exactly one primitive scenario:
        → Emit a single path containing one dict step {name: weight, ...}.
          This preserves the existing check_with_dfa random-choice representation
          and enables importance sampling across the choice point.

    Otherwise (any branch is multi-step or nested):
        → Expand each branch into separate paths weighted by branch probability.
    """
    total_weight = sum(step.values())
    branch_resolved: Dict[str, Tuple[List[Path], float]] = {
        name: (_resolve_container(name, containers, visited), weight / total_weight)
        for name, weight in step.items()
    }

    all_single_primitive = all(
        len(paths) == 1
        and len(paths[0][1]) == 1
        and isinstance(paths[0][1][0], str)
        for (paths, _) in branch_resolved.values()
    )

    if all_single_primitive:
        random_step: CompositionStep = {
            paths[0][1][0]: norm_w
            for name, (paths, norm_w) in branch_resolved.items()
        }
        return [(1.0, [random_step])]

    expanded: List[Path] = []
    for name, (paths, norm_w) in branch_resolved.items():
        expanded.extend([(norm_w * p, s) for (p, s) in paths])
    return expanded
