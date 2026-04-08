"""
End-to-end example: Scenic spec → parse → generate traces → check_with_dfa.

Usage:
    python example_random_pipeline.py
"""

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / ".."))

from verifai.monitor import automaton_specification
from verifai.compositional_analysis import ScenarioBase, CompositionalAnalysisEngine
from verifai.scenic_parser import scenic_to_check_input, get_primitives

SPEED_LIMIT_MS = 15.0
SLOW_THRESHOLD_MS = 6.5
N_EPISODES = 200
SAVE_DIR = os.path.join(os.path.dirname(__file__), "random_pipeline_traces")


def make_safety_spec():
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


def generate(scenario, save_dir, n=N_EPISODES, seed=0):
    from utils import generate_traces
    generate_traces(
        seed=seed, save_dir=save_dir, model_path=None,
        expert=True, n=n, scenario=scenario, gif=False,
    )
    return os.path.join(save_dir, scenario, "traces.csv")


def relabel(csv_path, spec):
    df = pd.read_csv(csv_path)
    df["trace_id"] = df["trace_id"].astype(str)
    new_labels = {}
    for tid, group in df.sort_values("step").groupby("trace_id"):
        traj = group.to_dict("records")
        word = [spec.L(row) for row in traj]
        new_labels[tid] = spec._dfa.label(word)
    df["label"] = df["trace_id"].map(new_labels)
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":

    # 1. Define the Scenic spec
    #    "Drive straight, then randomly encounter an intersection (60%)
    #     or a roundabout (40%)"
    scenic_spec = {
        "entrypoints": ["Main"],
        "containers": {
            "Main": {
                "kind": "scenario",
                "steps": [
                    "S",
                    {"X": 0.6, "O": 0.4},
                ],
            },
            "S": {"kind": "behavior", "steps": []},
            "X": {"kind": "behavior", "steps": []},
            "O": {"kind": "behavior", "steps": []},
        },
    }

    # 2. Parse into check_with_dfa input
    composition = scenic_to_check_input(scenic_spec)
    print(f"Parsed composition: {composition}")
    # → ["S", {"X": 0.6, "O": 0.4}]

    # 3. Figure out which primitive scenarios we need traces for
    primitives = get_primitives(composition)
    print(f"Primitives to generate: {primitives}")
    # → {"S", "X", "O"}

    # 4. Generate traces for each primitive
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"\nGenerating {N_EPISODES} traces per scenario ...\n")

    logs = {}
    for i, scenario in enumerate(sorted(primitives)):
        csv_path = generate(scenario, SAVE_DIR, n=N_EPISODES, seed=i)
        logs[scenario] = csv_path

    # 5. Build DFA spec and relabel all traces
    spec = make_safety_spec()
    for scenario, path in logs.items():
        relabel(path, spec)

    # Print per-scenario stats
    print()
    for scenario, path in sorted(logs.items()):
        df = pd.read_csv(path)
        n = df["trace_id"].nunique()
        rho = df.groupby("trace_id")["label"].last().astype(float).mean()
        print(f"  {scenario}: {n} traces, rho={rho:.4f}")

    # 6. Run check_with_dfa with the parsed composition
    sb = ScenarioBase(logs)
    engine = CompositionalAnalysisEngine(sb)

    rho, eps = engine.check_with_dfa(
        composition,
        spec,
        features=["x", "y", "heading", "speed"],
        center_feat_idx=[0, 1],
        bw_method="scott",
    )

    print(f"\n  Composition: {composition}")
    print(f"  rho = {rho:.4f} +/- {eps:.4f}")
    print(f"\nDone. Traces at: {os.path.abspath(SAVE_DIR)}")