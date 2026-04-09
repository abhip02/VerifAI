"""
examples/compositional_analysis/dfa_tests/test_check_with_dfa_synthetic.py

Test for check_with_dfa using synthetic traces.

Spec (non-Markovian):
    If the vehicle ever brakes, it must slow below 5 m/s before the scenario ends.

DFA:
    ok (accepting) --brake--> braking
    ok             --else---> ok
    braking        --slow---> ok
    braking        --else---> braking

Each scenario is evaluated independently.  A trace satisfies the spec iff
the DFA ends in "ok" — meaning either it never braked, or it braked and
then slowed below 5 m/s within the same scenario.

Scenarios:
    S (straight):  10 steps.  Each trace independently:
                   - with P=0.5: brakes at step 3, then slows below 5 at step 7  → ok
                   - with P=0.5: brakes at step 3, never slows                   → braking (fail)
                   → rho_S ≈ 0.5

    X (intersection):  10 steps.  Each trace independently:
                   - with P=0.8: brakes at step 2, then slows below 5 at step 5  → ok
                   - with P=0.2: brakes at step 2, never slows                   → braking (fail)
                   → rho_X ≈ 0.8

Expected:
    Monolithic rho_SX  ≈ rho_S * rho_X  ≈ 0.5 * 0.8 = 0.4
    Compositional rho  ≈ 0.4  (should match)
"""

import os
import tempfile

import numpy as np
import pandas as pd

from verifai.monitor import automaton_specification
from verifai.compositional_analysis import ScenarioBase, CompositionalAnalysisEngine


def make_spec():
    def transition(state, sym):
        if state == "ok":
            return "braking" if sym == "brake" else "ok"
        if state == "braking":
            return "ok" if sym == "slow" else "braking"

    return automaton_specification(
        start="ok",
        inputs={"normal", "brake", "slow"},
        transition=transition,
        label=lambda s: s == "ok",
        labeling_function=lambda row: (
            "slow"  if row["speed"] < 5.0
            else "brake" if row["braking"]
            else "normal"
        ),
    )


def dfa_label(spec, traj):
    word = [spec.L(r) for r in traj]
    return spec._dfa.advance(word).start == "ok"


def make_scenario_traces(rng, n, p_resolve, x_offset=0.0):
    """
    Build n traces for one scenario.

    Each trace: 10 steps.
    - Always brakes at step 3 (speed stays >= 5).
    - With probability p_resolve: slows below 5 at step 7.
    - Otherwise: stays above 5 the whole time.

    Returns a DataFrame with columns:
        trace_id, step, x, y, speed, braking, label
    """
    spec = make_spec()
    rows = []

    for tid in range(n):
        resolves = rng.random() < p_resolve

        traj = []
        for step in range(10):
            braking = (step == 3)
            if resolves and step >= 7:
                speed = 3.0 + rng.uniform(-0.2, 0.2)
            else:
                speed = 8.0 + rng.uniform(-0.5, 0.5)
            traj.append({"speed": speed, "braking": braking})

        label = dfa_label(spec, traj)

        for step, r in enumerate(traj):
            rows.append({
                "trace_id": tid,
                "step":     step,
                "x":        x_offset + step + rng.uniform(-0.1, 0.1),
                "y":        rng.uniform(-0.3, 0.3),
                "speed":    r["speed"],
                "braking":  r["braking"],
                "label":    label,
            })

    return pd.DataFrame(rows)


def make_traces(n=200, seed=42):
    rng = np.random.default_rng(seed)

    df_S = make_scenario_traces(rng, n, p_resolve=0.5, x_offset=0.0)
    df_X = make_scenario_traces(rng, n, p_resolve=0.8, x_offset=10.0)

    # SX: for each trace id, concatenate S then X steps
    # label_SX = label_S AND label_X  (both segments must satisfy independently)
    rows_SX = []
    for tid in range(n):
        s_rows = df_S[df_S["trace_id"] == tid].sort_values("step")
        x_rows = df_X[df_X["trace_id"] == tid].sort_values("step")
        label_SX = bool(s_rows["label"].iloc[0]) and bool(x_rows["label"].iloc[0])
        for step, (_, row) in enumerate(pd.concat([s_rows, x_rows]).iterrows()):
            rows_SX.append({**row.to_dict(), "step": step, "label": label_SX})

    df_SX = pd.DataFrame(rows_SX)
    return df_S, df_X, df_SX


def test_check_with_dfa():
    n = 200
    df_S, df_X, df_SX = make_traces(n=n)

    print("\n--- Label sanity check ---")
    for name, df in [("S", df_S), ("X", df_X), ("SX", df_SX)]:
        dist = df.groupby("trace_id")["label"].last().value_counts().to_dict()
        rho  = df.groupby("trace_id")["label"].last().mean()
        print(f"  {name}: {dist}  →  rho = {rho:.2f}")

    spec = make_spec()

    tmp = {}
    for name, df in [("S", df_S), ("X", df_X), ("SX", df_SX)]:
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        df.to_csv(path, index=False)
        tmp[name] = path

    try:
        # Monolithic
        sb_mono = ScenarioBase({"SX": tmp["SX"]})
        rho_mono = sb_mono.get_success_prob("SX")
        eps_mono = sb_mono.get_success_prob_uncertainty("SX")

        # Compositional
        sb_comp = ScenarioBase({"S": tmp["S"], "X": tmp["X"]})
        engine  = CompositionalAnalysisEngine(sb_comp)
        rho_comp, eps_comp = engine.check_with_dfa(
            ["S", "X"],
            spec,
            features=["x", "y", "speed"],
            center_feat_idx=[0, 1],
        )

        print(f"\nMonolithic    rho = {rho_mono:.4f} ± {eps_mono:.4f}  (expected ~0.40)")
        print(f"Compositional rho = {rho_comp:.4f} ± {eps_comp:.4f}  (expected ~0.40)")

    finally:
        for path in tmp.values():
            os.unlink(path)


if __name__ == "__main__":
    test_check_with_dfa()