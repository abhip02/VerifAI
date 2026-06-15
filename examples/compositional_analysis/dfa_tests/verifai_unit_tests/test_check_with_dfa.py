"""
Unit tests for check_with_dfa and _dfa_labels in verifai.compositional_analysis.

Tests mathematical properties only — no simulator, no file I/O.
All inputs are in-memory DataFrames with synthetic traces.

Run with:
    pytest examples/compositional_analysis/dfa_tests/verifai_unit_tests/test_check_with_dfa.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from verifai.compositional_analysis import (
    CompositionalAnalysisEngine,
    ScenarioBase,
)
from verifai.monitor import automaton_specification


# ---------------------------------------------------------------------------
# Infrastructure helpers
# ---------------------------------------------------------------------------

def make_engine(data_dict, delta=0.05):
    """
    Create a CompositionalAnalysisEngine backed by in-memory DataFrames,
    bypassing ScenarioBase.__init__ which requires CSV files on disk.
    """
    sb = object.__new__(ScenarioBase)
    sb.logbase = {}
    sb.delta = delta
    sb.data = {
        name: df.assign(trace_id=df["trace_id"].astype(str)).copy()
        for name, df in data_dict.items()
    }
    sb.success_stats = {}
    sb._compute_success_stats()
    return CompositionalAnalysisEngine(sb)


def make_safety_spec():
    """
    2-state safety DFA.
      ok  (start, accepting): ok_sym→ok,  bad_sym→bad
      bad (sink, rejecting):  *→bad
    Labeling function: row["event"]
    """
    def transition(state, sym):
        if state == "bad":
            return "bad"
        return "bad" if sym == "bad_sym" else "ok"

    return automaton_specification(
        start="ok",
        inputs={"ok_sym", "bad_sym"},
        transition=transition,
        label=lambda s: s == "ok",
        labeling_function=lambda row: row["event"],
    )


def make_warned_spec():
    """
    3-state safety DFA that tracks a first warning.
      fresh  (start, accepting): ok→fresh,  warn→warned, fail→failed
      warned (accepting):        ok→warned, warn→failed, fail→failed
      failed (sink, rejecting):  *→failed
    Labeling function: row["event"]
    """
    def transition(state, sym):
        if state == "failed":
            return "failed"
        if state == "fresh":
            if sym == "ok":
                return "fresh"
            if sym == "warn":
                return "warned"
            return "failed"
        # state == "warned"
        if sym == "ok":
            return "warned"
        return "failed"  # sym in {"warn", "fail"}

    return automaton_specification(
        start="fresh",
        inputs={"ok", "warn", "fail"},
        transition=transition,
        label=lambda s: s != "failed",
        labeling_function=lambda row: row["event"],
    )


def make_safety_traces(n_accept, n_reject, n_steps=3, x_offset=0):
    """
    Trace DataFrame for the 2-state safety spec.
      Accepting trace: all steps event="ok_sym"
      Rejecting trace: step 0 event="bad_sym", remaining "ok_sym"
    Columns: trace_id, step, event, x, label
    x is a distinct integer per trace (repeats across steps).
    """
    rows = []
    for i in range(n_accept + n_reject):
        accept = i < n_accept
        x_val = float(x_offset + i)
        for s in range(n_steps):
            event = "ok_sym" if (accept or s > 0) else "bad_sym"
            rows.append({
                "trace_id": i,
                "step": s,
                "event": event,
                "x": x_val,
                "label": accept,
            })
    return pd.DataFrame(rows)


def make_warned_traces_S(x_offset=0):
    """
    S traces for the warned spec (each trace is 1 step):
      4 traces: event="ok"   → end in "fresh"  (accept)
      4 traces: event="warn" → end in "warned" (accept)
      2 traces: event="fail" → end in "failed" (reject)
    """
    events = ["ok"] * 4 + ["warn"] * 4 + ["fail"] * 2
    rows = []
    for i, ev in enumerate(events):
        rows.append({
            "trace_id": i,
            "step": 0,
            "event": ev,
            "x": float(x_offset + i),
            "label": ev != "fail",
        })
    return pd.DataFrame(rows)


def make_warned_traces_T(n=8, x_offset=0):
    """
    T traces for the warned spec: all single-step "warn" events.
      From "fresh":  fresh+warn → "warned" (accept)
      From "warned": warned+warn → "failed" (reject)
    The actual label depends on q_init_dist (recomputed by _dfa_labels).
    """
    rows = []
    for i in range(n):
        rows.append({
            "trace_id": i,
            "step": 0,
            "event": "warn",
            "x": float(x_offset + i),
            "label": True,  # placeholder; _dfa_labels recomputes this
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Group 1: _dfa_labels
# ---------------------------------------------------------------------------

def test_dfa_labels_all_accept():
    spec = make_safety_spec()
    df = make_safety_traces(n_accept=4, n_reject=0)

    labels, q_final = CompositionalAnalysisEngine._dfa_labels(
        df, spec, {"ok": 1.0}
    )

    assert np.all(labels == 1.0), f"Expected all 1.0, got {labels}"
    assert q_final == {"ok": 1.0}, f"Expected q_final={{'ok':1.0}}, got {q_final}"


def test_dfa_labels_all_reject():
    spec = make_safety_spec()
    df = make_safety_traces(n_accept=0, n_reject=4)

    labels, q_final = CompositionalAnalysisEngine._dfa_labels(
        df, spec, {"ok": 1.0}
    )

    assert np.all(labels == 0.0), f"Expected all 0.0, got {labels}"
    # No accepting traces → fallback to {spec start: 1.0}
    assert q_final == {spec._dfa.start: 1.0}, f"Unexpected q_final: {q_final}"


def test_dfa_labels_mixed():
    """6 accept + 2 reject → mean label == 0.75."""
    spec = make_safety_spec()
    df = make_safety_traces(n_accept=6, n_reject=2)

    labels, _ = CompositionalAnalysisEngine._dfa_labels(
        df, spec, {"ok": 1.0}
    )

    assert abs(np.mean(labels) - 0.75) < 1e-9, (
        f"Expected mean=0.75, got {np.mean(labels)}"
    )


def test_dfa_labels_fractional_from_mixed_q_init():
    """
    q_init_dist = {ok: 0.5, bad: 0.5}.
    A single all-ok_sym trace:
      from "ok":  ok+ok_sym→ok  (accept) → contribution 0.5
      from "bad": bad+ok_sym→bad (reject) → contribution 0.0
    Expected label = 0.5.
    """
    spec = make_safety_spec()
    df = make_safety_traces(n_accept=1, n_reject=0, n_steps=3)

    labels, _ = CompositionalAnalysisEngine._dfa_labels(
        df, spec, {"ok": 0.5, "bad": 0.5}
    )

    assert len(labels) == 1
    assert abs(labels[0] - 0.5) < 1e-9, f"Expected 0.5, got {labels[0]}"


def test_dfa_labels_q_final_dist_only_accepting_states():
    """
    q_final_dist must contain only states reached by *accepting* traces.
    Safety DFA: accepting traces end in "ok", rejecting traces end in "bad".
    So q_final_dist must be {"ok": 1.0} regardless of how many traces reject.
    """
    spec = make_safety_spec()
    df = make_safety_traces(n_accept=6, n_reject=4)

    _, q_final = CompositionalAnalysisEngine._dfa_labels(
        df, spec, {"ok": 1.0}
    )

    assert "bad" not in q_final, (
        "Rejecting final state 'bad' should not appear in q_final_dist"
    )
    assert abs(q_final.get("ok", 0.0) - 1.0) < 1e-9, (
        f"Expected q_final={{'ok':1.0}}, got {q_final}"
    )


# ---------------------------------------------------------------------------
# Group 2: _advance_q_dist_through_step
# ---------------------------------------------------------------------------

def test_advance_q_dist_all_accept():
    """All traces accept → returned dist is {start: 1.0} (all end in 'ok')."""
    spec = make_safety_spec()
    df = make_safety_traces(n_accept=8, n_reject=0)
    engine = make_engine({"S": df})

    result = engine._advance_q_dist_through_step(
        {"S": 1.0}, spec, {"ok": 1.0}
    )

    assert abs(result.get("ok", 0.0) - 1.0) < 1e-9, (
        f"Expected {{'ok':1.0}}, got {result}"
    )


def test_advance_q_dist_conditioned_on_acceptance():
    """
    Warned DFA, S traces:
      4 "ok" → end in "fresh"  (accept)
      4 "warn" → end in "warned" (accept)
      2 "fail" → end in "failed" (reject)

    q_final_dist conditioned on acceptance must be
    {"fresh": 0.5, "warned": 0.5} — "failed" state excluded.
    """
    spec = make_warned_spec()
    df = make_warned_traces_S()
    engine = make_engine({"S": df})

    result = engine._advance_q_dist_through_step(
        {"S": 1.0}, spec, {"fresh": 1.0}
    )

    assert "failed" not in result, (
        "Rejecting state 'failed' should not appear in q_final_dist"
    )
    assert abs(result.get("fresh", 0.0) - 0.5) < 1e-9, (
        f"Expected 'fresh': 0.5, got {result}"
    )
    assert abs(result.get("warned", 0.0) - 0.5) < 1e-9, (
        f"Expected 'warned': 0.5, got {result}"
    )


# ---------------------------------------------------------------------------
# Group 3: check_with_dfa, single step
# ---------------------------------------------------------------------------

def test_check_single_step_all_accept():
    spec = make_safety_spec()
    df = make_safety_traces(n_accept=10, n_reject=0)
    engine = make_engine({"S": df})

    rho, _ = engine.check_with_dfa(["S"], spec)
    assert abs(rho - 1.0) < 1e-9, f"Expected rho=1.0, got {rho}"


def test_check_single_step_all_reject():
    spec = make_safety_spec()
    df = make_safety_traces(n_accept=0, n_reject=10)
    engine = make_engine({"S": df})

    rho, uncertainty = engine.check_with_dfa(["S"], spec)
    assert rho == 0.0, f"Expected rho=0.0, got {rho}"
    assert uncertainty == 0.0, f"Expected uncertainty=0.0, got {uncertainty}"


def test_check_single_step_partial():
    """8/10 accept → rho == 0.8 exactly (mean of binary labels, no weighting)."""
    spec = make_safety_spec()
    df = make_safety_traces(n_accept=8, n_reject=2)
    engine = make_engine({"S": df})

    rho, _ = engine.check_with_dfa(["S"], spec)
    assert abs(rho - 0.8) < 1e-9, f"Expected rho=0.8, got {rho}"


# ---------------------------------------------------------------------------
# Group 4: check_with_dfa, two sequential steps
# ---------------------------------------------------------------------------

def test_check_two_steps_product_formula():
    """
    When S_last and T_first features are identically distributed, KDE
    importance weights ≈ 1 for all samples, so:
        rho_comp ≈ rho_S × rho_T = 0.8 × 0.6 = 0.48

    Tolerance is 0.10 to accommodate KDE approximation error.
    """
    spec = make_safety_spec()
    rng = np.random.default_rng(0)
    N = 20

    # S: 16 accept / 4 reject, feature x~N(0,1)
    xs_s = rng.normal(0, 1, N)
    s_rows = []
    for i in range(N):
        accept = i < 16
        for step in range(3):
            s_rows.append({
                "trace_id": i, "step": step,
                "event": "ok_sym" if (accept or step > 0) else "bad_sym",
                "x": xs_s[i],
                "label": accept,
            })
    df_s = pd.DataFrame(s_rows)

    # T: 12 accept / 8 reject, feature x drawn from same distribution
    xs_t = rng.normal(0, 1, N)
    t_rows = []
    for i in range(N):
        accept = i < 12
        for step in range(3):
            t_rows.append({
                "trace_id": i, "step": step,
                "event": "ok_sym" if (accept or step > 0) else "bad_sym",
                "x": xs_t[i],
                "label": accept,
            })
    df_t = pd.DataFrame(t_rows)

    engine = make_engine({"S": df_s, "T": df_t})
    rho, _ = engine.check_with_dfa(["S", "T"], spec, features=["x"])

    assert abs(rho - 0.48) < 0.10, (
        f"Expected rho ≈ 0.48 (product formula), got {rho:.4f}"
    )


def test_check_two_steps_dfa_state_propagation():
    """
    Warned DFA propagation test.

    S traces:
      4 "ok"   → end in "fresh"  (accept)   rho_S = 0.8
      4 "warn" → end in "warned" (accept)
      2 "fail" → end in "failed" (reject)
    q_final_dist (conditioned on acceptance) = {fresh: 0.5, warned: 0.5}

    T traces: all single-step "warn" events.
      Evaluated with q_init = {fresh: 0.5, warned: 0.5}:
        from "fresh":  fresh+warn → "warned" (accept) → +0.5
        from "warned": warned+warn → "failed" (reject) → +0.0
        label per T trace = 0.5

    rho_comp = rho_S × rho_T = 0.8 × 0.5 = 0.4

    Without DFA propagation (naive q_init = {fresh: 1.0} for T):
      label per T trace = 1.0 → rho_T = 1.0 → rho_comp = 0.8
    This test distinguishes correct propagation (≈ 0.4) from naive (≈ 0.8).
    """
    spec = make_warned_spec()
    df_s = make_warned_traces_S()
    df_t = make_warned_traces_T(n=8)
    engine = make_engine({"S": df_s, "T": df_t})

    rho, _ = engine.check_with_dfa(["S", "T"], spec, features=["x"])

    assert abs(rho - 0.4) < 0.05, (
        f"Expected rho ≈ 0.4 (DFA propagation). "
        f"Got {rho:.4f} — if ≈ 0.8, DFA state is not being propagated."
    )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_dfa_labels_all_accept,
        test_dfa_labels_all_reject,
        test_dfa_labels_mixed,
        test_dfa_labels_fractional_from_mixed_q_init,
        test_dfa_labels_q_final_dist_only_accepting_states,
        test_advance_q_dist_all_accept,
        test_advance_q_dist_conditioned_on_acceptance,
        test_check_single_step_all_accept,
        test_check_single_step_all_reject,
        test_check_single_step_partial,
        test_check_two_steps_product_formula,
        test_check_two_steps_dfa_state_propagation,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
