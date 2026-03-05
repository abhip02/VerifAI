"""
examples/compositional_analysis/dfa_test.py

Unit tests for automaton_specification.
Run from the repo root:
    python examples/compositional_analysis/dfa_test.py
or with pytest:
    pytest examples/compositional_analysis/dfa_test.py -v
"""

import sys
import os

# Allow running from repo root or from this directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from verifai.monitor import automaton_specification


# ---------------------------------------------------------------------------
# Shared DFA definition
# ---------------------------------------------------------------------------
# Spec: "the vehicle must eventually reach the goal, and must never crash first"
#
# States:  q0 (start, non-accepting)
#          q1 (goal reached, accepting)
#          q2 (crashed, sink/reject)
#
# Alphabet: {"safe", "goal", "crash"}
#
#   q0 --safe-->  q0
#   q0 --goal-->  q1
#   q0 --crash--> q2
#   q1 --*-->     q1   (absorbing accept)
#   q2 --*-->     q2   (absorbing reject)

def _transition(state, symbol):
    if state == "q0":
        if symbol == "goal":
            return "q1"
        elif symbol == "crash":
            return "q2"
        else:
            return "q0"
    return state  # q1 and q2 are absorbing


def _label(state):
    return state == "q1"


def _labeling_fn(mdp_state: dict) -> str:
    """
    Toy labeling function.
    MDP state is a dict with at minimum {"event": str}.
    Maps directly to the DFA alphabet symbol.
    """
    return mdp_state["event"]


def make_monitor():
    return automaton_specification(
        start="q0",
        inputs={"safe", "goal", "crash"},
        transition=_transition,
        label=_label,
        labeling_function=_labeling_fn,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_success():
    """safe → safe → goal  should be accepted."""
    monitor = make_monitor()
    traj = [
        {"event": "safe"},
        {"event": "safe"},
        {"event": "goal"},
    ]
    assert monitor.evaluate(traj) == 1.0, "Expected acceptance (+1.0)"
    print("PASS  test_success")


def test_crash_before_goal():
    """safe → crash → goal  should be rejected (crash is absorbing)."""
    monitor = make_monitor()
    traj = [
        {"event": "safe"},
        {"event": "crash"},
        {"event": "goal"},
    ]
    assert monitor.evaluate(traj) == -1.0, "Expected rejection (-1.0)"
    print("PASS  test_crash_before_goal")


def test_no_goal():
    """Trace that never reaches goal should be rejected."""
    monitor = make_monitor()
    traj = [
        {"event": "safe"},
        {"event": "safe"},
    ]
    assert monitor.evaluate(traj) == -1.0, "Expected rejection (-1.0)"
    print("PASS  test_no_goal")


def test_immediate_goal():
    """Single-step trace that immediately hits goal should be accepted."""
    monitor = make_monitor()
    traj = [{"event": "goal"}]
    assert monitor.evaluate(traj) == 1.0, "Expected acceptance (+1.0)"
    print("PASS  test_immediate_goal")


def test_advance_on_trace_intermediate_state():
    """
    Partially consume the trace with advance_on_trace.
    After two 'safe' steps the DFA should still be at q0 (not accepting).
    """
    monitor = make_monitor()
    partial = [{"event": "safe"}, {"event": "safe"}]
    advanced = monitor.advance_on_trace(partial)

    assert advanced.start == "q0", f"Expected q0, got {advanced.start}"
    assert not advanced.label([]), "q0 should not be accepting"
    print("PASS  test_advance_on_trace_intermediate_state")


def test_advance_on_trace_then_goal():
    """
    Advance partway, then push one more 'goal' step and confirm acceptance.
    This tests the incremental / online monitoring use case.
    """
    monitor = make_monitor()
    partial = [{"event": "safe"}, {"event": "safe"}]
    advanced = monitor.advance_on_trace(partial)

    # One more step: the raw symbol (already labeled) goes straight into dfa.advance
    final = advanced.advance([_labeling_fn({"event": "goal"})])
    assert final.start == "q1", f"Expected q1, got {final.start}"
    assert final.label([]), "q1 should be accepting"
    print("PASS  test_advance_on_trace_then_goal")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_success,
        test_crash_before_goal,
        test_no_goal,
        test_immediate_goal,
        test_advance_on_trace_intermediate_state,
        test_advance_on_trace_then_goal,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    sys.exit(0 if failed == 0 else 1)