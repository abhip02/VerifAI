"""
examples/compositional_analysis/dfa_test.py

Unit tests for automaton_specification using a real MetaDrive trace.
Trace is loaded from dfa_test_example_trace.csv in the same directory.

Run from the repo root:
    python examples/compositional_analysis/dfa_test.py
or with pytest:
    pytest examples/compositional_analysis/dfa_test.py -v
"""

import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from verifai.monitor import automaton_specification

# ---------------------------------------------------------------------------
# Load trace from sibling CSV
# ---------------------------------------------------------------------------

TRACE_PATH = os.path.join(os.path.dirname(__file__), "example_trace.csv")


def parse_trace(path: str) -> list[dict]:
    """Parse the MetaDrive CSV trace into a list of row dicts."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [
            {
                "trace_id": int(row["trace_id"]),
                "step":     int(row["step"]),
                "x":        float(row["x"]),
                "y":        float(row["y"]),
                "heading":  float(row["heading"]),
                "speed":    float(row["speed"]),
                "reward":   float(row["reward"]),
            }
            for row in reader
        ]


# ---------------------------------------------------------------------------
# DFA: "never exceed speed limit"
#
# Two states:
#   ok   (start, accepting) — speed limit respected so far
#   fail (sink, rejecting)  — limit was exceeded at some point
#
# Alphabet: {"ok", "fast"}
#
#   ok   --ok-->   ok
#   ok   --fast--> fail
#   fail --*-->    fail   (absorbing)
# ---------------------------------------------------------------------------

def _speed_transition(state, symbol):
    if state == "ok" and symbol == "fast":
        return "fail"
    return state  # ok→ok on "ok", fail absorbs everything


def _speed_label(state):
    return state == "ok"


def make_speed_monitor(speed_limit: float):
    def labeling_fn(mdp_state: dict) -> str:
        return "fast" if mdp_state["speed"] > speed_limit else "ok"

    return automaton_specification(
        start="ok",
        inputs={"ok", "fast"},
        transition=_speed_transition,
        label=_speed_label,
        labeling_function=labeling_fn,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_speed_limit_15_violated():
    """
    Real MetaDrive trace with a 15 m/s limit.
    Steps 0-3 have speeds of ~19.9, 18.2, 16.7, 15.5 m/s — all violations.
    The DFA should reject.
    """
    trace = parse_trace(TRACE_PATH)
    monitor = make_speed_monitor(speed_limit=15.0)
    result = monitor.evaluate(trace)
    assert result == -1.0, f"Expected rejection (-1.0), got {result}"
    print(f"PASS  test_speed_limit_15_violated  "
          f"(step 0 speed={trace[0]['speed']:.2f} > 15.0 m/s)")


def test_speed_limit_25_compliant():
    """
    Real MetaDrive trace with a 25 m/s limit.
    Max speed in trace is ~19.9 m/s — never violated.
    The DFA should accept.
    """
    trace = parse_trace(TRACE_PATH)
    monitor = make_speed_monitor(speed_limit=25.0)
    result = monitor.evaluate(trace)
    assert result == 1.0, f"Expected acceptance (+1.0), got {result}"
    max_speed = max(r["speed"] for r in trace)
    print(f"PASS  test_speed_limit_25_compliant  "
          f"(max speed={max_speed:.2f} < 25.0 m/s)")


def test_speed_limit_15_violation_step():
    """
    Confirm the DFA transitions to 'fail' exactly at step 0 (first step)
    using advance_on_trace on just the first row.
    """
    trace = parse_trace(TRACE_PATH)
    monitor = make_speed_monitor(speed_limit=15.0)
    advanced = monitor.advance_on_trace(trace[:1])
    assert advanced.start == "fail", (
        f"Expected 'fail' after step 0 (speed={trace[0]['speed']:.2f}), "
        f"got '{advanced.start}'"
    )
    print(f"PASS  test_speed_limit_15_violation_step  "
          f"(DFA in 'fail' after step 0, speed={trace[0]['speed']:.2f})")


def test_speed_limit_25_still_ok_midtrace():
    """
    After consuming the first 50 steps under a 25 m/s limit,
    the DFA should still be in the 'ok' (accepting) state.
    """
    trace = parse_trace(TRACE_PATH)
    monitor = make_speed_monitor(speed_limit=25.0)
    advanced = monitor.advance_on_trace(trace[:50])
    assert advanced.start == "ok", (
        f"Expected 'ok' at step 50, got '{advanced.start}'"
    )
    print("PASS  test_speed_limit_25_still_ok_midtrace  "
          "(DFA in 'ok' after 50 steps)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_speed_limit_15_violated,
        test_speed_limit_25_compliant,
        test_speed_limit_15_violation_step,
        test_speed_limit_25_still_ok_midtrace,
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
            import traceback
            print(f"ERROR {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    sys.exit(0 if failed == 0 else 1)