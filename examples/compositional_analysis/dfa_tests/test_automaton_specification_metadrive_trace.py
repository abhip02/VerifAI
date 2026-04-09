"""
examples/compositional_analysis/dfa_tests/test_automaton_specification_metadrive_trace.py

Unit tests for automaton_specification using a real MetaDrive trace.
Trace is loaded from example_trace.csv in the same directory.

Run from the repo root:
    python examples/compositional_analysis/dfa_tests/test_automaton_specification_metadrive_trace.py
or with pytest:
    pytest examples/compositional_analysis/dfa_tests/test_automaton_specification_metadrive_trace.py -v
"""

import os
import csv

from verifai.monitor import automaton_specification

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


# DFA: "never exceed speed limit"
# States: ok (start, accepting), fail (sink, rejecting)
# ok --fast--> fail; fail absorbs everything

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
    q_final = monitor.advance_on_trace(trace[:1], start="ok")
    assert q_final == "fail", (
        f"Expected 'fail' after step 0 (speed={trace[0]['speed']:.2f}), "
        f"got '{q_final}'"
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
    q_final = monitor.advance_on_trace(trace[:50], start="ok")
    assert q_final == "ok", (
        f"Expected 'ok' at step 50, got '{q_final}'"
    )
    print("PASS  test_speed_limit_25_still_ok_midtrace  "
          "(DFA in 'ok' after 50 steps)")


def test_incremental_monitoring():
    """
    Process the trace in two chunks and verify the final state matches
    processing it all at once.
    """
    trace = parse_trace(TRACE_PATH)
    monitor = make_speed_monitor(speed_limit=15.0)

    # All at once
    result_full = monitor.evaluate(trace)

    # Incrementally: first half, then second half
    mid = len(trace) // 2
    q_mid = monitor.advance_on_trace(trace[:mid], start="ok")
    q_final = monitor.advance_on_trace(trace[mid:], start=q_mid)
    result_incremental = 1.0 if monitor._dfa._label(q_final) else -1.0

    assert result_full == result_incremental, (
        f"Incremental ({result_incremental}) != full ({result_full})"
    )
    print(f"PASS  test_incremental_monitoring  "
          f"(both give {result_full}, mid-state='{q_mid}')")


def test_fail_is_absorbing():
    """
    Once the DFA enters 'fail', continuing the trace cannot recover.
    """
    trace = parse_trace(TRACE_PATH)
    monitor = make_speed_monitor(speed_limit=15.0)

    # First step violates (speed > 15)
    q_after_one = monitor.advance_on_trace(trace[:1], start="ok")
    assert q_after_one == "fail"

    # Rest of the trace cannot escape 'fail'
    q_final = monitor.advance_on_trace(trace[1:], start=q_after_one)
    assert q_final == "fail", (
        f"Expected 'fail' (absorbing), got '{q_final}'"
    )
    assert not monitor._dfa._label(q_final)
    print("PASS  test_fail_is_absorbing")


def test_advance_from_custom_start():
    """
    Starting advance_on_trace from 'fail' stays in 'fail' regardless
    of trace content.
    """
    trace = parse_trace(TRACE_PATH)
    monitor = make_speed_monitor(speed_limit=25.0)  # trace complies at 25

    # Even a compliant trace can't escape 'fail' if we start there
    q_final = monitor.advance_on_trace(trace, start="fail")
    assert q_final == "fail", (
        f"Expected 'fail' when starting from 'fail', got '{q_final}'"
    )
    print("PASS  test_advance_from_custom_start")


if __name__ == "__main__":
    tests = [
        test_speed_limit_15_violated,
        test_speed_limit_25_compliant,
        test_speed_limit_15_violation_step,
        test_speed_limit_25_still_ok_midtrace,
        test_incremental_monitoring,
        test_fail_is_absorbing,
        test_advance_from_custom_start,
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