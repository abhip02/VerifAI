"""Four DFA specs lifted verbatim from dfa_tests/test_check_with_dfa_*.py.

Names exposed: spec_tollgate, spec_two_stops, spec_fast_twice, spec_slow2_accel.
For the two co-safety specs (fast_twice, slow2_accel) we use the
safety-complement DFA (absorbing-reject), per the explanation in
test_check_with_dfa_cosafety_*.py: compositional check_with_dfa is only
correct for absorbing-reject DFAs.
"""

from __future__ import annotations

from verifai.monitor import automaton_specification


STOP_THRESHOLD_MS = 3.5
REQUIRED_WAIT_STEPS = 3
NEAR_STOP_MS = 3.5
HIGH_SPEED_MS = 7.0
LOW_SPEED_MS = 3.5
SLOW_MS = 3.5
FAST_MS = 8.0


def spec_tollgate():
    """Tollgate mandatory-wait safety spec (K=3).

    Once speed drops below 3.5 m/s, vehicle must remain slow for at least 3
    consecutive steps before speeding up again. Lifted from
    dfa_tests/test_check_with_dfa_metadrive_tollgate.py.
    """
    K = REQUIRED_WAIT_STEPS
    wait_states = [f"wait_{i+1}" for i in range(K)]

    def transition(state, sym):
        if state == "moving":
            return "wait_1" if sym == "slow" else "moving"
        if state == "violated":
            return "violated"
        idx = wait_states.index(state)
        if sym == "slow":
            return "moving" if idx == K - 1 else wait_states[idx + 1]
        return "violated"

    return automaton_specification(
        start="moving",
        inputs={"slow", "fast"},
        transition=transition,
        label=lambda s: s != "violated",
        labeling_function=lambda row: "slow" if row["speed"] < STOP_THRESHOLD_MS else "fast",
    )


def spec_two_stops():
    """At most one near-stop episode; a second near_stop is a violation.

    Lifted from dfa_tests/test_check_with_dfa_metadrive_two_stops.py.
    """
    def transition(state, sym):
        if state == "moving":
            return "stopped_once" if sym == "near_stop" else "moving"
        if state == "stopped_once":
            return "stopped_twice" if sym == "near_stop" else "stopped_once"
        return "stopped_twice"

    return automaton_specification(
        start="moving",
        inputs={"moving", "near_stop"},
        transition=transition,
        label=lambda s: s != "stopped_twice",
        labeling_function=lambda row: "near_stop" if row["speed"] < NEAR_STOP_MS else "moving",
    )


def _fast_twice_sym(row):
    spd = row["speed"]
    if spd >= HIGH_SPEED_MS:
        return "fast"
    if spd <= LOW_SPEED_MS:
        return "slow"
    return "mid"


def spec_fast_twice():
    """Safety complement of 'eventually fast→slow→fast (V-shape)'.

    Once the vehicle has been fast, dipped slow, then goes fast again, the
    property is permanently violated. Absorbing-reject form — the one
    compositional check_with_dfa handles correctly. Lifted from
    dfa_tests/test_check_with_dfa_cosafety_fast_twice.py::make_safety_spec.
    """
    def transition(state, sym):
        if state == "violated":
            return "violated"
        if state == "ok":
            return "was_fast" if sym == "fast" else "ok"
        if state == "was_fast":
            return "was_slow" if sym == "slow" else "was_fast"
        return "violated" if sym == "fast" else "was_slow"

    return automaton_specification(
        start="ok",
        inputs={"slow", "mid", "fast"},
        transition=transition,
        label=lambda s: s != "violated",
        labeling_function=_fast_twice_sym,
    )


def _slow2_accel_sym(row):
    spd = row["speed"]
    if spd < SLOW_MS:
        return "slow"
    if spd >= FAST_MS:
        return "fast"
    return "mid"


def spec_slow2_accel():
    """Safety complement of 'eventually slow≥2 steps then immediately fast'.

    Once slow for two consecutive steps, going fast is a violation. Absorbing-
    reject form. Lifted from
    dfa_tests/test_check_with_dfa_cosafety_slow2_accel.py::make_safety_spec.
    """
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
            return "slow_1"
        if sym == "fast":
            return "violated"
        if sym == "slow":
            return "slow_2"
        return "slow_2"

    return automaton_specification(
        start="ok",
        inputs={"slow", "mid", "fast"},
        transition=transition,
        label=lambda s: s != "violated",
        labeling_function=_slow2_accel_sym,
    )
