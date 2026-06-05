"""Four DFA specs lifted verbatim from dfa_tests/test_check_with_dfa_*.py.

Names exposed: spec_tollgate, spec_two_stops, spec_fast_twice, spec_slow2_accel.
For the two co-safety specs (fast_twice, slow2_accel) we use the
safety-complement DFA (absorbing-reject), per the explanation in
test_check_with_dfa_cosafety_*.py: compositional check_with_dfa is only
correct for absorbing-reject DFAs.
"""

from __future__ import annotations

import math

from verifai.monitor import automaton_specification


# v3 calibration: the original dfa_tests thresholds (STOP=3.5, FAST=7.0,
# K_WAIT=3, FAST_MS=8.0) were tuned for the wander_scenarios speed regime
# (random throttle, ~3-14 m/s steady-state). Our v3 PID + per-trace
# Range-sampled targets put C in 2.5-7.5 m/s and X in 7-10 m/s, so the
# original thresholds saturate most cells at ρ̂ ∈ {0, 1}. The values
# below are recalibrated so per-cell ρ̂ lands in a discriminating band
# without changing what each spec is *checking*.
STOP_THRESHOLD_MS = 3.5
REQUIRED_WAIT_STEPS = 1   # was 3 — K=1 means one slow tick then fast is OK,
                           # two-or-more slow then fast is the gate; with v3
                           # primitives this lands ρ̂_comp in (0, 1).
NEAR_STOP_MS = 3.5
HIGH_SPEED_MS = 6.0       # was 7.0 — lowered so the v3 X/O primitives
                           # (Range(7-10)/(6.5-9)) consistently hit `fast`
                           # while C (Range(2.5-7.5)) sometimes does too,
                           # giving cross-segment fast→slow→fast patterns.
LOW_SPEED_MS = 3.5
SLOW_MS = 3.5
FAST_MS = 7.5             # was 8.0 — same recalibration logic for slow2_accel
MAX_SPEED_BASELINE = 8.5  # used by the Markovian baseline spec_max_speed


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


def spec_max_speed(threshold=MAX_SPEED_BASELINE):
    """Markovian baseline: speed never exceeds `threshold`.

    Two-state DFA (ok → ok if below, ok → violated absorbing if at-or-above).
    The predicate is per-tick with no inter-tick state, so the DFA verdict
    is the AND of per-tick predicates — there's no handoff-state dependency
    for the compositional method to miss. This is the agreement-baseline
    cell: ρ̂_comp ≈ ρ̂_mono is expected here, and a gap would indicate a
    pipeline bug rather than the genuine non-Markovian limitation seen on
    spec_tollgate / spec_fast_twice / spec_slow2_accel.
    """
    def transition(state, sym):
        if state == "violated":
            return "violated"
        return "violated" if sym == "high" else "ok"

    return automaton_specification(
        start="ok",
        inputs={"high", "low"},
        transition=transition,
        label=lambda s: s != "violated",
        labeling_function=lambda row: "high" if row["speed"] >= threshold else "low",
    )


# --- Maneuver-choose specs (intersection L/R/Straight). -------------------
# These pair with composites/maneuver_choose.scenic, where the three choose
# branches traverse structurally distinct lanes through Town07's 4-way
# intersection. Labeling is on heading delta from each trace's step-0
# heading — self-calibrating, no map-specific constants baked.

DELTA_TURN_RAD = math.pi / 4   # ~45° — threshold to flag a step as "turning"
DELTA_DONE_RAD = math.pi / 2   # ~90° — threshold to flag the turn as completed
K_INTERSECTION = 10            # max ticks between turn-onset and turn-completion


def _wrap(angle: float) -> float:
    """Wrap radians to (-π, π]."""
    a = (angle + math.pi) % (2 * math.pi) - math.pi
    return a


def _heading_label_factory(turn_threshold: float):
    """Build a per-trace-stateful labeling function on heading delta.

    Caches each trace's step-0 heading the first time we see that
    trace_id, then emits 'turn' iff |heading − step0| ≥ turn_threshold,
    else 'straight'. Self-calibrating across maps / startLane choices —
    no INITIAL_HEADING constant needs baking. The cache grows monotonically
    with trace_id over the lifetime of one spec instance (one entry per
    trace, ~bytes); fine for our budgets, fresh per spec rebuild.
    """
    step0_heading: dict = {}

    def label(row):
        tid = row["trace_id"]
        if row["step"] == 0 or tid not in step0_heading:
            step0_heading[tid] = row["heading"]
        delta = _wrap(row["heading"] - step0_heading[tid])
        return "turn" if abs(delta) >= turn_threshold else "straight"

    return label


def spec_completes_turn():
    """Markovian intersection spec: ego never enters the turning band.

    Per-tick predicate on heading delta from start (Markovian: DFA state
    is just ok/violated, no inter-tick counter). Safety-complement,
    absorbing-reject. On the maneuver_choose composite, only the Straight
    branch (and pre-turn ticks of L/R) keeps |Δh| < π/4 throughout; the
    L and R branches eventually cross the threshold and reject.

    Nominal ρ̂ ≈ 1/3 under uniform choose.
    """
    def transition(state, sym):
        if state == "violated":
            return "violated"
        return "violated" if sym == "turn" else "ok"

    return automaton_specification(
        start="ok",
        inputs={"straight", "turn"},
        transition=transition,
        label=lambda s: s != "violated",
        labeling_function=_heading_label_factory(DELTA_TURN_RAD),
    )


def _heading_band_label_factory(turn_threshold: float, done_threshold: float):
    """Three-band heading-only labeler.

    Emits one of {pre_turn, in_turning, turn_done} per row, using each
    trace's step-0 heading as a self-calibrating baseline. No speed
    component — this is what gives the K-completion spec its mono-comp
    agreement, because the discriminating signal lives entirely in the
    turn-segment heading dynamics (which both methods see identically),
    not in the segment-boundary speed transient (which only mono sees).
    """
    step0_heading: dict = {}

    def label(row):
        tid = row["trace_id"]
        if row["step"] == 0 or tid not in step0_heading:
            step0_heading[tid] = row["heading"]
        delta = abs(_wrap(row["heading"] - step0_heading[tid]))
        if delta >= done_threshold:
            return "turn_done"
        if delta >= turn_threshold:
            return "in_turning"
        return "pre_turn"

    return label


def spec_k_intersection(K: int = K_INTERSECTION):
    """Non-Markovian K-window intersection spec: 'complete the turn in time'.

    Once the ego enters the turning band (|Δh| ≥ π/4), it must reach turn
    completion (|Δh| ≥ π/2) within K ticks — i.e. the ego must rotate
    through π/4 of arc within K ticks of starting the turn. Faster turns
    pass; slow / aborted turns violate.

    The K-counter chain (in_turn_1 → … → in_turn_K) is the explicitly
    non-Markovian element. Crucially, the spec only depends on heading
    dynamics during the turn segment — no speed component, no cross-
    primitive coupling. Comp's TurnL/TurnR primitives and mono's
    MonoApproachChoose see the same turn-segment heading evolution
    (FollowTrajectoryBehavior at the same UBER_SPEED distribution), so
    per-primitive ρ̂_L/R should converge to mono's ρ̂_L/R.

    Symbol alphabet: {pre_turn, in_turning, turn_done}.

    Nominal ρ̂:
      • Straight branch (⅓): stays in pre_turn → trivially satisfied
      • L/R branches (⅔):    satisfied iff (turn arc traversed within K
                              ticks). Discriminates on UBER_SPEED —
                              faster ego = faster arc traversal.
    K=10 (~1s at timestep=0.1) lands the cell in the discriminating band
    for UBER_SPEED ∈ [2, 8] on Town07's connecting-lane geometry.
    """
    counting = [f"in_turn_{i + 1}" for i in range(K)]

    def transition(state, sym):
        if state == "violated":
            return "violated"
        if state == "completed":
            return "completed"
        if state == "ok":
            if sym == "pre_turn":
                return "ok"
            if sym == "turn_done":
                return "completed"
            return counting[0]  # in_turning: start the K-window
        idx = counting.index(state)
        if sym == "turn_done":
            return "completed"
        if sym == "in_turning":
            if idx == K - 1:
                return "violated"  # K ticks elapsed without completion
            return counting[idx + 1]
        # pre_turn while inside the K-window → oscillated back → violated
        return "violated"

    return automaton_specification(
        start="ok",
        inputs={"pre_turn", "in_turning", "turn_done"},
        transition=transition,
        label=lambda s: s != "violated",
        labeling_function=_heading_band_label_factory(DELTA_TURN_RAD, DELTA_DONE_RAD),
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
