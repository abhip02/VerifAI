"""at_most_one_brake DFA: at most one debounced slow->fast transition.

A "brake episode" is a contiguous slow run that ends in a fast step. The
DFA counts these episodes (debounced via the in-slow state) and rejects
after the second one. Lifted from
``test_4way_intersection_wander_scenarios.py::_make_spec_at_most_K_brake_episodes(1)``
so the wall-clock sweep tool can load it via ``--spec_module``.

Loaded by ``compare_budget_sweep.py`` via ``importlib.util``.
"""

from verifai.monitor import automaton_specification


WARMUP_STEPS = 5
# Positioned inside the cruising-speed band (~3–14 m/s) so the per-segment brake
# count varies across traces; a boundary outside that band makes every segment's
# verdict identical and the compositional convergence curve goes flat. Demo
# cutoff, not a literal stop speed. Kept in sync with _STOP_THRESHOLD in
# compare_budget_sweep.py.
STOP_THRESHOLD = 6.0      # m/s; below this counts as "slow"
MAX_BRAKE_EPISODES = 1    # K in the spec


def make_spec():
    K = MAX_BRAKE_EPISODES

    def transition(state, sym):
        if state == "violated":
            return "violated"
        if state.endswith("_in_slow"):
            n = int(state[1:state.index("_")])
            if sym == "slow":
                return state          # still slow, episode in progress
            n += 1                    # slow -> fast: episode just ended
            return "violated" if n > K else f"q{n}"
        # not currently in a slow run
        n = int(state[1:])
        return f"q{n}_in_slow" if sym == "slow" else state

    def label_row(row):
        if row["step"] < WARMUP_STEPS:
            return "fast"
        return "slow" if row["speed"] < STOP_THRESHOLD else "fast"

    return automaton_specification(
        start="q0",
        inputs={"slow", "fast"},
        transition=transition,
        label=lambda s: s != "violated",
        labeling_function=label_row,
    )
