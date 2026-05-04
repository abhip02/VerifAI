# Composed 4-way intersection scenario for compositional analysis.
#
# Reuses the lane/maneuver discovery code from intersection_base.scenic to
# spawn the ego near a 4-way intersection, then exposes 3 leaf behaviors
# composed via `do choose`. The behaviors are flat (only `take` actions, no
# inner `do`) so the scenic_composition_analysis parser registers them as
# leaf primitives.
#
# Composition: Main = do choose { GoStraight, TurnLeft, TurnRight }
# DFA-side intent: every primitive brakes mid-maneuver for `n_brake` steps
# sampled per scene from Uniform(1..5). With K=2 in the DFA, n_brake in {1,2}
# accepts and n_brake in {3,4,5} rejects, so each primitive's rho is graded
# (~0.4 in expectation, finite-N noise around that).

param map = localPath('../../../../../tests/scenic/scenic_tests/cases_realistic/CARLA/Town07.xodr')
param carla_map = localPath('../../../../../tests/scenic/scenic_tests/cases_realistic/CARLA/Town07.xodr')
param timestep = 0.1
param use2DMap = True
param render = 0

model scenic.simulators.metadrive.model

from scenic.domains.driving.actions import (
    SetThrottleAction,
    SetBrakeAction,
    SetSteerAction,
)

CRUISE_THROTTLE = 0.4
DISTANCE_TO_INTERSECTION = Range(-15, -5)

# --- Spawn-lane discovery (mirrors intersection_base.scenic) ---
fourWayIntersection = filter(lambda i: i.is4Way, network.intersections)
intersec = fourWayIntersection[0]
rightLanes = filter(
    lambda lane: all([section._laneToRight is None for section in lane.sections]),
    intersec.incomingLanes,
)
startLane = rightLanes[0]
uberSpawnPoint = startLane.centerline[-1]



# --- 3 flat leaf behaviors (no `do` — required for the composition parser) ---

# Multi-axis per-scene randomness so rho can't be read off the distribution
# directly. Each behavior samples four independent values when its body starts:
#   pre_cruise:   how many cruise steps before braking (shifts brake onset
#                 in/out of the WARMUP_STEPS window)
#   n_brake:      brake duration
#   brake_force:  brake intensity (continuous in [0.3, 1.0])
#   throttle:     cruise/recovery throttle (faster recovery vs slower)
# The interaction of all four with WARMUP_STEPS=10 and STOP_THRESHOLD=0.5 gives
# a non-trivial rho per trace; all three primitives share the same parameter
# distributions, so any rho difference between them comes from steer-induced
# physics (turns dissipate kinetic energy differently than straights), not
# from the discrete brake schedule.

behavior GoStraight():
    pre_cruise  = Uniform(1, 2, 3, 4, 5)
    n_brake     = Uniform(1, 2, 3, 4, 5, 6)
    brake_force = Range(0.3, 1.0)
    throttle    = Range(0.3, 0.6)
    for i in range(pre_cruise):
        take SetThrottleAction(throttle), SetBrakeAction(0), SetSteerAction(0)
    for i in range(n_brake):
        take SetThrottleAction(0), SetBrakeAction(brake_force), SetSteerAction(0)
    while True:
        take SetThrottleAction(throttle), SetBrakeAction(0), SetSteerAction(0)


behavior TurnLeft():
    pre_cruise  = Uniform(1, 2, 3, 4, 5)
    n_brake     = Uniform(1, 2, 3, 4, 5, 6)
    brake_force = Range(0.3, 1.0)
    throttle    = Range(0.3, 0.6)
    for i in range(pre_cruise):
        take SetThrottleAction(throttle), SetBrakeAction(0), SetSteerAction(-0.3)
    for i in range(n_brake):
        take SetThrottleAction(0), SetBrakeAction(brake_force), SetSteerAction(-0.3)
    while True:
        take SetThrottleAction(throttle), SetBrakeAction(0), SetSteerAction(-0.3)


behavior TurnRight():
    pre_cruise  = Uniform(1, 2, 3, 4, 5)
    n_brake     = Uniform(1, 2, 3, 4, 5, 6)
    brake_force = Range(0.3, 1.0)
    throttle    = Range(0.3, 0.6)
    for i in range(pre_cruise):
        take SetThrottleAction(throttle), SetBrakeAction(0), SetSteerAction(0.3)
    for i in range(n_brake):
        take SetThrottleAction(0), SetBrakeAction(brake_force), SetSteerAction(0.3)
    while True:
        take SetThrottleAction(throttle), SetBrakeAction(0), SetSteerAction(0.3)


# --- Compositional entrypoint ---
# `do choose { Behavior(): ... }` is rejected at simulate time by Scenic
# (choose expects sub-scenarios) — but the composition-analysis parser only
# inspects the AST and treats this as a 1-step random choice over leaf
# behaviors. Main is therefore parser-only: it is never compiled as a
# scenario for simulation. The compositional pipeline runs each primitive
# in isolation via generate_graph_traces (per-primitive wrappers).

scenario Main():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for DISTANCE_TO_INTERSECTION,
              with behavior GoStraight()
    compose:
        do choose {
            GoStraight(): 1,
            TurnLeft():   1,
            TurnRight():  1,
        }
