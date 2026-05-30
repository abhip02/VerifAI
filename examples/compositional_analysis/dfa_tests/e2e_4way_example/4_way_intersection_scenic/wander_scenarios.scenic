# Wander composition over self-contained SCENARIOS (not just behaviors).
#
# Mirrors composed_wander.scenic but wraps each leaf behavior in its own
# `scenario X():` block whose setup creates its own ego. Same compositional
# semantics (5x do choose over 4 primitives = 4^5 = 1024 paths), same
# monolithic counterpart, same multi-spec evaluation downstream.
#
# Why scenarios instead of bare behaviors:
#   - Demonstrates that compositional analysis works when the leaf primitives
#     are full Scenic scenarios (each with their own ego setup), not just
#     behaviors attached to a Main-owned ego. This is the "rich" composition
#     path used in test_4way_intersection_scenarios (the 2-step intersection
#     test), now generalized to 5 sequential decisions.
#   - Each scenario starts a FRESH ego at the same spawn point. The
#     compositional engine bridges segment boundaries via KDE on speed only
#     (position resets between segments since each scenario re-spawns; only
#     the speed feature is meaningfully shared at boundaries).

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

DISTANCE_TO_INTERSECTION = Range(-15, -5)

# --- Spawn-lane discovery (same as composed_wander.scenic) ---
fourWayIntersection = filter(lambda i: i.is4Way, network.intersections)
intersec = fourWayIntersection[0]
rightLanes = filter(
    lambda lane: all([section._laneToRight is None for section in lane.sections]),
    intersec.incomingLanes,
)
startLane = rightLanes[0]
uberSpawnPoint = startLane.centerline[-1]


# --- 4 leaf behaviors (same bodies as composed_wander.scenic) ---
# Each begins with a cruise PREWARM of variable length so per-primitive trace
# row 0 can be a warm state with varied speed (after PREWARM_TRIM removes the
# prewarm prefix).

behavior GoStraightBehavior():
    prewarm  = Uniform(0, 5, 10, 15, 25, 35)
    prewarm_throttle = Range(0.4, 0.8)
    throttle = Range(0.3, 0.6)
    for i in range(prewarm):
        take SetThrottleAction(prewarm_throttle), SetBrakeAction(0), SetSteerAction(0)
    while True:
        take SetThrottleAction(throttle), SetBrakeAction(0), SetSteerAction(0)


behavior TurnLeftBehavior():
    prewarm  = Uniform(0, 5, 10, 15, 25, 35)
    prewarm_throttle = Range(0.4, 0.8)
    throttle = Range(0.3, 0.6)
    steer    = Range(-0.4, -0.2)
    for i in range(prewarm):
        take SetThrottleAction(prewarm_throttle), SetBrakeAction(0), SetSteerAction(0)
    while True:
        take SetThrottleAction(throttle), SetBrakeAction(0), SetSteerAction(steer)


behavior TurnRightBehavior():
    prewarm  = Uniform(0, 5, 10, 15, 25, 35)
    prewarm_throttle = Range(0.4, 0.8)
    throttle = Range(0.3, 0.6)
    steer    = Range(0.2, 0.4)
    for i in range(prewarm):
        take SetThrottleAction(prewarm_throttle), SetBrakeAction(0), SetSteerAction(0)
    while True:
        take SetThrottleAction(throttle), SetBrakeAction(0), SetSteerAction(steer)


behavior BrakeBehavior():
    prewarm     = Uniform(0, 5, 10, 15, 25, 35)
    prewarm_throttle = Range(0.4, 0.8)
    brake_force = Range(0.5, 1.0)
    for i in range(prewarm):
        take SetThrottleAction(prewarm_throttle), SetBrakeAction(0), SetSteerAction(0)
    while True:
        take SetThrottleAction(0), SetBrakeAction(brake_force), SetSteerAction(0)


# --- 4 leaf scenarios (each creates its own ego with the matching behavior) ---

scenario GoStraightScenario():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for DISTANCE_TO_INTERSECTION,
                with behavior GoStraightBehavior(),
                with speed Range(0, 9)
    compose:
        while True:
            wait


scenario TurnLeftScenario():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for DISTANCE_TO_INTERSECTION,
                with behavior TurnLeftBehavior(),
                with speed Range(0, 9)
    compose:
        while True:
            wait


scenario TurnRightScenario():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for DISTANCE_TO_INTERSECTION,
                with behavior TurnRightBehavior(),
                with speed Range(0, 9)
    compose:
        while True:
            wait


scenario BrakeScenario():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for DISTANCE_TO_INTERSECTION,
                with behavior BrakeBehavior(),
                with speed Range(0, 9)
    compose:
        while True:
            wait


# --- Compositional entrypoint ---
# Five sequential `do choose` over the 4 scenarios -> 4^5 = 1024 paths.
# Parser-only — never compiled directly (Scenic's `do choose` rejects sub-
# scenarios at simulate time when there's no concrete branch).

scenario Main():
    compose:
        do choose { GoStraightScenario(): 1, TurnLeftScenario(): 1, TurnRightScenario(): 1, BrakeScenario(): 1 }
        do choose { GoStraightScenario(): 1, TurnLeftScenario(): 1, TurnRightScenario(): 1, BrakeScenario(): 1 }
        do choose { GoStraightScenario(): 1, TurnLeftScenario(): 1, TurnRightScenario(): 1, BrakeScenario(): 1 }
        do choose { GoStraightScenario(): 1, TurnLeftScenario(): 1, TurnRightScenario(): 1, BrakeScenario(): 1 }
        do choose { GoStraightScenario(): 1, TurnLeftScenario(): 1, TurnRightScenario(): 1, BrakeScenario(): 1 }


# --- Monolithic counterpart ---
# One MetaDrive sim per trace, ego runs WanderBehavior end-to-end (5 segments
# of WANDER_SEGMENT_LEN = MAX_STEPS-PREWARM_TRIM ticks each, total
# 5*(MAX_STEPS-PREWARM_TRIM) ticks — matches per-primitive useful length × 5).
# Each segment's action tuple is sampled INDEPENDENTLY at scene creation
# (matching the per-primitive independence assumption of the engine).

GO_TUPLE     = (0.4, 0, 0)
LEFT_TUPLE   = (0.4, 0, -0.3)
RIGHT_TUPLE  = (0.4, 0, 0.3)
BRAKE_TUPLE  = (0, 1.0, 0)

WANDER_SEGMENT_LEN = 40   # matches per-primitive USEFUL length (MAX_STEPS - PREWARM_TRIM)

behavior WanderBehavior():
    s1 = Uniform(GO_TUPLE, LEFT_TUPLE, RIGHT_TUPLE, BRAKE_TUPLE)
    s2 = Uniform(GO_TUPLE, LEFT_TUPLE, RIGHT_TUPLE, BRAKE_TUPLE)
    s3 = Uniform(GO_TUPLE, LEFT_TUPLE, RIGHT_TUPLE, BRAKE_TUPLE)
    s4 = Uniform(GO_TUPLE, LEFT_TUPLE, RIGHT_TUPLE, BRAKE_TUPLE)
    s5 = Uniform(GO_TUPLE, LEFT_TUPLE, RIGHT_TUPLE, BRAKE_TUPLE)
    for i in range(WANDER_SEGMENT_LEN):
        take SetThrottleAction(s1[0]), SetBrakeAction(s1[1]), SetSteerAction(s1[2])
    for i in range(WANDER_SEGMENT_LEN):
        take SetThrottleAction(s2[0]), SetBrakeAction(s2[1]), SetSteerAction(s2[2])
    for i in range(WANDER_SEGMENT_LEN):
        take SetThrottleAction(s3[0]), SetBrakeAction(s3[1]), SetSteerAction(s3[2])
    for i in range(WANDER_SEGMENT_LEN):
        take SetThrottleAction(s4[0]), SetBrakeAction(s4[1]), SetSteerAction(s4[2])
    for i in range(WANDER_SEGMENT_LEN):
        take SetThrottleAction(s5[0]), SetBrakeAction(s5[1]), SetSteerAction(s5[2])


scenario MonolithicWander():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for DISTANCE_TO_INTERSECTION,
              with behavior WanderBehavior()
    compose:
        while True:
            wait
