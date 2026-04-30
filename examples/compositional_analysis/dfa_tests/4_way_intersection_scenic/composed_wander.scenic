# "Wander" composition: N random turn decisions in a row.
#
#   compose:
#     do choose { GoStraight, TurnLeft, TurnRight }     <- step 1
#     do choose { GoStraight, TurnLeft, TurnRight }     <- step 2
#     do choose { GoStraight, TurnLeft, TurnRight }     <- step 3
#     do choose { GoStraight, TurnLeft, TurnRight }     <- step 4
#     do choose { GoStraight, TurnLeft, TurnRight }     <- step 5
#
# Each `do choose` is sampled INDEPENDENTLY at parser-time, so the partner
# format yields a 5-step path with 3^5 = 243 distinct execution paths.
# Compositional analysis handles them all via importance-sampling without
# enumerating. To wander longer, append more `do choose` lines (or shorter,
# remove some). The trace pool is still 3 leaf-primitive CSVs no matter how
# many steps the composition has.
#
# (Avoid `for i in range(N): do choose {...}` — Scenic's parser collapses it
# into a single parallel-style step, not 5 sequential decision points.)

param map = localPath('../../../../tests/scenic/scenic_tests/cases_realistic/CARLA/Town07.xodr')
param carla_map = localPath('../../../../tests/scenic/scenic_tests/cases_realistic/CARLA/Town07.xodr')
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

# --- Spawn-lane discovery (mirrors composed.scenic / intersection_base.scenic) ---
fourWayIntersection = filter(lambda i: i.is4Way, network.intersections)
intersec = fourWayIntersection[0]
rightLanes = filter(
    lambda lane: all([section._laneToRight is None for section in lane.sections]),
    intersec.incomingLanes,
)
startLane = rightLanes[0]
uberSpawnPoint = startLane.centerline[-1]


# --- 3 leaf behaviors with per-scene throttle/steer randomness ---

behavior GoStraight():
    throttle = Range(0.3, 0.6)
    while True:
        take SetThrottleAction(throttle), SetBrakeAction(0), SetSteerAction(0)


behavior TurnLeft():
    throttle = Range(0.3, 0.6)
    steer    = Range(-0.4, -0.2)   # negative = left
    while True:
        take SetThrottleAction(throttle), SetBrakeAction(0), SetSteerAction(steer)


behavior TurnRight():
    throttle = Range(0.3, 0.6)
    steer    = Range(0.2, 0.4)     # positive = right
    while True:
        take SetThrottleAction(throttle), SetBrakeAction(0), SetSteerAction(steer)


# --- Compositional entrypoint ---
# Five sequential `do choose` statements -> 5 random turn decisions.
# Add or remove lines to change the wander length.

scenario Main():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for DISTANCE_TO_INTERSECTION,
              with behavior GoStraight()
    compose:
        do choose { GoStraight(): 1, TurnLeft(): 1, TurnRight(): 1 }
        do choose { GoStraight(): 1, TurnLeft(): 1, TurnRight(): 1 }
        do choose { GoStraight(): 1, TurnLeft(): 1, TurnRight(): 1 }
        do choose { GoStraight(): 1, TurnLeft(): 1, TurnRight(): 1 }
        do choose { GoStraight(): 1, TurnLeft(): 1, TurnRight(): 1 }
