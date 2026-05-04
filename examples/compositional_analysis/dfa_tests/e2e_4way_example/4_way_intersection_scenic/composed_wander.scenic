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

# --- Spawn-lane discovery (mirrors composed.scenic / intersection_base.scenic) ---
fourWayIntersection = filter(lambda i: i.is4Way, network.intersections)
intersec = fourWayIntersection[0]
rightLanes = filter(
    lambda lane: all([section._laneToRight is None for section in lane.sections]),
    intersec.incomingLanes,
)
startLane = rightLanes[0]
uberSpawnPoint = startLane.centerline[-1]


# --- 4 leaf behaviors ---
# Three cruise variants (straight/left/right) + a Brake primitive.
# Each behavior begins with a cruise PREWARM of variable length: a random
# number of cruise ticks at high throttle to ramp the ego up to a varied
# starting speed BEFORE the primitive's actual actions begin. The test
# trims `PREWARM_TRIM` rows from each per-primitive CSV after generation,
# so the recorded "row 0" represents a warm state with varied speed across
# traces, not always rest. This widens the boundary distribution that the
# compositional engine's KDE re-weighting needs at segment boundaries.

behavior GoStraight():
    prewarm  = Uniform(0, 5, 10, 15, 25, 35)
    prewarm_throttle = Range(0.4, 0.8)
    throttle = Range(0.3, 0.6)
    for i in range(prewarm):
        take SetThrottleAction(prewarm_throttle), SetBrakeAction(0), SetSteerAction(0)
    while True:
        take SetThrottleAction(throttle), SetBrakeAction(0), SetSteerAction(0)


behavior TurnLeft():
    prewarm  = Uniform(0, 5, 10, 15, 25, 35)
    prewarm_throttle = Range(0.4, 0.8)
    throttle = Range(0.3, 0.6)
    steer    = Range(-0.4, -0.2)   # negative = left
    for i in range(prewarm):
        take SetThrottleAction(prewarm_throttle), SetBrakeAction(0), SetSteerAction(0)
    while True:
        take SetThrottleAction(throttle), SetBrakeAction(0), SetSteerAction(steer)


behavior TurnRight():
    prewarm  = Uniform(0, 5, 10, 15, 25, 35)
    prewarm_throttle = Range(0.4, 0.8)
    throttle = Range(0.3, 0.6)
    steer    = Range(0.2, 0.4)     # positive = right
    for i in range(prewarm):
        take SetThrottleAction(prewarm_throttle), SetBrakeAction(0), SetSteerAction(0)
    while True:
        take SetThrottleAction(throttle), SetBrakeAction(0), SetSteerAction(steer)


behavior Brake():
    # Cruise prewarm so the brake starts from a varied non-zero speed
    # (matching what continuous-drive segments would see — Brake in the
    # middle of a trace doesn't start from rest).
    prewarm     = Uniform(0, 5, 10, 15, 25, 35)
    prewarm_throttle = Range(0.4, 0.8)
    brake_force = Range(0.5, 1.0)
    for i in range(prewarm):
        take SetThrottleAction(prewarm_throttle), SetBrakeAction(0), SetSteerAction(0)
    while True:
        take SetThrottleAction(0), SetBrakeAction(brake_force), SetSteerAction(0)


# --- Monolithic-wander helpers ---
# Pre-sampled (throttle, brake, steer) tuples — one per "segment" of the
# wander. Scenic forbids branching on a sampled value at simulation time,
# so each `s_k = Uniform(GO, LEFT, RIGHT, BRAKE)` resolves to ONE concrete
# tuple at scene creation, and we iterate through those tuples in order
# inside the Wander behavior. End result: a single MetaDrive simulation
# whose ego executes 5 random per-segment action sets in sequence —
# the literal continuous-drive analogue of the do-choose composition.

GO_TUPLE     = (0.4, 0, 0)
LEFT_TUPLE   = (0.4, 0, -0.3)
RIGHT_TUPLE  = (0.4, 0, 0.3)
BRAKE_TUPLE  = (0, 1.0, 0)

WANDER_SEGMENT_LEN = 40   # ticks per segment; matches per-primitive MAX_STEPS

behavior Wander():
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
              with behavior Wander()
    compose:
        while True:
            wait


# --- Compositional entrypoint ---
# Five sequential `do choose` statements -> 5 random decisions.
# 4 options per step -> 4^5 = 1024 distinct execution paths.

scenario Main():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for DISTANCE_TO_INTERSECTION,
              with behavior GoStraight()
    compose:
        do choose { GoStraight(): 1, TurnLeft(): 1, TurnRight(): 1, Brake(): 1 }
        do choose { GoStraight(): 1, TurnLeft(): 1, TurnRight(): 1, Brake(): 1 }
        do choose { GoStraight(): 1, TurnLeft(): 1, TurnRight(): 1, Brake(): 1 }
        do choose { GoStraight(): 1, TurnLeft(): 1, TurnRight(): 1, Brake(): 1 }
        do choose { GoStraight(): 1, TurnLeft(): 1, TurnRight(): 1, Brake(): 1 }
