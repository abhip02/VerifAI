# VerifAI-driven primitive: GoStraight.
# `param DISTANCE_TO_INTERSECTION = VerifaiRange(...)` is sampled by VerifAI's
# external sampler when this file is loaded via verifai.samplers.ScenicSampler
# and driven by generic_falsifier + ScenicServer. record directives expose the
# ego trajectory to the test's spec_monitor for trace-CSV construction.

param map = localPath('../../../../tests/scenic/scenic_tests/cases_realistic/CARLA/Town07.xodr')
param carla_map = localPath('../../../../tests/scenic/scenic_tests/cases_realistic/CARLA/Town07.xodr')
param timestep = 0.1
param use2DMap = True
param render = 0

param DISTANCE_TO_INTERSECTION = VerifaiRange(-15, -5)

model scenic.simulators.metadrive.model

from scenic.domains.driving.actions import (
    SetThrottleAction,
    SetBrakeAction,
    SetSteerAction,
)

CRUISE_THROTTLE = 0.4

fourWayIntersection = filter(lambda i: i.is4Way, network.intersections)
intersec = fourWayIntersection[0]
rightLanes = filter(
    lambda lane: all([section._laneToRight is None for section in lane.sections]),
    intersec.incomingLanes,
)
startLane = rightLanes[0]
uberSpawnPoint = startLane.centerline[-1]


behavior GoStraight():
    n_brake = Uniform(1, 2, 3, 4, 5)
    for i in range(3):
        take SetThrottleAction(CRUISE_THROTTLE), SetBrakeAction(0), SetSteerAction(0)
    for i in range(n_brake):
        take SetThrottleAction(0), SetBrakeAction(1.0), SetSteerAction(0)
    while True:
        take SetThrottleAction(CRUISE_THROTTLE), SetBrakeAction(0), SetSteerAction(0)


ego = new Car following roadDirection from uberSpawnPoint for globalParameters.DISTANCE_TO_INTERSECTION,
        with behavior GoStraight()

record ego.speed as ego_speed
record ego.position as ego_position
record ego.heading as ego_heading
