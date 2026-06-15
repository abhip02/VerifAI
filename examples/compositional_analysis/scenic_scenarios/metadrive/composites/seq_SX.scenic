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

fourWayIntersection = filter(lambda i: i.is4Way, network.intersections)
intersec = fourWayIntersection[0]
rightLanes = filter(
    lambda lane: all([section._laneToRight is None for section in lane.sections]),
    intersec.incomingLanes,
)
startLane = rightLanes[0]
uberSpawnPoint = startLane.centerline[-1]

DIST  = Range(-25, -15)
TICKS = 40

behavior HoldSpeedBehavior(target_speed):
    lon_controller, _ = simulation().getLaneFollowingControllers(self)
    while True:
        current = self.speed if self.speed is not None else 0
        u = lon_controller.run_step(target_speed - current)
        if u >= 0:
            take SetThrottleAction(min(u, 1.0)), SetBrakeAction(0), SetSteerAction(0)
        else:
            take SetThrottleAction(0), SetBrakeAction(min(-u, 1.0)), SetSteerAction(0)


behavior SlowToStopAndHold():
    lon_controller, _ = simulation().getLaneFollowingControllers(self)
    while self.speed is None or self.speed > 0.4:
        current = self.speed if self.speed is not None else 0
        u = lon_controller.run_step(1.5 - current)
        if u >= 0:
            take SetThrottleAction(min(u, 1.0)), SetBrakeAction(0), SetSteerAction(0)
        else:
            take SetThrottleAction(0), SetBrakeAction(min(-u, 1.0)), SetSteerAction(0)
    for i in range(5):
        take SetThrottleAction(0), SetBrakeAction(1.0), SetSteerAction(0)


behavior SlowToStopAndStay():
    do SlowToStopAndHold()
    while True:
        take SetThrottleAction(0), SetBrakeAction(1.0), SetSteerAction(0)


# Per-primitive leaf scenarios — each spawns its own ego and runs one
# closed-loop segment for TICKS steps. Mirror the bodies in
# ../primitives.scenic exactly.
scenario S():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for DIST,
              with speed Range(0, 10),
              with behavior SlowToStopAndStay()
        terminate after TICKS steps
    compose:
        while True:
            wait

scenario X():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for DIST,
              with speed Range(0, 10),
              with behavior HoldSpeedBehavior(Range(7.0, 10.0))
        terminate after TICKS steps
    compose:
        while True:
            wait

scenario C():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for DIST,
              with speed Range(0, 10),
              with behavior HoldSpeedBehavior(Range(2.5, 7.5))
        terminate after TICKS steps
    compose:
        while True:
            wait

scenario O():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for DIST,
              with speed Range(0, 10),
              with behavior HoldSpeedBehavior(Range(6.5, 9.0))
        terminate after TICKS steps
    compose:
        while True:
            wait

scenario Main():
    compose:
        do S()
        do X()

behavior MonoSXBehavior():
    do SlowToStopAndHold()
    do HoldSpeedBehavior(Range(7.0, 10.0)) for TICKS steps

scenario MonoSX():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for DIST,
              with behavior MonoSXBehavior()
        terminate after 80 steps
    compose:
        while True:
            wait
