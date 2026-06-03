# Webots primitives — S, X, C, O on simple.wbt.
# Same scenario/behavior structure as the MetaDrive sibling; only the
# `model` line and spawn-lane discovery differ. PID closed-loop speed
# control bypasses RegulatedControlAction's 0.5-throttle clamp so the
# X / O primitives can clear the FAST_THRESHOLD=7.0 m/s labelling boundary.

param timestep = 0.1

from scenic.simulators.webots.road.world import setLocalWorld
setLocalWorld(__file__, 'world/simple.wbt')
from scenic.simulators.webots.road.model import *

from scenic.domains.driving.actions import (
    SetThrottleAction,
    SetBrakeAction,
    SetSteerAction,
)

startLane = network.lanes[0]
uberSpawnPoint = startLane.centerline[5]

DIST  = Range(0, 4)
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


behavior SlowBehavior():
    lon_controller, _ = simulation().getLaneFollowingControllers(self)
    while self.speed is None or self.speed > 0.4:
        current = self.speed if self.speed is not None else 0
        u = lon_controller.run_step(1.5 - current)
        if u >= 0:
            take SetThrottleAction(min(u, 1.0)), SetBrakeAction(0), SetSteerAction(0)
        else:
            take SetThrottleAction(0), SetBrakeAction(min(-u, 1.0)), SetSteerAction(0)
    take SetThrottleAction(0), SetBrakeAction(1.0), SetSteerAction(0)
    while True:
        wait


behavior FastBehavior():
    do HoldSpeedBehavior(Range(7.0, 10.0))


behavior CruiseBehavior():
    do HoldSpeedBehavior(Range(2.5, 7.5))


behavior OvertakeBehavior():
    do HoldSpeedBehavior(Range(6.5, 9.0))


scenario S():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for DIST,
              with speed Range(0, 10),
              with behavior SlowBehavior()
        terminate after TICKS steps
    compose:
        while True:
            wait


scenario X():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for DIST,
              with speed Range(0, 10),
              with behavior FastBehavior()
        terminate after TICKS steps
    compose:
        while True:
            wait


scenario C():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for DIST,
              with speed Range(0, 10),
              with behavior CruiseBehavior()
        terminate after TICKS steps
    compose:
        while True:
            wait


scenario O():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for DIST,
              with speed Range(0, 10),
              with behavior OvertakeBehavior()
        terminate after TICKS steps
    compose:
        while True:
            wait
