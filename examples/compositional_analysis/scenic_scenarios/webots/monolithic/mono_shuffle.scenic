param timestep = 0.1

from scenic.simulators.webots.road.world import setLocalWorld
setLocalWorld(__file__, '../world/simple.wbt')
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
              with behavior HoldSpeedBehavior(9.0)
        terminate after TICKS steps
    compose:
        while True:
            wait

scenario C():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for DIST,
              with speed Range(0, 10),
              with behavior HoldSpeedBehavior(5.0)
        terminate after TICKS steps
    compose:
        while True:
            wait

scenario O():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for DIST,
              with speed Range(0, 10),
              with behavior HoldSpeedBehavior(8.0)
        terminate after TICKS steps
    compose:
        while True:
            wait

scenario Main():
    compose:
        do S()
        do shuffle {
            C(): 1,
            X(): 1,
            O(): 1,
        }

# Pick one of the 6 speed permutations (C=5, X=9, O=8) at scene creation.
# MonoBehavior chains the three speed segments on one ego in that order.
perm_speeds = Uniform((5.0, 9.0, 8.0), (5.0, 8.0, 9.0), (9.0, 5.0, 8.0),
                     (9.0, 8.0, 5.0), (8.0, 5.0, 9.0), (8.0, 9.0, 5.0))

behavior MonoSShuffleCXOBehavior(speed_a, speed_b, speed_c):
    do SlowToStopAndHold()
    do HoldSpeedBehavior(speed_a) for TICKS steps
    do HoldSpeedBehavior(speed_b) for TICKS steps
    do HoldSpeedBehavior(speed_c) for TICKS steps

scenario MonoSShuffleCXO():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for DIST,
              with behavior MonoSShuffleCXOBehavior(perm_speeds[0], perm_speeds[1], perm_speeds[2])
        terminate after 160 steps
    compose:
        while True:
            wait
