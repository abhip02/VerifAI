# MetaDrive primitives — S, X, C, O on Town07.
# Spawn-lane discovery is byte-identical to wander_scenarios.scenic; each
# leaf scenario spawns its own ego and runs a closed-loop PID speed
# controller at target_speed picked so final speeds map cleanly to the
# S < 3.5 < C < 7 <= X,O DFA labelling alphabet.
#
# We bypass scenic.domains.driving's RegulatedControlAction (which clamps
# throttle to 0.5 and caps the achievable speed below 7 m/s on Town07's
# approach lane) by emitting SetThrottleAction/SetBrakeAction directly
# from a PID loop. The PID gains come from the metadrive simulator's
# getLaneFollowingControllers binding, so this is closed-loop control —
# only the actuation clamp differs from upstream FollowLaneBehavior.

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

# --- Spawn-lane discovery (verbatim from wander_scenarios.scenic) ---
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


# Closed-loop speed controller: PID on (target - current_speed), emitting
# raw SetThrottleAction / SetBrakeAction. Lateral steer kept at 0; ego runs
# straight through Town07's incoming lane and into the straight maneuver.
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
