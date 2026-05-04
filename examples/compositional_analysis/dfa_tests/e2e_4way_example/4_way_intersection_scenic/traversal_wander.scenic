# Traversal-wander composition (split-primitive redesign).
#
# Primitives:
#   ApproachScenario   — ego spawns far from intersection, drives up to it
#   TurnLeftScenario   — ego spawns near intersection (prewarm), turns left
#   TurnRightScenario  — ego spawns near intersection (prewarm), turns right
#   GoStraightScenario — ego spawns near intersection (prewarm), goes straight
#
# Compositional Main: 10 steps = 5× (Approach ; do choose{L/R/S})
#   3^5 = 243 distinct execution paths; 4 trace pools sampled independently.
#
# Monolithic5: ONE simulation, ego chains 5 full traversals back-to-back.
#   Each traversal = FollowTrajectoryBehavior(approach_only, sa_i) then
#   FollowTrajectoryBehavior(turn_traj, st_i) with INDEPENDENT speeds sa_i, st_i.
#   After each turn the ego is on an exit lane; FollowTrajectoryBehavior for the
#   next approach steers toward startLane from wherever it is — physically
#   approximate but the spec observes speed only.

param map = localPath('../../../../../tests/scenic/scenic_tests/cases_realistic/CARLA/Town07.xodr')
param carla_map = localPath('../../../../../tests/scenic/scenic_tests/cases_realistic/CARLA/Town07.xodr')
param timestep = 0.1
param use2DMap = True
param render = 0

model scenic.simulators.metadrive.model

DISTANCE_TO_INTERSECTION = Range(-25, -10)   # ApproachScenario: spawn far
SUB_TURN_DIST            = Range(-3, 0)      # TurnX: spawn at intersection entry

# --- Road network setup ---
fourWayIntersection = filter(lambda i: i.is4Way, network.intersections)
intersec = fourWayIntersection[0]
rightLanes = filter(
    lambda lane: all([section._laneToRight is None for section in lane.sections]),
    intersec.incomingLanes,
)
startLane      = rightLanes[0]
uberSpawnPoint = startLane.centerline[-1]

straight_maneuvers = filter(lambda i: i.type == ManeuverType.STRAIGHT, startLane.maneuvers)
straight_maneuver  = Uniform(*straight_maneuvers)
left_maneuvers     = filter(lambda i: i.type == ManeuverType.LEFT_TURN, startLane.maneuvers)
left_maneuver      = Uniform(*left_maneuvers)
right_maneuvers    = filter(lambda i: i.type == ManeuverType.RIGHT_TURN, startLane.maneuvers)
right_maneuver     = Uniform(*right_maneuvers)

# Approach-only trajectory (just the incoming lane up to the intersection box)
approach_only = [straight_maneuver.startLane]

# Full turn trajectories: approach lane → connecting lane → exit lane
left_full     = [straight_maneuver.startLane,
                 left_maneuver.connectingLane,
                 left_maneuver.endLane]
right_full    = [straight_maneuver.startLane,
                 right_maneuver.connectingLane,
                 right_maneuver.endLane]
straight_full = [straight_maneuver.startLane,
                 straight_maneuver.connectingLane,
                 straight_maneuver.endLane]


# --- Shared behaviors ---

behavior EgoBehavior(trajectory, speed):
    do FollowTrajectoryBehavior(trajectory=trajectory, target_speed=speed)
    while True:
        wait


# Prewarm: short random throttle burst before the main trajectory so
# TurnX primitives start with varied non-zero speed (needed for KDE bridge).
behavior EgoBehaviorWithPrewarm(trajectory):
    prewarm_steps    = Uniform(0, 5, 10, 15, 20, 25)
    prewarm_throttle = Range(0.4, 0.8)
    for i in range(prewarm_steps):
        take SetThrottleAction(prewarm_throttle), SetBrakeAction(0), SetSteerAction(0)
    do FollowTrajectoryBehavior(trajectory=trajectory, target_speed=Range(2, 8))
    while True:
        wait


# --- 4 leaf scenarios ---

scenario ApproachScenario():
    setup:
        speed = Range(2, 8)
        ego = new Car following roadDirection from uberSpawnPoint for DISTANCE_TO_INTERSECTION,
                with behavior EgoBehavior(approach_only, speed)
    compose:
        while True:
            wait


scenario TurnLeftScenario():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for SUB_TURN_DIST,
                with behavior EgoBehaviorWithPrewarm(left_full)
    compose:
        while True:
            wait


scenario TurnRightScenario():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for SUB_TURN_DIST,
                with behavior EgoBehaviorWithPrewarm(right_full)
    compose:
        while True:
            wait


scenario GoStraightScenario():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for SUB_TURN_DIST,
                with behavior EgoBehaviorWithPrewarm(straight_full)
    compose:
        while True:
            wait


# --- Compositional entrypoint ---
# 10-step composition: 5 repetitions of (Approach → random turn).
# Parser-only — never compiled directly.

scenario Main():
    compose:
        do ApproachScenario()
        do choose { TurnLeftScenario(): 1, TurnRightScenario(): 1, GoStraightScenario(): 1 }
        do ApproachScenario()
        do choose { TurnLeftScenario(): 1, TurnRightScenario(): 1, GoStraightScenario(): 1 }
        do ApproachScenario()
        do choose { TurnLeftScenario(): 1, TurnRightScenario(): 1, GoStraightScenario(): 1 }
        do ApproachScenario()
        do choose { TurnLeftScenario(): 1, TurnRightScenario(): 1, GoStraightScenario(): 1 }
        do ApproachScenario()
        do choose { TurnLeftScenario(): 1, TurnRightScenario(): 1, GoStraightScenario(): 1 }


# --- Monolithic5: 5 chained traversals in one simulation ---
# sa1..sa5: independent approach speeds; st1..st5: independent turn speeds.
# After each turn the ego is on the exit lane; FollowTrajectoryBehavior for
# the next approach navigates back toward startLane (position-approximate, OK).

behavior Monolithic5Behavior(
        turn1, turn2, turn3, turn4, turn5,
        sa1, sa2, sa3, sa4, sa5,
        st1, st2, st3, st4, st5):
    do FollowTrajectoryBehavior(trajectory=approach_only, target_speed=sa1)
    do FollowTrajectoryBehavior(trajectory=turn1, target_speed=st1)
    do FollowTrajectoryBehavior(trajectory=approach_only, target_speed=sa2)
    do FollowTrajectoryBehavior(trajectory=turn2, target_speed=st2)
    do FollowTrajectoryBehavior(trajectory=approach_only, target_speed=sa3)
    do FollowTrajectoryBehavior(trajectory=turn3, target_speed=st3)
    do FollowTrajectoryBehavior(trajectory=approach_only, target_speed=sa4)
    do FollowTrajectoryBehavior(trajectory=turn4, target_speed=st4)
    do FollowTrajectoryBehavior(trajectory=approach_only, target_speed=sa5)
    do FollowTrajectoryBehavior(trajectory=turn5, target_speed=st5)
    while True:
        wait


scenario Monolithic5():
    setup:
        turn1 = Uniform(left_full, right_full, straight_full)
        turn2 = Uniform(left_full, right_full, straight_full)
        turn3 = Uniform(left_full, right_full, straight_full)
        turn4 = Uniform(left_full, right_full, straight_full)
        turn5 = Uniform(left_full, right_full, straight_full)
        sa1 = Range(2, 8)
        sa2 = Range(2, 8)
        sa3 = Range(2, 8)
        sa4 = Range(2, 8)
        sa5 = Range(2, 8)
        st1 = Range(2, 8)
        st2 = Range(2, 8)
        st3 = Range(2, 8)
        st4 = Range(2, 8)
        st5 = Range(2, 8)
        ego = new Car following roadDirection from uberSpawnPoint for DISTANCE_TO_INTERSECTION,
                with behavior Monolithic5Behavior(
                    turn1, turn2, turn3, turn4, turn5,
                    sa1, sa2, sa3, sa4, sa5,
                    st1, st2, st3, st4, st5)
    compose:
        while True:
            wait
