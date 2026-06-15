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
param real_time = 0

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
        take SetBrakeAction(1.0)


# Gentle prewarm for TurnX primitives: accelerate to a speed drawn from the
# same Range(2, 8) as the approach target_speed, then follow the turn.
# This matches the initial speed distribution the car has when entering a turn
# in the monolithic scenario (where it arrives from the preceding approach).
behavior EgoTurnBehavior(trajectory):
    prewarm_speed = Range(2, 5)
    while self.speed < prewarm_speed:
        take SetThrottleAction(0.3), SetBrakeAction(0), SetSteerAction(0)
    do FollowTrajectoryBehavior(trajectory=trajectory, target_speed=Range(2, 8))
    while True:
        take SetBrakeAction(1.0)


# --- 4 leaf scenarios ---

scenario ApproachScenario():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for DISTANCE_TO_INTERSECTION,
                with behavior EgoBehavior(approach_only, Range(2, 8))
    compose:
        while True:
            wait


scenario TurnLeftScenario():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for SUB_TURN_DIST,
                with behavior EgoTurnBehavior(left_full)
    compose:
        while True:
            wait


scenario TurnRightScenario():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for SUB_TURN_DIST,
                with behavior EgoTurnBehavior(right_full)
    compose:
        while True:
            wait


scenario GoStraightScenario():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for SUB_TURN_DIST,
                with behavior EgoTurnBehavior(straight_full)
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


# --- Monolithic5: 5 chained traversals in one continuous simulation ---
#
# Each traversal = full closed loop: startLane → intersection → turn → exit →
# return path back to start of startLane.  BFS over Town07 road network found
# exact lane sequences for each turn type (left/right/straight).
#
# Return path lane IDs (verified by BFS with connecting lanes):
#   after left  (road0_lane0 exit)  : 7 lanes, ~215m
#   after right (road55_lane0 exit) : 7 lanes, ~293m
#   after straight (road44_lane1 exit): 13 lanes, ~381m
#
# After each traversal the ego arrives at end of road562_lane0 / road544_lane0,
# which is exactly the start of startLane (road45_lane1).

behavior Monolithic5Behavior(t1, t2, t3, t4, t5, s1, s2, s3, s4, s5):
    do FollowTrajectoryBehavior(trajectory=t1, target_speed=s1)
    do FollowTrajectoryBehavior(trajectory=t2, target_speed=s2)
    do FollowTrajectoryBehavior(trajectory=t3, target_speed=s3)
    do FollowTrajectoryBehavior(trajectory=t4, target_speed=s4)
    do FollowTrajectoryBehavior(trajectory=t5, target_speed=s5)
    while True:
        wait


scenario Monolithic5():
    setup:
        lane_lut = {l.id: l for l in network.lanes}

        ret_left = [lane_lut[i] for i in [
            'road128_lane0', 'road1_lane0', 'road153_lane0',
            'road60_lane0', 'road464_lane0', 'road61_lane0', 'road562_lane0',
        ]]
        ret_right = [lane_lut[i] for i in [
            'road280_lane0', 'road56_lane0', 'road38_lane1',
            'road23_lane0', 'road336_lane0', 'road62_lane1', 'road544_lane0',
        ]]
        ret_straight = [lane_lut[i] for i in [
            'road622_lane0', 'road43_lane1', 'road917_lane0', 'road47_lane1',
            'road218_lane0', 'road37_lane0', 'road145_lane0', 'road1_lane0',
            'road153_lane0', 'road60_lane0', 'road464_lane0', 'road61_lane0',
            'road562_lane0',
        ]]

        trav_left     = left_full     + ret_left
        trav_right    = right_full    + ret_right
        trav_straight = straight_full + ret_straight

        t1 = Uniform(trav_left, trav_right, trav_straight)
        t2 = Uniform(trav_left, trav_right, trav_straight)
        t3 = Uniform(trav_left, trav_right, trav_straight)
        t4 = Uniform(trav_left, trav_right, trav_straight)
        t5 = Uniform(trav_left, trav_right, trav_straight)
        s1 = Range(2, 8)
        s2 = Range(2, 8)
        s3 = Range(2, 8)
        s4 = Range(2, 8)
        s5 = Range(2, 8)
        ego = new Car following roadDirection from uberSpawnPoint for DISTANCE_TO_INTERSECTION,
                with behavior Monolithic5Behavior(t1, t2, t3, t4, t5, s1, s2, s3, s4, s5)
    compose:
        while True:
            wait
