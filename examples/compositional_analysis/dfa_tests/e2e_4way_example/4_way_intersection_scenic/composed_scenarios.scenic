# Composition over real Scenic scenarios (not just behaviors).
#
# Each leaf primitive is a `scenario X():` block whose setup creates its own
# ego with `EgoBehavior(trajectory)` (lane-network-aware FollowTrajectoryBehavior).
# This is the "rich" composition path — the ego actually navigates the road
# graph at the intersection rather than running open-loop `take` actions.
#
# Composition (per Main):
#   Subscenario1                                      # approach intersection
#   do choose { Subscenario2L, Subscenario2R, Subscenario2S }   # turn or go straight
#
# Caveats:
#   - Each sub-scenario creates its OWN ego in setup. The wrapper builder in
#     scenic_composition_analysis.py was updated so the per-primitive wrapper
#     emits `setup: pass` (instead of a default `ego = new Car`) when the
#     scenario primitive is self-contained. Otherwise we'd have a double-ego
#     and `_trajectory_rows` would capture the wrong one.
#   - The original subscenario2*.scenic files read post-condition CSVs from
#     subscenario1 to seed initial conditions. That cross-CSV state-passing
#     is replaced here by the compositional engine's KDE re-weighting at
#     primitive boundaries — the whole point of compositional analysis.
#   - VerifaiRange replaced with plain Range so scene generation works
#     without a VerifAI sampler (this is the standard scenic.generate path,
#     not a falsifier path).

param map = localPath('../../../../../tests/scenic/scenic_tests/cases_realistic/CARLA/Town07.xodr')
param carla_map = localPath('../../../../../tests/scenic/scenic_tests/cases_realistic/CARLA/Town07.xodr')
param timestep = 0.1
param use2DMap = True
param render = 0

DISTANCE_TO_INTERSECTION       = Range(-25, -10)   # used by Subscenario1 (approach phase, far from intersection)
SUB2_DISTANCE_TO_INTERSECTION  = Range(-3, 0)      # used by Subscenario2*: spawn AT the intersection (matching where Sub1 ends) so per-primitive Sub2 trace boundary overlaps Sub1's end position
SHUFFLE_SUB2_SPAWN_DIST        = -20.0             # used by ShuffleSub2*: spawn 20m before uberSpawnPoint so ego uses FollowLaneBehavior (high gain) to reach UBER_SPEED before entering the intersection; spawning AT/near the intersection triggers TurnBehavior (tiny PID gain ~0.04) which tops out at ~0.5 m/s
UBER_SPEED = Range(2, 8)   # widened — gives per-trace variance in target speed
                            # so rho is graded rather than 0/1

model scenic.simulators.metadrive.model

# Ego vehicle just follows the trajectory specified later on.
# We DON'T `terminate` after FollowTrajectoryBehavior because if the
# trajectory is short or partially traversed, terminate fires immediately
# and the trace is only 2 frames. Instead, let MAX_STEPS in the simulator
# call cap the trace length naturally.
behavior EgoBehavior(trajectory):
    spd = UBER_SPEED
    do FollowTrajectoryBehavior(trajectory=trajectory, target_speed=spd, turn_speed=spd)
    while True:
        wait


# Used by Subscenario2L/R/S so per-primitive Sub2 traces start at varied
# non-zero speed. The test post-processing trims PREWARM_TRIM rows so each
# trace's recorded "row 0" represents a warm state (ego at varied speed
# near the intersection). This widens Sub2's start-of-segment feature
# distribution to overlap with Sub1's end-of-segment distribution, which
# is what the compositional engine's KDE re-weighting needs at boundaries.
behavior EgoBehaviorWithPrewarm(trajectory):
    prewarm_steps    = Uniform(0, 5, 10, 15, 20, 25)
    prewarm_throttle = Range(0.4, 0.8)
    for i in range(prewarm_steps):
        take SetThrottleAction(prewarm_throttle), SetBrakeAction(0), SetSteerAction(0)
    do FollowTrajectoryBehavior(trajectory=trajectory, target_speed=UBER_SPEED)
    while True:
        wait

# --- Module-level lane / maneuver discovery (same as the original
# intersection_base.scenic, used by all 4 sub-scenarios) ---
fourWayIntersection = filter(lambda i: i.is4Way, network.intersections)
intersec = fourWayIntersection[0]
rightLanes = filter(
    lambda lane: all([section._laneToRight is None for section in lane.sections]),
    intersec.incomingLanes,
)
startLane = rightLanes[0]
uberSpawnPoint = startLane.centerline[-1]

straight_maneuvers = filter(lambda i: i.type == ManeuverType.STRAIGHT, startLane.maneuvers)
straight_maneuver = Uniform(*straight_maneuvers)

left_maneuvers = filter(lambda i: i.type == ManeuverType.LEFT_TURN, startLane.maneuvers)
left_maneuver = Uniform(*left_maneuvers)

right_maneuvers = filter(lambda i: i.type == ManeuverType.RIGHT_TURN, startLane.maneuvers)
right_maneuver = Uniform(*right_maneuvers)


# --- Sub-scenario 1: approach the intersection (cruise to it on startLane) ---
scenario Subscenario1():
    setup:
        ego_trajectory = [straight_maneuver.startLane]
        ego = new Car following roadDirection from uberSpawnPoint for DISTANCE_TO_INTERSECTION,
                with behavior EgoBehavior(trajectory=ego_trajectory)
    compose:
        while True:
            wait


# --- Sub-scenario 2L: left turn at the intersection ---
scenario Subscenario2L():
    setup:
        ego_trajectory = [straight_maneuver.startLane,
                          left_maneuver.connectingLane,
                          left_maneuver.endLane]
        ego = new Car following roadDirection from uberSpawnPoint for SUB2_DISTANCE_TO_INTERSECTION,
                with behavior EgoBehaviorWithPrewarm(trajectory=ego_trajectory)
    compose:
        while True:
            wait


# --- Sub-scenario 2R: right turn at the intersection ---
scenario Subscenario2R():
    setup:
        ego_trajectory = [straight_maneuver.startLane,
                          right_maneuver.connectingLane,
                          right_maneuver.endLane]
        ego = new Car following roadDirection from uberSpawnPoint for SUB2_DISTANCE_TO_INTERSECTION,
                with behavior EgoBehaviorWithPrewarm(trajectory=ego_trajectory)
    compose:
        while True:
            wait


# --- Sub-scenario 2S: straight through the intersection ---
scenario Subscenario2S():
    setup:
        ego_trajectory = [straight_maneuver.startLane,
                          straight_maneuver.connectingLane,
                          straight_maneuver.endLane]
        ego = new Car following roadDirection from uberSpawnPoint for SUB2_DISTANCE_TO_INTERSECTION,
                with behavior EgoBehaviorWithPrewarm(trajectory=ego_trajectory)
    compose:
        while True:
            wait


# --- Compositional entrypoint ---
# Sequential approach (Subscenario1) then a random-choice turn at the
# intersection. Parser-only — never compiled directly.

scenario Main():
    compose:
        do Subscenario1()
        do choose {
            Subscenario2L(): 1,
            Subscenario2R(): 1,
            Subscenario2S(): 1,
        }


# --- Terminable sub-scenarios for ShuffleMain ---
# These are variants of the existing Subscenario* that terminate when the
# ego reaches the endpoint of its trajectory (compose block exits → Scenic
# advances to the next sub-scenario in do shuffle). The existing Subscenario*
# are left unchanged so per-primitive trace generation still works.

scenario ShuffleSub1():
    setup:
        end_pt = straight_maneuver.startLane.centerline.end
        ego = new Car following roadDirection from uberSpawnPoint for DISTANCE_TO_INTERSECTION,
                with behavior EgoBehavior(trajectory=[straight_maneuver.startLane])
    compose:
        while (distance from ego to end_pt) > 5:
            wait


scenario ShuffleSub2L():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for SHUFFLE_SUB2_SPAWN_DIST,
                with behavior EgoBehavior(trajectory=[straight_maneuver.startLane,
                                                       left_maneuver.connectingLane,
                                                       left_maneuver.endLane])
    compose:
        # Wait until ego enters the intersection, then until it exits.
        # Distance-to-point termination is wrong because after a turn the car
        # moves AWAY from connectingLane.end, so distance increases forever.
        while ego not in intersection:
            wait
        while ego in intersection:
            wait


scenario ShuffleSub2R():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for SHUFFLE_SUB2_SPAWN_DIST,
                with behavior EgoBehavior(trajectory=[straight_maneuver.startLane,
                                                       right_maneuver.connectingLane,
                                                       right_maneuver.endLane])
    compose:
        while ego not in intersection:
            wait
        while ego in intersection:
            wait


scenario ShuffleSub2S():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for SHUFFLE_SUB2_SPAWN_DIST,
                with behavior EgoBehavior(trajectory=[straight_maneuver.startLane,
                                                       straight_maneuver.connectingLane,
                                                       straight_maneuver.endLane])
    compose:
        while ego not in intersection:
            wait
        while ego in intersection:
            wait


# ShuffleMain: approach the intersection, then run ALL THREE maneuvers in a
# random ORDER (do shuffle = random permutation, not random pick).
# Engine averages rho over all 3! = 6 permutations.
#
# Three fixes vs the broken version:
#   1. Dummy Car in setup → MetaDrive accepts scene (needs >=1 object at creation)
#   2. ShuffleSub* terminate when ego reaches trajectory end → do shuffle advances
#   3. _trajectory_rows uses frame[-1] → tracks active ego, not dummy
scenario ShuffleMain():
    setup:
        # Dummy car placed far from the intersection to satisfy MetaDrive's
        # >=1 Scenic object at scene-creation requirement.
        # ShuffleSub* create their own egos at sim time; the dummy stays put.
        _ = new Car at uberSpawnPoint, with allowCollisions True
    compose:
        do ShuffleSub1()
        do shuffle {
            ShuffleSub2L(): 1,
            ShuffleSub2R(): 1,
            ShuffleSub2S(): 1,
        }


# --- Monolithic entrypoint ---
# Real continuous-drive analogue of `Main`: one MetaDrive simulation per
# trace, ego runs Subscenario1's approach trajectory immediately followed
# by a Uniform-sampled turn trajectory (Sub2L / Sub2R / Sub2S). The turn is
# picked at scene creation (Uniform on lists, which Scenic accepts) and the
# behavior chains the two FollowTrajectoryBehavior calls back-to-back, so
# segment 2 starts from segment 1's actual end-state speed/position rather
# than from rest. This is the ground truth that compositional rho should
# approximate.

approach_traj = [straight_maneuver.startLane]
left_full     = [straight_maneuver.startLane,
                 left_maneuver.connectingLane,
                 left_maneuver.endLane]
right_full    = [straight_maneuver.startLane,
                 right_maneuver.connectingLane,
                 right_maneuver.endLane]
straight_full = [straight_maneuver.startLane,
                 straight_maneuver.connectingLane,
                 straight_maneuver.endLane]

behavior MonolithicEgoBehavior(approach, full_path, speed1, speed2):
    # Two INDEPENDENT per-segment target speeds (`speed1`, `speed2`) — must
    # match the per-primitive setup where each Sub2 scene freshly samples
    # UBER_SPEED rather than inheriting Sub1's. Without this, monolithic
    # measures P(scene-level UBER OK) ≈ P(stay below per segment), while
    # compositional measures P(Sub1 OK) × P(Sub2 OK) which is the square,
    # giving a structural gap (~0.24 in our setup) that the engine cannot
    # close. Two-sample monolithic lines up with compositional independence.
    do FollowTrajectoryBehavior(trajectory=approach, target_speed=speed1)
    do FollowTrajectoryBehavior(trajectory=full_path, target_speed=speed2)
    while True:
        wait


scenario MonolithicMain():
    setup:
        chosen_full = Uniform(left_full, right_full, straight_full)
        speed1 = Range(2, 8)   # ≡ UBER_SPEED for Sub1's per-primitive scene
        speed2 = Range(2, 8)   # ≡ UBER_SPEED for Sub2's per-primitive scene
        ego = new Car following roadDirection from uberSpawnPoint for DISTANCE_TO_INTERSECTION,
                with behavior MonolithicEgoBehavior(approach_traj, chosen_full, speed1, speed2)
    compose:
        while True:
            wait


# --- Monolithic shuffle counterpart ---
# Chains Sub1 + ALL THREE Sub2 variants sequentially in one simulation.
# The permutation order is sampled uniformly over all 3! = 6 orderings,
# mirroring the compositional shuffle operator which averages rho over all
# permutations. Each segment gets an independent target speed to match the
# per-primitive independence assumption (same rationale as MonolithicMain).
# Segments 3-4 follow trajectories that begin at the intersection approach
# point even though the ego is physically elsewhere after segment 2; the
# spec only observes speed, so position inconsistency is acceptable here.

behavior MonolithicShuffleBehavior(approach, perm, s1, s2, s3, s4):
    do FollowTrajectoryBehavior(trajectory=approach, target_speed=s1)
    do FollowTrajectoryBehavior(trajectory=perm[0], target_speed=s2)
    do FollowTrajectoryBehavior(trajectory=perm[1], target_speed=s3)
    do FollowTrajectoryBehavior(trajectory=perm[2], target_speed=s4)
    while True:
        wait


scenario MonolithicShuffle():
    setup:
        perm = Uniform(
            (left_full, right_full, straight_full),
            (left_full, straight_full, right_full),
            (right_full, left_full, straight_full),
            (right_full, straight_full, left_full),
            (straight_full, left_full, right_full),
            (straight_full, right_full, left_full),
        )
        s1 = Range(2, 8)
        s2 = Range(2, 8)
        s3 = Range(2, 8)
        s4 = Range(2, 8)
        ego = new Car following roadDirection from uberSpawnPoint for DISTANCE_TO_INTERSECTION,
                with behavior MonolithicShuffleBehavior(approach_traj, perm, s1, s2, s3, s4)
    compose:
        while True:
            wait


# --- MonolithicShuffleV2: physically correct 3-maneuver shuffle ---
# Fixes the broken MonolithicShuffle by using BFS-found return paths (same as
# Monolithic5 in traversal_wander.scenic) to drive the car physically back to
# the start of startLane after each maneuver, so perm[i+1]'s trajectory starts
# where the car actually is. No trajectory mismatch.
#
# Return path lane IDs (identical to traversal_wander.scenic's Monolithic5):
#   after left  (7 lanes, ~215 m): ends at start of startLane
#   after right (7 lanes, ~293 m): ends at start of startLane
#   after straight (13 lanes, ~381 m): ends at start of startLane
#
# Each traversal uses an independent target speed (s1..s4). The last segment
# omits the return path (not needed). Each perm tuple is (trav_with_ret,
# trav_with_ret, turn_only) so the 3rd segment ends on endLane without an
# unnecessary return drive.

behavior MonolithicShuffleV2Behavior(approach, perm, s1, s2, s3, s4):
    do FollowTrajectoryBehavior(trajectory=approach, target_speed=s1)
    do FollowTrajectoryBehavior(trajectory=perm[0], target_speed=s2)
    do FollowTrajectoryBehavior(trajectory=perm[1], target_speed=s3)
    do FollowTrajectoryBehavior(trajectory=perm[2], target_speed=s4)
    while True:
        wait


# --- Far-spawn Sub2 variants for steer gap analysis ---
# Identical trajectory to Subscenario2*, but spawn at SHUFFLE_SUB2_SPAWN_DIST
# (-20m) using EgoBehavior (no prewarm throttle jitter). With 25-step prewarm
# trimming the approach, the intersection turn falls at rows ~15-45 of the CSV;
# index-based warmup (masking rows 0-24) then partially exposes the turn for
# evaluation, matching how ShuffleMain evaluates Sub2 segments.

scenario Subscenario2L_far():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for SHUFFLE_SUB2_SPAWN_DIST,
                with behavior EgoBehavior(trajectory=[straight_maneuver.startLane,
                                                       left_maneuver.connectingLane,
                                                       left_maneuver.endLane])
    compose:
        while True:
            wait


scenario Subscenario2R_far():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for SHUFFLE_SUB2_SPAWN_DIST,
                with behavior EgoBehavior(trajectory=[straight_maneuver.startLane,
                                                       right_maneuver.connectingLane,
                                                       right_maneuver.endLane])
    compose:
        while True:
            wait


scenario Subscenario2S_far():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for SHUFFLE_SUB2_SPAWN_DIST,
                with behavior EgoBehavior(trajectory=[straight_maneuver.startLane,
                                                       straight_maneuver.connectingLane,
                                                       straight_maneuver.endLane])
    compose:
        while True:
            wait


scenario MonolithicShuffleV2():
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

        # trav_X = turn trajectory + return path to start of startLane
        trav_left     = left_full     + ret_left
        trav_right    = right_full    + ret_right
        trav_straight = straight_full + ret_straight

        # Each permutation tuple: (trav_with_ret, trav_with_ret, turn_only)
        # The third element has no return path — car stays on endLane at finish.
        perm = Uniform(
            (trav_left,     trav_right,    straight_full),
            (trav_left,     trav_straight, right_full),
            (trav_right,    trav_left,     straight_full),
            (trav_right,    trav_straight, left_full),
            (trav_straight, trav_left,     right_full),
            (trav_straight, trav_right,    left_full),
        )

        s1 = Range(2, 8)
        s2 = Range(2, 8)
        s3 = Range(2, 8)
        s4 = Range(2, 8)
        ego = new Car following roadDirection from uberSpawnPoint for DISTANCE_TO_INTERSECTION,
                with behavior MonolithicShuffleV2Behavior(approach_traj, perm, s1, s2, s3, s4)
    compose:
        while True:
            wait
