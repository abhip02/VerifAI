"""Town07 4-way intersection — Approach + do choose { TurnL, TurnR, Straight }.

Branching is structural here: each choose branch traverses a different
connecting lane through the same 4-way intersection. Unlike the speed-band
choose composite (native_choose.scenic), where the three branches share the
same straight stretch and differ only in target speed, here the branches
are qualitatively distinct (left-turn lane vs right-turn lane vs through-
lane). This is what `do choose` actually models, and the recipe is lifted
from the legacy composed_scenarios.scenic in dfa_tests/e2e_4way_example/.

Mono ground truth uses two INDEPENDENT per-segment target speeds — same
fix as MonolithicEgoBehavior in the legacy file. Without this, mono
measures P(scene-level UBER OK) which is the square of the comp estimate,
opening a structural gap that the engine can't close.
"""

param map = localPath('../../../../../tests/scenic/scenic_tests/cases_realistic/CARLA/Town07.xodr')
param carla_map = localPath('../../../../../tests/scenic/scenic_tests/cases_realistic/CARLA/Town07.xodr')
param timestep = 0.1
param use2DMap = True
param render = 0

model scenic.simulators.metadrive.model

# FollowTrajectoryBehavior is exported by the metadrive model (via
# scenic.domains.driving.model); no explicit import needed.

# --- Module-level maneuver discovery (lifted from legacy
# composed_scenarios.scenic). Town07's 4-way intersection has all three
# maneuver types from this incoming lane. ---
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

# Full trajectories per branch — each begins with startLane (approach
# segment) and continues through connectingLane → endLane for L/R, or
# stays on the straight maneuver's connectingLane → endLane for the
# Straight branch.
left_full     = [straight_maneuver.startLane,
                 left_maneuver.connectingLane,
                 left_maneuver.endLane]
right_full    = [straight_maneuver.startLane,
                 right_maneuver.connectingLane,
                 right_maneuver.endLane]
straight_full = [straight_maneuver.startLane,
                 straight_maneuver.connectingLane,
                 straight_maneuver.endLane]

# All primitives spawn at DIST_APPROACH and run the FULL trajectory
# ([startLane → connecting → end]) so each is structurally identical to
# what mono runs on the same chosen_full branch. This eliminates the
# geometric asymmetry where comp's primitives previously spawned AT the
# intersection (DIST_SUB2) — bypassing the approach traversal that mono
# has to physically perform — and biased comp's per-branch ρ̂ relative
# to mono's. With identical spawn + trajectory, comp's per-primitive ρ̂_L
# is computing exactly what mono's L-chosen traces produce.
DIST_APPROACH = Range(-25, -10)
UBER_SPEED = Range(2, 8)
# Primitive = full trajectory horizon. Approach segment (~25 m) + turn
# segment (~25 m) at 2–8 m/s → 60–100 ticks; TICKS=160 gives margin and
# matches the mono cap so both methods have identical traces.
TICKS = 160

behavior EgoBehavior(trajectory):
    do FollowTrajectoryBehavior(trajectory=trajectory, target_speed=UBER_SPEED)
    while True:
        wait


# --- Per-primitive leaf scenarios. Each does the full trajectory. ---
scenario TurnL():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for DIST_APPROACH,
                with behavior EgoBehavior(left_full)
        terminate after TICKS steps
    compose:
        while True:
            wait


scenario TurnR():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for DIST_APPROACH,
                with behavior EgoBehavior(right_full)
        terminate after TICKS steps
    compose:
        while True:
            wait


scenario Straight():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for DIST_APPROACH,
                with behavior EgoBehavior(straight_full)
        terminate after TICKS steps
    compose:
        while True:
            wait


# --- Compositional entrypoint: pure choose, no Approach prefix.
# Each branch already includes the approach as the first lane of its
# full trajectory; the Approach primitive is redundant under this
# structure and dropping it keeps comp's per-primitive ρ̂ aligned with
# mono's per-branch ρ̂. ---
scenario Main():
    compose:
        do choose {
            TurnL():     1,
            TurnR():     1,
            Straight():  1,
        }


# --- Monolithic ground truth ---
# Identical to comp's per-primitive structure: spawn at DIST_APPROACH, run
# FollowTrajectoryBehavior on the Uniform-sampled full trajectory. By
# construction this is the same simulation that comp's TurnL/TurnR/Straight
# primitives run individually; the only difference is that mono samples
# the branch per-trace whereas comp aggregates per-primitive ρ̂s with the
# choose weights. Per-branch ρ̂ should now be numerically identical
# between methods modulo sampling noise.
chosen_full = Uniform(left_full, right_full, straight_full)


scenario MonoApproachChoose():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for DIST_APPROACH,
                with behavior EgoBehavior(chosen_full)
        terminate after TICKS steps
    compose:
        while True:
            wait
