# Reference: https://github.com/kevinchang73/VerifAI_Multi_Objective/blob/main/examples/multi_objective/uberCrashNewton.scenic

mapPath = "/Users/abhi/Documents/seshia_research/compositional-analysis/Scenic/assets/maps/CARLA/Town07.xodr"

param map = mapPath
param carla_map = mapPath
param render = True

param DISTANCE_TO_INTERSECTION = VerifaiRange(-15, -5)
param UBER_SPEED = VerifaiRange(3, 12)

model scenic.simulators.metadrive.model

# Ego vehicle just follows the trajectory specified later on.
behavior EgoBehavior(trajectory):
    do FollowTrajectoryBehavior(trajectory=trajectory, target_speed=globalParameters.UBER_SPEED)
    terminate

# Find all 4-way intersections and set up trajectories for each vehicle.
fourWayIntersection = filter(lambda i: i.is4Way, network.intersections)

# choose intersection
intersec = fourWayIntersection[0] # choose one
# intersec = Uniform(*fourWayIntersection) # random


rightLanes = filter(lambda lane: all([section._laneToRight is None for section in lane.sections]), intersec.incomingLanes)
startLane = rightLanes[0] # choose one
# startLane = Uniform(*rightLanes) # random

straight_maneuvers = filter(lambda i: i.type == ManeuverType.STRAIGHT, startLane.maneuvers)
straight_maneuver = Uniform(*straight_maneuvers)

otherLane = straight_maneuver.endLane.group.opposite.lanes[-1]
left_maneuvers = filter(lambda i: i.type == ManeuverType.LEFT_TURN, startLane.maneuvers)
left_maneuver = Uniform(*left_maneuvers)

right_maneuvers = filter(lambda i: i.type == ManeuverType.RIGHT_TURN, startLane.maneuvers)
right_maneuver = Uniform(*right_maneuvers)

# go straight at the intersection
straight_trajectory = [straight_maneuver.startLane, straight_maneuver.connectingLane, straight_maneuver.endLane]

# turn left at the intersection
left_trajectory = [straight_maneuver.startLane, left_maneuver.connectingLane, left_maneuver.endLane]

# turn right at the intersection
right_trajectory = [straight_maneuver.startLane, right_maneuver.connectingLane, right_maneuver.endLane]

chosen_trajectory = Uniform(*[straight_trajectory, left_trajectory, right_trajectory])

# Spawn each vehicle in the middle of its starting lane.
uberSpawnPoint = startLane.centerline[-1]

ego = new Car following roadDirection from uberSpawnPoint for globalParameters.DISTANCE_TO_INTERSECTION,
        with behavior EgoBehavior(trajectory = chosen_trajectory)

record ego.speed as ego_speed
record ego.heading as ego_heading
record ego.position as ego_position