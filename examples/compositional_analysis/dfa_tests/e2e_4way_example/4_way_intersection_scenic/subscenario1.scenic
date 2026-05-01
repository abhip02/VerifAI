mapPath = "/Users/abhi/Documents/seshia_research/compositional-analysis/Scenic/assets/maps/CARLA/Town07.xodr"

param map = mapPath
param carla_map = mapPath
param render = True

param DISTANCE_TO_INTERSECTION = VerifaiRange(-45, -35)
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

# go straight until intersection, stop at intersection
ego_trajectory = [straight_maneuver.startLane]

# Spawn each vehicle in the middle of its starting lane.
uberSpawnPoint = startLane.centerline[-1]

ego = new Car following roadDirection from uberSpawnPoint for globalParameters.DISTANCE_TO_INTERSECTION,
        with behavior EgoBehavior(trajectory = ego_trajectory)

record ego.speed as ego_speed
record ego.velocity.x as ego_vx
record ego.velocity.y as ego_vy
record ego.heading as ego_heading
record ego.position as ego_position
