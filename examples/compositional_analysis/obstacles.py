import numpy as np

# Weighted pool of obstacle types. Duplicates increase probability.
OBSTACLE_TYPES = [
    "traffic_light",
    "traffic_light",
    "traffic_light",
    "box",
    "box",
    "cone",
    "cone",
    "warning_triangle",
    "barrier",
    "debris",
    "broken_vehicle",
    "crashed_vehicle",
]

def add_obstacles(env, rng):
    """
    Add random obstacles to a MetaDrive environment ahead of the agent.

    Returns a list of spawned objects that the caller must pass to
    ``env.engine.clear_objects()`` before the next ``env.reset()``.

    Obstacle types:
      - traffic_light    : red traffic light on the lane
      - box              : static box on the road
      - cone             : traffic cone
      - warning_triangle : road warning triangle
      - barrier          : concrete road barrier
      - debris           : cluster of small debris pieces
      - broken_vehicle   : stationary broken-down vehicle in the lane
      - crashed_vehicle  : vehicle angled across the lane (simulates a crash)

    Scenarios:
      - many_close  : 3-6 obstacles at 10-40 m ahead
      - few_far     : 1-3 obstacles at 50-100 m ahead
      - mixed       : 2-5 obstacles at 15-70 m ahead
      - blocked     : 1-2 large obstacles (vehicle/barrier) at 20-40 m, forcing a hard stop
      - no_stop     : no obstacles
    """
    from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight

    lane = env.agent.lane
    lane_width = lane.width
    agent_pos = env.agent.position[0]
    spawned = []

    scenario = rng.choice(["many_close", "few_far", "mixed", "blocked", "no_stop"])

    if scenario == "many_close":
        num_obstacles = rng.integers(3, 7)
        pos_range = (10, 40)
        type_pool = OBSTACLE_TYPES
    elif scenario == "few_far":
        num_obstacles = rng.integers(1, 4)
        pos_range = (50, 100)
        type_pool = OBSTACLE_TYPES
    elif scenario == "mixed":
        num_obstacles = rng.integers(2, 6)
        pos_range = (15, 70)
        type_pool = OBSTACLE_TYPES
    elif scenario == "blocked":
        num_obstacles = rng.integers(1, 3)
        pos_range = (20, 40)
        type_pool = ["barrier", "broken_vehicle", "crashed_vehicle", "barrier", "crashed_vehicle"]
    else:  # no_stop
        return spawned

    for _ in range(num_obstacles):
        obstacle_type = rng.choice(type_pool)
        pos_ahead = rng.uniform(*pos_range)
        lateral_offset = rng.uniform(-0.3, 0.3) * lane_width
        position = lane.position(agent_pos + pos_ahead, 0)
        position = (position[0], position[1] + lateral_offset)
        seed = int(rng.integers(0, 10000))

        if obstacle_type == "traffic_light":
            try:
                tl = env.engine.spawn_object(
                    BaseTrafficLight,
                    position=position,
                    lane=lane,
                    random_seed=seed,
                )
                tl.set_red()
                spawned.append(tl)
            except Exception:
                pass

        elif obstacle_type == "box":
            try:
                obj = env.engine.spawn_object(
                    "box",
                    position=position,
                    heading=0.0,
                    size=(1.0, 0.5),
                    random_seed=seed,
                )
                spawned.append(obj)
            except Exception:
                pass

        elif obstacle_type == "cone":
            try:
                obj = env.engine.spawn_object(
                    "cone",
                    position=position,
                    random_seed=seed,
                )
                spawned.append(obj)
            except Exception:
                pass

        elif obstacle_type == "warning_triangle":
            try:
                obj = env.engine.spawn_object(
                    "triangle",
                    position=position,
                    random_seed=seed,
                )
                spawned.append(obj)
            except Exception:
                pass

        elif obstacle_type == "barrier":
            try:
                obj = env.engine.spawn_object(
                    "barrier",
                    position=position,
                    heading=0.0,
                    random_seed=seed,
                )
                spawned.append(obj)
            except Exception:
                pass

        elif obstacle_type == "debris":
            # Scatter 2–4 small debris pieces around the base position.
            n_pieces = int(rng.integers(2, 5))
            for _ in range(n_pieces):
                scatter = rng.uniform(-0.8, 0.8)
                debris_pos = (position[0] + scatter, position[1] + rng.uniform(-0.4, 0.4))
                try:
                    obj = env.engine.spawn_object(
                        "box",
                        position=debris_pos,
                        heading=float(rng.uniform(0, 360)),
                        size=(0.3, 0.3),
                        random_seed=int(rng.integers(0, 10000)),
                    )
                    spawned.append(obj)
                except Exception:
                    pass

        elif obstacle_type == "broken_vehicle":
            try:
                from metadrive.component.vehicle.base_vehicle import BaseVehicle
                obj = env.engine.spawn_object(
                    BaseVehicle,
                    position=position,
                    heading=0.0,
                    random_seed=seed,
                )
                spawned.append(obj)
            except Exception:
                pass

        elif obstacle_type == "crashed_vehicle":
            # Angle the vehicle across the lane to simulate a crash.
            angle = float(rng.choice([30.0, 45.0, 60.0, -30.0, -45.0, -60.0]))
            try:
                from metadrive.component.vehicle.base_vehicle import BaseVehicle
                obj = env.engine.spawn_object(
                    BaseVehicle,
                    position=position,
                    heading=angle,
                    random_seed=seed,
                )
                spawned.append(obj)
            except Exception:
                pass

    return spawned
