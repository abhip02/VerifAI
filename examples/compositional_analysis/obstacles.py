import numpy as np

from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
from metadrive.component.static_object.traffic_object import TrafficBarrier


def add_obstacles(env, rng, max_pos_ahead: float = None):
    """
    Spawn 1-3 tollgate-style checkpoints: a red traffic light paired with a
    physical TrafficBarrier that blocks the lane.

    Returns (lights, barriers) — two lists of spawned objects.
    Call ``release(lights, barriers, env)`` to turn lights green and clear barriers.
    Pass both lists to ``env.engine.clear_objects()`` before the next reset.
    """
    lane = env.agent.lane
    agent_lon = lane.local_coordinates(env.agent.position)[0]

    lights, barriers = [], []

    num = int(rng.integers(1, 3))
    pos_min, pos_max = 50.0, 90.0

    if max_pos_ahead is not None:
        pos_max = min(pos_max, max_pos_ahead)
        if pos_min >= pos_max:
            return lights, barriers

    positions = sorted(rng.uniform(pos_min, pos_max, size=num))

    for pos_ahead in positions:
        position = lane.position(agent_lon + pos_ahead, 0)
        seed = int(rng.integers(0, 10000))

        try:
            tl = env.engine.spawn_object(
                BaseTrafficLight,
                position=position,
                lane=lane,
                random_seed=seed,
            )
            tl.set_red()
            lights.append(tl)
        except Exception:
            pass

        try:
            bar = env.engine.spawn_object(
                TrafficBarrier,
                position=position,
                heading_theta=90.0,
                static=True,
                random_seed=seed + 1,
            )
            barriers.append(bar)
        except Exception:
            pass

    return lights, barriers


def release(lights, barriers, env):
    """Turn lights green and remove physical barriers."""
    for tl in lights:
        tl.set_green()
    if barriers:
        env.engine.clear_objects([b.id for b in barriers])
