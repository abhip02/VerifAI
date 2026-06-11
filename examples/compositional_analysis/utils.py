import os
import csv
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from train import make_env
from obstacles import add_obstacles


def generate_traces(
    seed: int = 0,
    save_dir: str = "storage/run0",
    model_path: str = None,
    expert: bool = False,
    n: int = 50,
    scenario: str = "XX",
    gif: bool = False,
    extra_obstacles: bool = False,
    obstacle_seed: int = 0,
    initial_speed_range: tuple = (40 / 3.6, 90 / 3.6),
    max_obstacle_distance: float = None,
    time_budget: float = float("inf"),
):
    """
    Runs MetaDrive simulation using a trained PPO model or expert policy and logs trajectory traces.

    Args:
        seed (int): Random seed for reproducibility.
        save_dir (str): Directory where traces or gifs will be saved.
        model_path (str): Path to the trained PPO model (.zip file). Not used if expert=True.
        expert (bool): If True, use expert policy instead of trained model.
        n (int): Number of test episodes to run.
        scenario (str or int): Scenario string or ID.
        gif (bool): If True, generate top-down gifs instead of CSV traces.
        extra_obstacles (bool): If True, add random obstacles to force stopping.
        obstacle_seed (int): Random seed for obstacle generation.
        time_budget (float): Wall-clock cap in seconds; episode loop stops
            when it elapses even if fewer than ``n`` episodes have run.
    """

    if not expert:
        assert model_path is not None, "You must provide a valid model_path (.zip file)"

    set_random_seed(seed)

    scenario_id = int(scenario) if str(scenario).isdigit() else scenario
    env = make_env(scenario=scenario_id, monitor=False)
    
    if expert:
        from metadrive.policy.expert_policy import ExpertPolicy
        model = None
        use_expert = True
    else:
        model = PPO.load(model_path)
        use_expert = False

    all_traces = []
    trace_id = 0
    spawned_obstacles = []

    os.makedirs(save_dir, exist_ok=True)

    if not gif:
        csv_path = os.path.join(save_dir, scenario, "traces.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        f = open(csv_path, "w", newline="")

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trace_id", "step", "x", "y", "heading",
                "speed", "action", "reward", "label"
            ]
        )
        writer.writeheader()

    deadline = time.monotonic() + float(time_budget)

    for ep in range(n):
        if time.monotonic() >= deadline:
            print(f"[{scenario}] time budget ({time_budget:.0f}s) reached "
                  f"after {trace_id} episodes")
            break
        if spawned_obstacles:
            env.engine.clear_objects([obj.id for obj in spawned_obstacles])
            spawned_obstacles = []

        obs, _ = env.reset()

        rng = np.random.default_rng(seed + ep)

        if extra_obstacles and rng.random() < 0.6:
            try:
                spawned_obstacles = add_obstacles(env, rng, max_pos_ahead=max_obstacle_distance) or []
            except Exception:
                pass
        
        if use_expert:
            expert_policy = ExpertPolicy(env.agent)

        initial_speed = rng.uniform(low=initial_speed_range[0], high=initial_speed_range[1])
        initial_velocity = env.agent.lane.direction * initial_speed
        env.agent.set_velocity(initial_velocity)

        done = False
        total_reward = 0.0
        step = 0
        label = False

        while not done and step <= env.config.horizon:
            if use_expert:
                action = expert_policy.act()
            else:
                action, _states = model.predict(obs, deterministic=True)
                
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            label = not done or info.get("arrive_dest")

            if gif:
                env.render(mode="topdown", screen_record=True, window=False)
            else:
                agent = env.agent
                pos = agent.position
                heading = agent.heading_theta
                vel = agent.speed

                row = {
                    "trace_id": trace_id,
                    "step": step,
                    "x": pos[0],
                    "y": pos[1],
                    "heading": heading,
                    "speed": vel,
                    "action": action.tolist() if hasattr(action, "tolist") else action,
                    "reward": reward,
                    "label": label
                }
                writer.writerow(row)

            step += 1

        if gif:
            gif_path = os.path.join(save_dir, f"trace_{trace_id:03d}.gif")
            env.top_down_renderer.generate_gif(gif_path)
            print(f"Saved gif to {gif_path}")

        trace_id += 1

    if not gif:
        f.close()

    env.close()
    return
