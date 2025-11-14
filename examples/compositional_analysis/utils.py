import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import os
import csv
import argparse
import numpy as np
import gymnasium as gym
from functools import partial
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from metadrive.envs import MetaDriveEnv
from IPython.display import Image, clear_output
from metadrive.utils.doc_utils import generate_gif
from metadrive.component.map.base_map import BaseMap
from stable_baselines3.common.utils import set_random_seed
from metadrive.component.map.pg_map import MapGenerateMethod
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from metadrive.utils.draw_top_down_map import draw_top_down_map
from train import make_env
import time

import pandas as pd
from verifai.compositional_analysis import CompositionalAnalysisEngine, ScenarioBase

def run_SMC(mode="monolithic"):
    """
    Run Statistical Model Checking (SMC) on the given logs.

    Args:
        logs (dict): Mapping from scenario name to its trace CSV path.
        mode (str): "monolithic" or "compositional".
                    - "monolithic" computes rho directly from traces.
                    - "compositional" uses compositional reasoning on base traces.

    Returns:
        dict: A dictionary mapping each scenario to (rho, uncertainty, counterexample)
    """
    assert mode in ["monolithic", "compositional"], "Mode must be 'monolithic' or 'compositional'."

    logs = {
        "S": "storage/traces/S/traces.csv",
        "X": "storage/traces/X/traces.csv",
        "O": "storage/traces/O/traces.csv",
        "C": "storage/traces/C/traces.csv",
        "SX": "storage/traces/SX/traces.csv",
        "SO": "storage/traces/SO/traces.csv",
        "SC": "storage/traces/SC/traces.csv",
        "SXS": "storage/traces/SXS/traces.csv",
        "SOS": "storage/traces/SOS/traces.csv",
        "SCS": "storage/traces/SCS/traces.csv",
    }

    scenario_base = ScenarioBase(logs)
    results = {}

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    if mode == "monolithic":
        print("=== MONOLITHIC SMC ===")
        for s in logs:
            rho = scenario_base.get_success_rate(s)
            uncertainty = scenario_base.get_success_rate_uncertainty(s)
            results[s] = {
                "rho": rho,
                "uncertainty": uncertainty,
                "counterexample": None,
            }
            print(f"{s}: rho = {rho:.4f} ± {uncertainty:.4f}")

    else:  # compositional
        print("=== COMPOSITIONAL SMC ===")
        engine = CompositionalAnalysisEngine(scenario_base)
        for s in logs:
            rho, uncertainty = engine.check(
                s,
                features=["x", "y", "heading", "speed"],
                norm_feat_idx=[0, 1],
            )
            cex = engine.falsify(
                s,
                features=["x", "y", "heading", "speed"],
                norm_feat_idx=[0, 1],
                align_feat_idx=[0, 1],
            )
            results[s] = {
                "rho": rho,
                "uncertainty": uncertainty,
                "counterexample": cex,
            }
            print(f"{s}: rho = {rho:.4f} ± {uncertainty:.4f}")
            print(f"Counterexample = {cex}")

    return results


def generate_traces(
    seed: int = 0,
    save_dir: str = "storage/run0",
    model_path: str = None,
    n: int = 50,
    scenario: str = "XX",
    gif: bool = False
):
    """
    Runs MetaDrive simulation using a trained PPO model and logs trajectory traces.

    Args:
        seed (int): Random seed for reproducibility.
        save_dir (str): Directory where traces or gifs will be saved.
        model_path (str): Path to the trained PPO model (.zip file).
        n (int): Number of test episodes to run.
        scenario (str or int): Scenario string or ID.
        gif (bool): If True, generate top-down gifs instead of CSV traces.
    """

    assert model_path is not None, "You must provide a valid model_path (.zip file)"

    set_random_seed(seed)

    scenario_id = int(scenario) if str(scenario).isdigit() else scenario
    env = make_env(scenario=scenario_id, monitor=False)
    model = PPO.load(model_path)

    all_traces = []
    trace_id = 0

    # Create save dir
    os.makedirs(save_dir, exist_ok=True)

    if not gif:
        csv_path = save_dir + "/" + scenario + "/traces.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        # csv_path = os.path.join(save_dir, "/" + scenario + "/traces.csv")
        f = open(csv_path, "w", newline="")
        
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trace_id", "step", "x", "y", "heading",
                "speed", "action", "reward", "label"
            ]
        )
        writer.writeheader()

    for ep in range(n):
        obs, _ = env.reset()

        initial_speed = np.random.uniform(low=70/3.6, high=80/3.6)
        initial_velocity = env.agent.lane.direction * initial_speed
        env.agent.set_velocity(initial_velocity)

        done = False
        total_reward = 0.0
        step = 0
        label = False

        # print(f"\n=== Episode {ep+1}/{n} ===")
        while not done and step <= env.config.horizon:
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

        # print(f"Label: {label}")
        # print(f"Episode reward: {total_reward:.2f}")

        if gif:
            gif_path = os.path.join(save_dir, f"trace_{trace_id:03d}.gif")
            env.top_down_renderer.generate_gif(gif_path)
            print(f"Saved gif to {gif_path}")

        trace_id += 1

    if not gif:
        f.close()

    env.close()
    return