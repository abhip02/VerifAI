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
import shutil

from utils import run_SMC, generate_traces


def run_SMC_monolithic(save_dir, scenarios, time_budget):
    for s in scenarios:
        scenario_dir = os.path.join(save_dir, s)

        # Remove old scenario directory completely
        if os.path.exists(scenario_dir):
            shutil.rmtree(scenario_dir)
    
    print("=== Running Monolithic SMC ===")
    results = {}
    all_traces = []
    start_time = time.time()

    for s in scenarios:
        elapsed = time.time() - start_time
        remaining_time = time_budget - elapsed
        if remaining_time <= 0:
            print(f"Time budget exhausted before scenario {s}")
            break
        print(f"Generating traces for {s} (time left: {remaining_time:.2f}s) ...")
        generate_traces(n=250, save_dir=save_dir, scenario=s, model_path="storage/models/model_map_2.zip")
        # all_traces.extend(traces)
        print(f"Finished {s}: generated {len(scenarios)} traces")
        
    logs = {}
    for s in scenarios:
        csv_path = os.path.join(save_dir, s, "traces.csv")
        if os.path.exists(csv_path):
            logs[s] = csv_path
        else:
            print(f"[INFO] Skipping {s}: no traces found at {csv_path}")
        scenario_base = ScenarioBase(logs)
    
    print("SMC")
    for s in logs:
        print(f"{s}: rho = {scenario_base.get_success_rate(s):.4f} ± {scenario_base.get_success_rate_uncertainty(s):.4f}")
    
    return logs


import os
import time
import shutil
import multiprocessing as mp

from verifai.compositional_analysis import ScenarioBase
from utils import generate_traces


def _worker_generate_traces(save_dir, scenario, n):
    print(f"[PID={os.getpid()}] Starting scenario {scenario}")
    generate_traces(
        n=n,
        save_dir=save_dir,
        scenario=scenario,
        model_path="storage/models/model_map_2.zip"
    )
    print(f"[PID={os.getpid()}] Finished scenario {scenario}")


def run_SMC_monolithic_parallel(n, save_dir, scenarios, time_budget):
    """
    Parallel monolithic SMC using multiprocessing:
    - Clears old trace directories
    - Launches trace generation jobs in parallel
    - Respects a soft time budget (stops launching new jobs once time expires)
    """

    # ----------------------------------------
    # 1. Clear old trace directories
    # ----------------------------------------
    for s in scenarios:
        scenario_dir = os.path.join(save_dir, s)
        if os.path.exists(scenario_dir):
            shutil.rmtree(scenario_dir)

    print("=== Running Monolithic SMC (Parallel) ===")

    # ----------------------------------------
    # 2. Launch multiprocessing jobs within time budget
    # ----------------------------------------
    processes = []
    start_time = time.time()

    for s in scenarios:
        elapsed = time.time() - start_time
        if elapsed >= time_budget:
            print(f"⛔ Time budget exhausted before launching scenario {s}")
            break

        print(f"Launching scenario {s} (elapsed: {elapsed:.2f}s)")

        p = mp.Process(
            target=_worker_generate_traces,
            args=(save_dir, s, n)
        )
        p.start()
        processes.append((s, p))

    # ----------------------------------------
    # 3. Wait for all launched jobs to finish
    # ----------------------------------------
    for scenario_name, proc in processes:
        print(f"⏳ Waiting for scenario {scenario_name}...")
        proc.join()   # no timeout → soft budget (only affects launch phase)
        print(f"✔ Scenario {scenario_name} finished.")

    # ----------------------------------------
    # 4. Build logs dict ONLY for existing traces
    # ----------------------------------------
    logs = {}
    for s, proc in processes:    # only consider scenarios that actually launched
        csv_path = os.path.join(save_dir, s, "traces.csv")
        if os.path.exists(csv_path):
            logs[s] = csv_path
        else:
            print(f"[INFO] Scenario {s} produced no traces.")

    if not logs:
        print("❌ No traces generated — cannot run SMC.")
        return {}

    # ----------------------------------------
    # 5. Run monolithic SMC
    # ----------------------------------------
    scenario_base = ScenarioBase(logs)

    print("\n=== Monolithic SMC Results ===")
    for s in logs:
        rho = scenario_base.get_success_rate(s)
        unc = scenario_base.get_success_rate_uncertainty(s)
        print(f"{s}: rho = {rho:.4f} ± {unc:.4f}")

    return logs



import time
from verifai.compositional_analysis import CompositionalAnalysisEngine, ScenarioBase

def run_SMC_compositional(scenarios, time_budget, logs):
    print("\n \n=== Running Compositional SMC ===")
    start_time = time.time()
    results = {}

    # Load base scenarios (assumes traces already exist in logs)
    scenario_base = ScenarioBase(logs)
    engine = CompositionalAnalysisEngine(scenario_base)

    for s in scenarios:
        elapsed = time.time() - start_time
        remaining_time = time_budget - elapsed
        if remaining_time <= 0:
            print(f"Time budget exhausted before scenario {s}")
            break

        # print(f"Analyzing {s} (time left: {remaining_time:.2f}s) ...")
        rho, uncertainty = engine.check(
            s,
            features=["x", "y", "heading", "speed"],
            norm_feat_idx=[0, 1],
        )

        print(f"Estimated {s}: rho = {rho:.4f} ± {uncertainty:.4f}")

        cex = engine.falsify(
            s,
            features=["x", "y", "heading", "speed"],
            norm_feat_idx=[0, 1],
            align_feat_idx=[0, 1],
        )

        # print(f"Counterexample for {s}: {cex}")
        results[s] = {"rho": rho, "uncertainty": uncertainty, "counterexample": cex}

    return results



def run_analysis(budget):
    n = 5000
    
    base_scenarios = ["S", "X", "O","C"]
    all_scenarios = ["S", "X", "O","C","SX","SO","SC","SXS","SOS","SCS"]
    nonbase_scenarios = ["SX","SO","SC","SXS","SOS","SCS"]

    
    save_dir = "storage/run0"
    
    # MONOLITHIC:
    # logs = run_SMC_monolithic(save_dir, scenarios=base_scenarios, time_budget=budget)
    logs = run_SMC_monolithic_parallel(n=n, save_dir=save_dir, scenarios=base_scenarios, time_budget=budget)
    
    existing_base_logs = {
    "S": "storage/run0/S/traces.csv",
    "X": "storage/run0/X/traces.csv",
    "O": "storage/run0/O/traces.csv",
    "C": "storage/run0/C/traces.csv"
    }
    
    logs.update(existing_base_logs)
    
    # COMPOSITIONAL
    results_compositional = run_SMC_compositional(scenarios=all_scenarios, time_budget=budget, logs = logs)
        
    return


if __name__ == "__main__":
    mp.set_start_method("spawn")   # required on macOS
    run_analysis(180)
