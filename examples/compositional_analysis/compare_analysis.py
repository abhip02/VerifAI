import os
import time
import shutil
import multiprocessing as mp
from verifai.compositional_analysis import ScenarioBase, CompositionalAnalysisEngine
from utils import generate_traces


def _worker_generate_traces(save_dir, scenario, n, expert):
    print(f"[PID={os.getpid()}] Starting scenario {scenario}")
    generate_traces(
        n=n,
        save_dir=save_dir,
        scenario=scenario,
        model_path="storage/models/model_map_2.zip",
        expert=expert
    )
    print(f"[PID={os.getpid()}] Finished scenario {scenario}")


def generate_traces_parallel_soft_stop(n, save_dir, scenarios, time_budget, expert):
    """
    Generate traces in parallel using multiprocessing with SOFT STOP.
    Stops launching new processes after time budget, but lets running processes finish.
    """
    # Clear old trace directories
    for s in scenarios:
        scenario_dir = os.path.join(save_dir, s)
        if os.path.exists(scenario_dir):
            shutil.rmtree(scenario_dir)
    
    print("=== Generating Traces (Parallel - SOFT STOP) ===")
    
    # Launch multiprocessing jobs within time budget
    processes = []
    start_time = time.time()
    
    for s in scenarios:
        elapsed = time.time() - start_time
        if elapsed >= time_budget:
            print(f"Time budget exhausted before launching scenario {s}")
            break
        
        print(f"Launching scenario {s} (elapsed: {elapsed:.2f}s)")
        p = mp.Process(
            target=_worker_generate_traces,
            args=(save_dir, s, n, expert)
        )
        p.start()
        processes.append((s, p))
    
    # Wait for all launched jobs to finish
    for scenario_name, proc in processes:
        print(f"Waiting for scenario {scenario_name}...")
        proc.join()
        print(f"Scenario {scenario_name} finished.")
    
    # Build logs dict ONLY for existing traces
    logs = {}
    for s, proc in processes:
        csv_path = os.path.join(save_dir, s, "traces.csv")
        if os.path.exists(csv_path):
            logs[s] = csv_path
        else:
            print(f"[INFO] Scenario {s} produced no traces.")
    
    if not logs:
        print("No traces generated.")
    
    return logs


def generate_traces_parallel_hard_stop(n, save_dir, scenarios, time_budget, expert):
    """
    Generate traces in parallel using multiprocessing with HARD STOP.
    Terminates all processes when time budget is reached, discarding only the current partial trace.
    Completed traces are kept.
    """
    # Clear old trace directories
    for s in scenarios:
        scenario_dir = os.path.join(save_dir, s)
        if os.path.exists(scenario_dir):
            shutil.rmtree(scenario_dir)
    
    print("=== Generating Traces (Parallel - HARD STOP) ===")
    
    # Launch all processes
    processes = []
    start_time = time.time()
    
    for s in scenarios:
        print(f"Launching scenario {s}")
        p = mp.Process(
            target=_worker_generate_traces,
            args=(save_dir, s, n, expert)
        )
        p.start()
        processes.append((s, p))
    
    # Monitor time budget and terminate if exceeded
    trace_counts_before_termination = {}
    while True:
        elapsed = time.time() - start_time
        
        # Check if time budget exceeded
        if elapsed >= time_budget:
            print(f"\n[HARD STOP] Time budget ({time_budget}s) reached at {elapsed:.2f}s")
            
            # Record trace counts RIGHT BEFORE termination
            for s in scenarios:
                csv_path = os.path.join(save_dir, s, "traces.csv")
                if os.path.exists(csv_path):
                    with open(csv_path, 'r') as f:
                        lines = f.readlines()
                        # Count lines (subtract 1 for header if present)
                        trace_counts_before_termination[s] = len(lines) - 1 if lines else 0
                else:
                    trace_counts_before_termination[s] = 0
            
            print("Terminating all running processes...")
            
            for scenario_name, proc in processes:
                if proc.is_alive():
                    print(f"Terminating scenario {scenario_name} (PID={proc.pid})")
                    proc.terminate()
                    proc.join(timeout=5)  # Wait up to 5 seconds for graceful termination
                    if proc.is_alive():
                        print(f"Force killing scenario {scenario_name}")
                        proc.kill()
                        proc.join()
            break
        
        # Check if all processes finished naturally
        all_done = all(not proc.is_alive() for _, proc in processes)
        if all_done:
            print(f"All processes finished before time budget (elapsed: {elapsed:.2f}s)")
            break
        
        time.sleep(0.1)  # Check every 100ms
    
    # Build logs dict for scenarios with valid traces
    logs = {}
    for s, proc in processes:
        csv_path = os.path.join(save_dir, s, "traces.csv")
        
        if os.path.exists(csv_path):
            # If process was terminated, restore to pre-termination state
            # (remove the last partial trace that was being written)
            if proc.exitcode != 0 and s in trace_counts_before_termination:
                with open(csv_path, 'r') as f:
                    lines = f.readlines()
                
                current_count = len(lines) - 1 if lines else 0
                expected_count = trace_counts_before_termination[s]
                
                if current_count > expected_count:
                    # There's a partial trace - keep only the completed traces
                    print(f"[INFO] Scenario {s}: Removing partial trace (had {current_count}, keeping {expected_count})")
                    with open(csv_path, 'w') as f:
                        # Keep header + expected number of complete traces
                        f.writelines(lines[:expected_count + 1])
                
                # Only add to logs if there are any complete traces
                if expected_count > 0:
                    logs[s] = csv_path
                    print(f"[INFO] Scenario {s} has {expected_count} completed traces.")
                else:
                    print(f"[INFO] Scenario {s} had no completed traces.")
            else:
                # Process completed successfully
                logs[s] = csv_path
                print(f"[INFO] Scenario {s} completed successfully with traces.")
        else:
            print(f"[INFO] Scenario {s} produced no traces.")
    
    if not logs:
        print("No traces generated.")
    
    return logs


def run_monolithic_smc(logs):
    """
    Run monolithic SMC analysis on generated traces.
    """
    if not logs:
        print("No traces to analyze.")
        return {}
    
    scenario_base = ScenarioBase(logs)
    
    print("\n=== Monolithic SMC Results ===")
    for s in logs:
        rho = scenario_base.get_success_rate(s)
        unc = scenario_base.get_success_rate_uncertainty(s)
        print(f"{s}: rho = {rho:.4f} ± {unc:.4f}")
    
    return logs


def run_SMC_compositional(scenarios, time_budget, logs):
    print("\n=== Running Compositional SMC ===")
    start_time = time.time()
    results = {}

    scenario_base = ScenarioBase(logs)
    engine = CompositionalAnalysisEngine(scenario_base)

    for s in scenarios:
        elapsed = time.time() - start_time
        remaining_time = time_budget - elapsed
        if remaining_time <= 0:
            print(f"Time budget exhausted before scenario {s}")
            break

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

        results[s] = {"rho": rho, "uncertainty": uncertainty, "counterexample": cex}

    return results


def parse_scenario(input_scenario):
    scenarios_set = set()
    for s in input_scenario:
        scenarios_set.add(s)
    return scenarios_set


def testScenario(input_scenario, isCompositional, time_budget, n, save_dir, expert, hard_stop=True):
    """
    Test scenario with either soft or hard time budget enforcement.
    
    Args:
        hard_stop: If True, uses hard stop (terminates processes at time budget).
                   If False, uses soft stop (stops launching new processes but lets running ones finish).
    """
    # Select appropriate trace generation function
    if hard_stop:
        generate_traces_func = generate_traces_parallel_hard_stop
    else:
        generate_traces_func = generate_traces_parallel_soft_stop
    
    # monolithic trace generation
    if not isCompositional:
        print("Running MONOLITHIC SMC")
        scenarios = [input_scenario]
        
        logs = generate_traces_func(n=n, save_dir=save_dir, scenarios=scenarios, time_budget=time_budget, expert=expert)
        run_monolithic_smc(logs)
        
    # compositional trace generation
    else:
        print("Running COMPOSITIONAL SMC on primitive cases (what compositional will use)")
        scenarios_set = parse_scenario(input_scenario)
        scenarios = list(scenarios_set)
        
        logs = generate_traces_func(n=n, save_dir=save_dir, scenarios=scenarios, time_budget=time_budget, expert=expert)
        
        # checking individual rho's
        run_monolithic_smc(logs)
        
        # compositional rho
        run_SMC_compositional(scenarios=[input_scenario], time_budget=time_budget, logs=logs)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    
    input_scenario = "SXC"
    isCompositional = True
    time_budget = 25
    n = 5000
    expert = True
    save_dir = "storage/run1"
    hard_stop = True  # Set to False for soft stop behavior
    
    testScenario(input_scenario=input_scenario, isCompositional=isCompositional, 
                 time_budget=time_budget, n=n, save_dir=save_dir,
                 expert=expert, hard_stop=hard_stop)