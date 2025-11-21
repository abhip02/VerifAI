import os
import time
import shutil
import multiprocessing as mp
from verifai.compositional_analysis import ScenarioBase, CompositionalAnalysisEngine
from utils import generate_traces


def _worker_generate_traces(save_dir, scenario, n, expert, model_path):
    print(f"[PID={os.getpid()}] Starting scenario {scenario}")
    generate_traces(
        n=n,
        save_dir=save_dir,
        scenario=scenario,
        model_path=model_path,
        expert=expert
    )
    print(f"[PID={os.getpid()}] Finished scenario {scenario}")


def generate_traces_parallel(n, save_dir, scenarios, time_budget, expert, model_path):
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
            args=(save_dir, s, n, expert, model_path)
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


def testScenario(input_scenario, isCompositional, time_budget, n, save_dir, expert, model_path):
    """
    Test scenario with hard time budget enforcement.
    Terminates all processes when time budget is reached.
    """
    # monolithic trace generation
    if not isCompositional:
        print("Running MONOLITHIC SMC")
        scenarios = [input_scenario]
        
        logs = generate_traces_parallel(n=n, save_dir=save_dir, scenarios=scenarios, time_budget=time_budget, expert=expert, model_path=model_path)
        run_monolithic_smc(logs)
        
    # compositional trace generation
    else:
        print("Running COMPOSITIONAL SMC on primitive cases (what compositional will use)")
        scenarios_set = parse_scenario(input_scenario)
        scenarios = list(scenarios_set)
        
        logs = generate_traces_parallel(n=n, save_dir=save_dir, scenarios=scenarios, time_budget=time_budget, expert=expert, model_path=model_path)
        
        # checking individual rho's
        run_monolithic_smc(logs)
        
        # compositional rho
        run_SMC_compositional(scenarios=[input_scenario], time_budget=time_budget, logs=logs)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SMC tests with compositional or monolithic approaches")
    parser.add_argument("--scenario", type=str, default="SXC", help="Input scenario string (default: SXC)")
    parser.add_argument("--compositional", action="store_true", help="Use compositional approach (default: False)")
    parser.add_argument("--time_budget", type=int, default=25, help="Time budget in seconds (default: 25)")
    parser.add_argument("--n", type=int, default=5000, help="Number of traces to generate (default: 5000)")
    parser.add_argument("--expert", action="store_true", help="Use expert mode (default: False)")
    parser.add_argument("--save_dir", type=str, default="storage/run1", help="Directory to save traces (default: storage/run1)")
    parser.add_argument("--model_path", type=str, default="storage/models/model_map_2.zip", help="Path to model file (default: storage/models/model_map_2.zip)")
    
    args = parser.parse_args()
    
    mp.set_start_method("spawn")
    
    testScenario(
        input_scenario=args.scenario,
        isCompositional=args.compositional,
        time_budget=args.time_budget,
        n=args.n,
        save_dir=args.save_dir,
        expert=args.expert,
        model_path=args.model_path
    )