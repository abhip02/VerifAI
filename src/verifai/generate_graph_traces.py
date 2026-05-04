from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import re
from pathlib import Path
import shutil
import tempfile
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from verifai.compositional_analysis import CompositionalAnalysisEngine, ScenarioBase
from verifai.scenic_composition_analysis import build_enriched_graph

DEFAULT_SCENIC_MODEL = "scenic.simulators.newtonian.model"

# -----------------------------------------------------------------------------
# Simulator backend registry
#
# Each entry maps a short CLI name to a backend specification:
#   "kind"          : "scenic"   -> runs through scenario.getSimulator() with
#                                   the given Scenic model string
#                     "native"   -> handled by a custom adapter below
#                                   (used for VerifAI's pyglet car_simulator)
#   "scenic_model"  : Scenic model name (for kind="scenic")
#   "requires"      : list of importable Python module names that must be
#                     installed for the backend to actually run
#   "install_hint"  : human-readable pip/setup hint
#   "description"   : short description surfaced in --help
# -----------------------------------------------------------------------------
BACKENDS: Dict[str, Dict[str, object]] = {
    "newtonian": {
        "kind": "scenic",
        "scenic_model": "scenic.simulators.newtonian.model",
        "requires": ["scenic.simulators.newtonian"],
        "install_hint": "bundled with scenic",
        "description": "Scenic's built-in Newtonian simulator (plain kinematic model).",
    },
    "newtonian-driving": {
        "kind": "scenic",
        "scenic_model": "scenic.simulators.newtonian.driving_model",
        "requires": ["scenic.simulators.newtonian"],
        "install_hint": "bundled with scenic; requires an .xodr map via `param map`",
        "description": "Newtonian simulator + driving domain (Car, SetThrottleAction, ...).",
    },
    "carla": {
        "kind": "scenic",
        "scenic_model": "scenic.simulators.carla.model",
        "requires": ["carla", "scenic.simulators.carla"],
        "install_hint": "install CARLA 0.9.x and `pip install carla`; run the CARLA server first",
        "description": "CARLA 0.9.x via Scenic. Requires a running CARLA server on :2000.",
    },
    "metadrive": {
        "kind": "scenic",
        "scenic_model": "scenic.simulators.metadrive.model",
        "requires": ["metadrive", "scenic.simulators.metadrive"],
        "install_hint": "`pip install metadrive-simulator` (and `scenic[metadrive]` if prompted)",
        "description": "MetaDrive via Scenic. Self-contained, no external server.",
    },
    "lgsvl": {
        "kind": "scenic",
        "scenic_model": "scenic.simulators.lgsvl.model",
        "requires": ["lgsvl", "scenic.simulators.lgsvl"],
        "install_hint": "`pip install lgsvl`; requires a running LGSVL simulator",
        "description": "LGSVL via Scenic. Requires LGSVL running.",
    },
    "webots": {
        "kind": "scenic",
        "scenic_model": "scenic.simulators.webots.model",
        "requires": ["scenic.simulators.webots"],
        "install_hint": "requires the Webots GUI and a .wbt world; launch webots first",
        "description": "Webots via Scenic. Requires Webots running with the target world.",
    },
    "xplane": {
        "kind": "scenic",
        "scenic_model": "scenic.simulators.xplane.model",
        "requires": ["scenic.simulators.xplane"],
        "install_hint": "requires X-Plane running with the Scenic plugin",
        "description": "X-Plane via Scenic. Requires X-Plane running.",
    },
    "car_simulator": {
        "kind": "native",
        "scenic_model": None,
        "requires": ["pyglet", "verifai.simulators.car_simulator.car_object"],
        "install_hint": "bundled with verifai; uses bicycle-model dynamics (no road network)",
        "description": (
            "VerifAI's native pyglet car simulator. Does NOT consume Scenic "
            "scenes; runs the bicycle_model with a per-primitive control "
            "policy and writes the same traces.csv schema."
        ),
    },
}


def _backend_available(backend: str) -> Tuple[bool, str]:
    """Return (True, '') if every required module for *backend* can be imported,
    else (False, human-readable reason)."""
    import importlib

    spec = BACKENDS[backend]
    missing: List[str] = []
    for mod in spec["requires"]:  # type: ignore[assignment]
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    if missing:
        return (
            False,
            f"backend '{backend}' is unavailable: missing {missing}. "
            f"Hint: {spec['install_hint']}.",
        )
    return True, ""


def list_backends() -> str:
    """Human-readable backend table for --help / CLI output."""
    lines = ["Available backends:"]
    for name, spec in BACKENDS.items():
        ok, reason = _backend_available(name)
        tag = "OK " if ok else "NO "
        lines.append(f"  [{tag}] {name:<20s} {spec['description']}")
        if not ok:
            lines.append(f"         -> {reason}")
    return "\n".join(lines)


def compute_hoeffding_samples(confidence_level: float, error_bound: float) -> int:
    """Compute the number of samples needed using Hoeffding's inequality."""

    delta = 1 - confidence_level
    n = math.log(2 / delta) / (2 * error_bound**2)
    return int(math.ceil(n))


def resolve_backend(
    backend: Optional[str],
    explicit_model: Optional[str],
    source_text: str,
) -> Tuple[str, Optional[str]]:
    """Decide which backend/scenic-model to run under.

    Precedence:
      1. If --backend is given, it wins (and we validate availability).
      2. Else if --model is given, we honor it (kind='scenic').
      3. Else we sniff the source's `model ...` directive. If it names a
         driving-flavored model we auto-pick 'newtonian-driving' in spirit
         (leaving the scenic model as declared).
      4. Fallback: 'newtonian'.

    Returns (backend_name, scenic_model_string_or_None).
    scenic_model is None when kind='native' (car_simulator).
    """
    if backend:
        if backend not in BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}. Known: {list(BACKENDS)}")
        ok, reason = _backend_available(backend)
        if not ok:
            raise RuntimeError(reason)
        spec = BACKENDS[backend]
        return backend, spec["scenic_model"]  # type: ignore[return-value]

    if explicit_model:
        return ("custom", explicit_model)

    declared = re.search(
        r"(?m)^\s*model\s+([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+)",
        source_text or "",
    )
    if declared:
        name = declared.group(1)
        # Match whichever backend advertises exactly this scenic model.
        for bname, spec in BACKENDS.items():
            if spec["scenic_model"] == name:
                return bname, name
        return ("custom", name)

    return "newtonian", BACKENDS["newtonian"]["scenic_model"]  # type: ignore[return-value]


def build_trace_jobs(
    source: Union[str, Path],
    save_dir: Union[str, Path],
    n: Optional[int] = None,
    *,
    mode2d: bool = True,
    model: Optional[str] = None,
    backend: Optional[str] = None,
    max_iterations: int = 2000,
    max_steps: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Create one Scenic trace-generation job per primitive spec in the enriched graph."""

    source_text = (
        Path(source).read_text(encoding="utf-8") if Path(source).exists() else ""
    )
    backend_name, scenic_model = resolve_backend(backend, model, source_text)

    enriched = build_enriched_graph(source, model=scenic_model)
    save_dir = Path(save_dir)

    # Persist the parsed enriched graph as JSON for downstream compositional
    # analysis (e.g. mapping primitives to DFA states).
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        graph_json_path = save_dir / f"{Path(source).stem}_graph.json"
        with graph_json_path.open("w", encoding="utf-8") as f:
            json.dump(enriched, f, indent=2, default=str)
        print(f"[graph] wrote parsed graph to {graph_json_path}")
    except Exception as exc:
        print(f"[graph] WARNING: could not write parsed graph JSON: {exc}")

    jobs: List[Dict[str, object]] = []
    for primitive in enriched["primitive_specs"]:
        if not primitive.get("is_executable"):
            continue
        jobs.append(
            {
                "primitive": primitive["name"],
                "kind": primitive["kind"],
                "source_path": primitive["source_path"],
                "root_source_path": primitive.get("root_source_path"),
                "wrapper_source": primitive["wrapper_source"],
                "wrapper_entrypoint": primitive["wrapper_entrypoint"],
                "save_dir": str(save_dir),
                "n": n,
                "mode2d": mode2d,
                "model": scenic_model,
                "backend": backend_name,
                "max_iterations": max_iterations,
                "max_steps": max_steps,
            }
        )

    return jobs


def _csv_path_for_primitive(save_dir: Union[str, Path], primitive: str) -> Path:
    return Path(save_dir) / primitive / "traces.csv"


def _count_trace_ids(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) <= 1:
        return 0
    trace_ids = set()
    for line in lines[1:]:
        parts = line.split(",")
        if parts:
            trace_ids.add(parts[0])
    return len(trace_ids)


def _trim_partial_trace(csv_path: Path, completed_count: int) -> None:
    with csv_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    with csv_path.open("w", encoding="utf-8") as f:
        f.write(lines[0])
        for line in lines[1:]:
            parts = line.split(",")
            if not parts:
                continue
            trace_id = int(parts[0])
            if trace_id < completed_count:
                f.write(line)


def _derive_heading_and_speed(
    positions: Sequence[Tuple[float, float]],
    step: int,
    dt: float,
) -> Tuple[float, float]:
    if len(positions) <= 1:
        return 0.0, 0.0

    if step < len(positions) - 1:
        x0, y0 = positions[step]
        x1, y1 = positions[step + 1]
    else:
        x0, y0 = positions[step - 1]
        x1, y1 = positions[step]

    dx = x1 - x0
    dy = y1 - y0
    heading = math.atan2(dy, dx) if dx or dy else 0.0
    speed = math.hypot(dx, dy) / dt if dt > 0 else 0.0
    return heading, speed


def _trajectory_rows(simulation, trace_id: int) -> List[Dict[str, object]]:
    trajectory = simulation.result.trajectory
    dt = float(getattr(simulation, "timestep", 1.0) or 1.0)
    termination_type = getattr(simulation.result, "terminationType", None)
    terminated_complete = getattr(termination_type, "name", "") == "scenarioComplete"

    positions = [(frame[0].x, frame[0].y) for frame in trajectory]
    actions_per_step = getattr(simulation.result, "actions", ()) or ()
    rewards_per_step = getattr(simulation.result, "rewards", None)

    rows: List[Dict[str, object]] = []
    n = len(positions)

    for step, (x, y) in enumerate(positions):
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        heading, speed = _derive_heading_and_speed(positions, step, dt)

        action_val: object = None
        if step < len(actions_per_step):
            a = actions_per_step[step]
            if isinstance(a, dict) and a:
                first = next(iter(a.values()))
                action_val = (
                    list(first)
                    if hasattr(first, "__iter__") and not isinstance(first, str)
                    else first
                )
            else:
                action_val = a

        reward_val = 0.0
        if rewards_per_step is not None and step < len(rewards_per_step):
            reward_val = float(rewards_per_step[step])

        # Expert-trace label semantics: True while episode is in-progress or ended in success.
        is_last = step == n - 1
        label = (not is_last) or terminated_complete

        rows.append(
            {
                "trace_id": trace_id,
                "step": step,
                "x": x,
                "y": y,
                "heading": heading,
                "speed": speed,
                "action": action_val,
                "reward": reward_val,
                "label": label,
            }
        )

    return rows


def _load_scenic():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    import scenic

    return scenic


def _car_simulator_control_policy(primitive_name: str):
    """Return a (trajectory, control_trajectory) -> [omega, acc] callable.

    Each primitive gets a deterministic-but-distinct control policy so
    traces have meaningful per-primitive structure. These are tiny
    signatures that exercise the bicycle_model — not a realistic driving
    policy. Swap in your own callable by naming a primitive
    ``Custom__<module>__<function>`` and we'll attempt to import it.
    """
    import numpy as np

    # User-pluggable hook: primitives named "Custom__mod__fn" load a policy.
    if primitive_name.startswith("Custom__"):
        try:
            _, mod, fn = primitive_name.split("__", 2)
            import importlib

            return getattr(importlib.import_module(mod), fn)
        except Exception:
            pass

    key = sum(ord(c) for c in primitive_name) % 7

    def policy(trajectory, control_trajectory):
        t = len(control_trajectory)
        if key == 0:  # steady cruise
            return [0.0, 1.5]
        if key == 1:  # stop-and-go
            return [0.0, 1.5 if t < 8 else (-3.0 if t < 14 else 1.5)]
        if key == 2:  # gentle left turn
            return [0.15, 1.0]
        if key == 3:  # gentle right turn
            return [-0.15, 1.0]
        if key == 4:  # decel then creep
            return [0.05, -2.0 if t < 6 else 0.2]
        if key == 5:  # sinusoidal steer
            return [0.3 * np.sin(0.2 * t), 1.2]
        return [0.0, 0.5]  # plod forward

    return policy


def _worker_native_car_simulator(job: Mapping[str, object]) -> None:
    """Native adapter: run VerifAI's car_simulator bicycle_model.

    This does NOT load the Scenic scene — it integrates the bicycle model
    for ``n`` episodes of ``max_steps`` ticks each, using a per-primitive
    control policy. Output matches the Scenic-backed trace CSV schema
    (trace_id, step, x, y, heading, speed, label) so downstream analysis
    is identical.
    """
    from verifai.simulators.car_simulator.car_object import bicycle_model
    import numpy as np

    primitive = str(job["primitive"])
    save_dir = Path(str(job["save_dir"]))
    n = job.get("n")
    max_steps = job.get("max_steps") or 50
    traces_to_generate = 1 if n is None else int(n)

    print(f"[PID={os.getpid()}] Starting primitive {primitive} (car_simulator)")
    primitive_dir = save_dir / primitive
    primitive_dir.mkdir(parents=True, exist_ok=True)
    csv_path = primitive_dir / "traces.csv"
    policy = _car_simulator_control_policy(primitive)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trace_id", "step", "x", "y", "heading", "speed", "label"],
        )
        writer.writeheader()

        dt = 0.1
        for trace_id in range(traces_to_generate):
            # Slight per-episode randomization so traces aren't identical.
            rng = np.random.default_rng(
                (hash(primitive) ^ trace_id) & 0xFFFFFFFFFFFFFFFF
            )
            x0 = [float(rng.uniform(-2, 2)), 0.0, 0.0, float(rng.uniform(-0.1, 0.1))]
            car = bicycle_model(
                x0=x0,
                u_domain=None,
                compute_control=policy,
                wheelbase=3.0,
                dt=dt,
            )
            for _ in range(int(max_steps)):
                car.step()

            # trajectory entries are [x, y, v, heading]
            for step_idx, state in enumerate(car.trajectory):
                x, y, v, heading = (
                    float(state[0]),
                    float(state[1]),
                    float(state[2]),
                    float(state[3]),
                )
                writer.writerow(
                    {
                        "trace_id": trace_id,
                        "step": step_idx,
                        "x": x,
                        "y": y,
                        "heading": heading,
                        "speed": v,
                        "label": False,
                    }
                )
    print(f"[PID={os.getpid()}] Finished primitive {primitive} (car_simulator)")


def _worker_dispatch(job: Mapping[str, object]) -> None:
    """Dispatch a single job to the right backend worker."""
    backend = str(job.get("backend") or "newtonian")
    if BACKENDS.get(backend, {}).get("kind") == "native":
        _worker_native_car_simulator(job)
    else:
        _worker_generate_traces(job)


def _worker_generate_traces(job: Mapping[str, object]) -> None:
    scenic = _load_scenic()

    primitive = str(job["primitive"])
    source_path = Path(str(job["source_path"]))
    root_source_path = Path(str(job.get("root_source_path") or source_path))
    wrapper_source = str(job["wrapper_source"])
    wrapper_entrypoint = str(job["wrapper_entrypoint"])
    save_dir = Path(str(job["save_dir"]))
    n = job.get("n")
    mode2d = bool(job.get("mode2d", True))
    model = job.get("model") or DEFAULT_SCENIC_MODEL
    max_iterations = int(job.get("max_iterations", 2000))
    max_steps = job.get("max_steps")

    print(f"[PID={os.getpid()}] Starting primitive {primitive}")

    traces_to_generate = 10**9 if n is None or n == float("inf") else int(n)
    primitive_dir = save_dir / primitive
    primitive_dir.mkdir(parents=True, exist_ok=True)
    csv_path = primitive_dir / "traces.csv"

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".scenic",
        prefix=f"__graph_trace_{primitive}_",
        dir=root_source_path.parent,
        encoding="utf-8",
        delete=False,
    ) as wrapper_file:
        wrapper_file.write(wrapper_source)
        wrapper_path = Path(wrapper_file.name)

    try:
        try:
            scenario = scenic.scenarioFromFile(
                str(wrapper_path),
                scenario=wrapper_entrypoint,
                mode2D=mode2d,
                model=model,
            )
            simulator = scenario.getSimulator()
        except Exception as exc:
            print(
                f"[PID={os.getpid()}] Primitive {primitive}: "
                f"failed to compile wrapper scenario: {exc}"
            )
            return

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "trace_id",
                    "step",
                    "x",
                    "y",
                    "heading",
                    "speed",
                    "action",
                    "reward",
                    "label",
                ],
            )
            writer.writeheader()

            trace_id = 0
            attempts = 0
            max_attempts = max(1000, traces_to_generate * 20)

            while trace_id < traces_to_generate and attempts < max_attempts:
                attempts += 1
                try:
                    scene, _ = scenario.generate(
                        maxIterations=max_iterations,
                        verbosity=0,
                    )
                    simulation = simulator.simulate(
                        scene,
                        maxSteps=max_steps,
                        verbosity=0,
                        maxIterations=1,
                    )
                except Exception as exc:
                    print(
                        f"[PID={os.getpid()}] Primitive {primitive}: "
                        f"skipping failed attempt {attempts}: {exc}"
                    )
                    continue

                if simulation is None:
                    continue

                for row in _trajectory_rows(simulation, trace_id):
                    writer.writerow(row)
                f.flush()

                if hasattr(simulation, "destroy"):
                    try:
                        simulation.destroy()
                    except Exception:
                        pass

                trace_id += 1
    finally:
        try:
            wrapper_path.unlink()
        except Exception:
            pass

    print(f"[PID={os.getpid()}] Finished primitive {primitive}")


def generate_traces_parallel(
    jobs: Sequence[Mapping[str, object]],
    time_budget: Union[int, float],
    reuse_traces: bool = False,
) -> Dict[str, str]:
    """Generate traces in parallel using multiprocessing with a hard stop."""

    if not jobs:
        return {}

    if not reuse_traces:
        for job in jobs:
            primitive_dir = Path(job["save_dir"]) / str(job["primitive"])
            if primitive_dir.exists():
                shutil.rmtree(primitive_dir)

    print("=== Generating Graph Traces (Parallel - HARD STOP) ===")

    processes: List[Tuple[str, mp.Process]] = []
    start_time = time.time()

    if not reuse_traces:
        for job in jobs:
            primitive = str(job["primitive"])
            print(f"Launching primitive {primitive}")
            process = mp.Process(target=_worker_dispatch, args=(job,))
            process.start()
            processes.append((primitive, process))

    trace_counts_before_termination: Dict[str, int] = {}
    if not reuse_traces:
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        total_target = sum(int(j.get("n") or 0) for j in jobs)
        bar = tqdm(total=total_target, desc="traces", unit="trace") if tqdm and total_target else None

        while True:
            elapsed = time.time() - start_time

            if bar is not None:
                done = sum(
                    _count_trace_ids(_csv_path_for_primitive(j["save_dir"], str(j["primitive"])))
                    for j in jobs
                )
                bar.n = min(done, total_target)
                bar.refresh()

            if time_budget != float("inf") and elapsed >= time_budget:
                print(
                    f"\n[HARD STOP] Time budget ({time_budget}s) reached at {elapsed:.2f}s"
                )
                for job in jobs:
                    primitive = str(job["primitive"])
                    trace_counts_before_termination[primitive] = _count_trace_ids(
                        _csv_path_for_primitive(job["save_dir"], primitive)
                    )

                print("Terminating all running processes...")
                for primitive, process in processes:
                    if process.is_alive():
                        print(f"Terminating primitive {primitive} (PID={process.pid})")
                        process.terminate()
                        process.join(timeout=5)
                        if process.is_alive():
                            print(f"Force killing primitive {primitive}")
                            process.kill()
                            process.join()
                break

            if all(not process.is_alive() for _, process in processes):
                print(f"All processes finished naturally (elapsed: {elapsed:.2f}s)")
                break

            time.sleep(1.0)

        if bar is not None:
            bar.close()

    logs: Dict[str, str] = {}
    for job in jobs:
        primitive = str(job["primitive"])
        csv_path = _csv_path_for_primitive(job["save_dir"], primitive)
        process = next((proc for name, proc in processes if name == primitive), None)

        if not csv_path.exists():
            print(f"[INFO] Primitive {primitive} produced no traces.")
            continue

        if (
            process is not None
            and process.exitcode != 0
            and primitive in trace_counts_before_termination
        ):
            current_count = _count_trace_ids(csv_path)
            expected_count = trace_counts_before_termination[primitive]

            if current_count > expected_count:
                print(
                    f"[INFO] Primitive {primitive}: removing partial episode "
                    f"(had {current_count} episodes, keeping {expected_count})"
                )
                _trim_partial_trace(csv_path, expected_count)

            if expected_count > 0:
                logs[primitive] = str(csv_path)
                print(
                    f"[INFO] Primitive {primitive} has {expected_count} completed episodes."
                )
            else:
                print(f"[INFO] Primitive {primitive} had no completed episodes.")
        else:
            episode_count = _count_trace_ids(csv_path)
            if episode_count > 0:
                logs[primitive] = str(csv_path)
            print(
                f"[INFO] Primitive {primitive} completed successfully "
                f"with {episode_count} episodes."
            )

    if not logs:
        print("No traces generated.")
    return logs


def run_monolithic_smc(logs: Mapping[str, str]) -> Dict[str, str]:
    """Run monolithic SMC analysis on generated traces."""

    if not logs:
        print("No traces to analyze.")
        return {}

    scenario_base = ScenarioBase(dict(logs))

    print("\n=== Monolithic SMC Results ===")
    for scenario_name in logs:
        rho = scenario_base.get_success_prob(scenario_name)
        unc = scenario_base.get_success_prob_uncertainty(scenario_name)
        print(f"{scenario_name}: rho = {rho:.4f} ± {unc:.4f}")

    return dict(logs)


def run_smc_compositional(
    scenarios: Sequence[Sequence[str]],
    time_budget: Union[int, float],
    logs: Mapping[str, str],
    delta: float = 0.05,
) -> Dict[str, Dict[str, object]]:
    print("\n=== Running Compositional SMC ===")
    start_time = time.time()
    results: Dict[str, Dict[str, object]] = {}

    scenario_base = ScenarioBase(dict(logs), delta=delta)
    engine = CompositionalAnalysisEngine(scenario_base)

    for scenario in scenarios:
        elapsed = time.time() - start_time
        remaining_time = time_budget - elapsed
        scenario_key = "".join(scenario)
        if time_budget != float("inf") and remaining_time <= 0:
            print(f"Time budget exhausted before scenario {scenario_key}")
            break

        rho, uncertainty = engine.check(
            list(scenario),
            features=["x", "y", "heading", "speed"],
            center_feat_idx=[0, 1],
        )
        print(f"Estimated {scenario_key}: rho = {rho:.4f} ± {uncertainty:.4f}")

        cex = engine.falsify(
            list(scenario),
            features=["x", "y", "heading", "speed"],
            center_feat_idx=[0, 1],
            align_feat_idx=[0, 1],
        )
        results[scenario_key] = {
            "rho": rho,
            "uncertainty": uncertainty,
            "counterexample": cex,
        }

    return results


def parse_scenario(input_scenario: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(input_scenario, str):
        return [input_scenario]
    return list(dict.fromkeys(input_scenario))


def load_existing_logs(
    primitives: Iterable[str],
    save_dir: Union[str, Path],
) -> Dict[str, str]:
    logs: Dict[str, str] = {}
    for primitive in primitives:
        csv_path = _csv_path_for_primitive(save_dir, primitive)
        if csv_path.exists():
            logs[primitive] = str(csv_path)
            print(f"[INFO] Using existing traces for primitive {primitive}")
        else:
            print(
                f"[ERROR] No existing traces found for primitive {primitive} at {csv_path}"
            )
    return logs


def _worker_generate_scenario(job: Mapping[str, object]) -> Tuple[str, str]:
    """Subprocess worker for generate_graph_scenarios — compiles a single
    self-contained `scenario X():` block directly via scenarioFromFile and
    runs `n` traces of it. Bypasses the wrapper-builder that
    generate_graph_traces uses, which is incompatible with leaf scenarios
    that create their own ego in setup (the wrapper's `do X()` runs setup
    at sim time, but MetaDrive needs >=1 Scenic object at scene creation;
    also the wrapper's idle ego shadows the leaf's real ego in
    `_trajectory_rows`)."""
    scenic = _load_scenic()

    scenic_file = str(job["scenic_file"])
    scenario_name = str(job["scenario_name"])
    save_dir = Path(str(job["save_dir"]))
    n = int(job["n"])
    max_steps = job.get("max_steps")
    mode2d = bool(job.get("mode2d", True))
    model = job.get("model") or DEFAULT_SCENIC_MODEL
    max_iterations = int(job.get("max_iterations", 2000))
    position = int(job.get("position", 0))

    save_dir = save_dir / scenario_name
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "traces.csv"

    sc = scenic.scenarioFromFile(
        scenic_file,
        scenario=scenario_name,
        mode2D=mode2d,
        model=model,
    )
    sim = sc.getSimulator()

    try:
        from tqdm import tqdm
        bar = tqdm(total=n, desc=f"{scenario_name:<18s}", unit="trace",
                   position=position, leave=True)
    except ImportError:
        bar = None

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trace_id", "step", "x", "y", "heading", "speed",
                        "action", "reward", "label"],
        )
        writer.writeheader()
        trace_id = 0
        attempts = 0
        max_attempts = max(1000, n * 20)
        while trace_id < n and attempts < max_attempts:
            attempts += 1
            try:
                scene, _ = sc.generate(maxIterations=max_iterations,
                                       verbosity=0)
                simulation = sim.simulate(scene, maxSteps=max_steps,
                                          verbosity=0, maxIterations=1)
            except Exception as exc:
                if bar is not None:
                    bar.write(f"[{scenario_name}] attempt {attempts} failed: {exc}")
                continue
            if simulation is None:
                continue
            for row in _trajectory_rows(simulation, trace_id):
                writer.writerow(row)
            f.flush()
            if hasattr(simulation, "destroy"):
                try:
                    simulation.destroy()
                except Exception:
                    pass
            trace_id += 1
            if bar is not None:
                bar.update(1)

    if bar is not None:
        bar.close()
    return scenario_name, str(csv_path)


def generate_graph_scenarios(
    source: Union[str, Path],
    primitives: Sequence[str],
    *,
    n: Union[int, Mapping[str, int]] = 30,
    save_dir: Union[str, Path] = "storage/graph_scenarios",
    max_steps: Union[int, Mapping[str, int], None] = None,
    model: Optional[str] = None,
    backend: Optional[str] = None,
    mode2d: bool = True,
    max_iterations: int = 2000,
) -> Dict[str, str]:
    """Per-primitive trace generation for SELF-CONTAINED scenario primitives.

    Companion to `generate_graph_traces` for the case where each leaf primitive
    is its own `scenario X():` block whose setup creates its own ego (rather
    than a behavior attached to a Main-owned ego). Each primitive is compiled
    directly via `scenic.scenarioFromFile(scenario=name)` and run in its own
    subprocess; one CSV per primitive at `{save_dir}/{name}/traces.csv`.

    Use this when `generate_graph_traces` would build a wrapper of the form
    `GraphTraceEntry_X(): setup: ego = new Car; compose: do X()` — that
    pattern double-egos with self-contained leaf scenarios (the wrapper's
    idle ego shadows the leaf's real ego in `_trajectory_rows`), and the
    leaf's setup-at-sim-time fails MetaDrive's "requires >=1 Scenic object
    at scene creation" check.

    Parameters
    ----------
    source : path to the Scenic file declaring the leaf scenarios.
    primitives : list of leaf scenario names to generate. Caller is expected
        to have parsed `Main` (e.g. via `parse_scenic_spec`) to determine
        which leaves are referenced. Pass a single name to generate one
        scenario (e.g. a monolithic counterpart).
    n : int OR `{scenario_name: int}` — traces per leaf. Use a dict for
        per-leaf overrides; an int applies to all.
    save_dir : root output directory. Per-leaf CSVs go to
        `{save_dir}/{scenario_name}/traces.csv`.
    max_steps : int OR `{scenario_name: int}` OR None — sim ticks per trace.
        Use a dict when leaves need different lengths (e.g. some have a
        prewarm prefix that gets trimmed afterward and need extra raw ticks
        to compensate). None defers to the simulator default.
    model, backend : Scenic model / backend name. If both unset, the
        backend is sniffed from the source's `model ...` directive.
    mode2d, max_iterations : passed through to `scenarioFromFile` /
        `scenario.generate`.

    Returns `{scenario_name: csv_path}`. Each worker shows its own tqdm bar
    pinned to its own line via `position=index`.
    """
    source_text = (
        Path(source).read_text(encoding="utf-8") if Path(source).exists() else ""
    )
    _backend_name, scenic_model = resolve_backend(backend, model, source_text)

    primitives = list(primitives)
    if not primitives:
        return {}

    def _resolve(spec, name):
        if isinstance(spec, Mapping):
            return spec.get(name)
        return spec

    args_list = [
        {
            "scenic_file": str(source),
            "scenario_name": name,
            "save_dir": str(save_dir),
            "n": _resolve(n, name),
            "max_steps": _resolve(max_steps, name),
            "mode2d": mode2d,
            "model": scenic_model,
            "max_iterations": max_iterations,
            "position": idx,
        }
        for idx, name in enumerate(primitives)
    ]

    with mp.Pool(processes=len(args_list)) as pool:
        results = pool.map(_worker_generate_scenario, args_list)

    # tqdm bars leave the cursor below the last bar; print a newline so any
    # subsequent prints don't overwrite the bottom bar.
    print()
    return dict(results)


def generate_graph_traces(
    source: Union[str, Path],
    *,
    time_budget: Union[int, float] = float("inf"),
    n: Optional[int] = None,
    save_dir: Union[str, Path] = "storage/graph_traces",
    reuse_traces: bool = False,
    mode2d: bool = True,
    model: Optional[str] = None,
    backend: Optional[str] = None,
    max_iterations: int = 2000,
    max_steps: Optional[int] = None,
) -> Dict[str, str]:
    jobs = build_trace_jobs(
        source=source,
        save_dir=save_dir,
        n=n,
        mode2d=mode2d,
        model=model,
        backend=backend,
        max_iterations=max_iterations,
        max_steps=max_steps,
    )

    if not jobs:
        print("No executable primitive jobs were discovered in the enriched graph.")
        return {}

    if reuse_traces:
        return load_existing_logs((str(job["primitive"]) for job in jobs), save_dir)

    return generate_traces_parallel(
        jobs=jobs,
        time_budget=time_budget,
        reuse_traces=False,
    )


def test_scenario(
    input_scenario: Union[str, Sequence[str]],
    source: Union[str, Path],
    is_compositional: bool,
    time_budget: Union[int, float],
    n: Optional[int],
    save_dir: Union[str, Path],
    *,
    ground_truth: bool = False,
    confidence_level: Optional[float] = None,
    error_bound: Optional[float] = None,
    reuse_traces: bool = False,
    delta: float = 0.05,
    mode2d: bool = True,
    model: Optional[str] = None,
    max_iterations: int = 2000,
    max_steps: Optional[int] = None,
) -> None:
    if ground_truth:
        if confidence_level is None or error_bound is None:
            raise ValueError(
                "Ground truth mode requires --confidence_level and --error_bound"
            )
        n = compute_hoeffding_samples(confidence_level, error_bound)
        time_budget = float("inf")

    primitives = parse_scenario(input_scenario)
    if not is_compositional:
        primitives = ["".join(primitives)]

    logs = generate_graph_traces(
        source=source,
        time_budget=time_budget,
        n=n,
        save_dir=save_dir,
        reuse_traces=reuse_traces,
        mode2d=mode2d,
        model=model,
        max_iterations=max_iterations,
        max_steps=max_steps,
    )

    run_monolithic_smc(logs)

    if is_compositional:
        run_smc_compositional(
            scenarios=[primitives],
            time_budget=time_budget,
            logs=logs,
            delta=delta,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Scenic traces from the enriched composition graph."
    )
    parser.add_argument("source", type=str, help="Composite Scenic file to analyze")
    parser.add_argument(
        "--time_budget",
        type=float,
        default=float("inf"),
        help="Time budget in seconds (default: unlimited)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of traces to generate per primitive",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="storage/graph_traces",
        help="Directory to save trace CSVs",
    )
    parser.add_argument(
        "--reuse_traces",
        action="store_true",
        help="Use existing traces from save_dir without generating new ones",
    )
    parser.add_argument(
        "--mode2d",
        action="store_true",
        help="Compile Scenic scenarios in 2D mode",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional Scenic model override passed to scenarioFromFile",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=list(BACKENDS),
        help=(
            "Simulator backend name (overrides --model). "
            "Run with --list-backends to see availability."
        ),
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="Print available simulator backends and exit.",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=2000,
        help="Maximum rejection-sampling iterations per scene generation",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum simulation steps per trace",
    )

    args = parser.parse_args()

    if args.list_backends:
        print(list_backends())
        raise SystemExit(0)

    mp.set_start_method("spawn")

    logs = generate_graph_traces(
        source=args.source,
        time_budget=args.time_budget,
        n=args.n,
        save_dir=args.save_dir,
        reuse_traces=args.reuse_traces,
        mode2d=args.mode2d,
        model=args.model,
        backend=args.backend,
        max_iterations=args.max_iterations,
        max_steps=args.max_steps,
    )

    print("\nGenerated logs:")
    for primitive, csv_path in sorted(logs.items()):
        print(f"  {primitive}: {csv_path}")


if __name__ == "__main__":
    import json

    g = build_enriched_graph(
        "tests/scenic/scenic_tests/cases_realistic/tollgate_test_metadrive/main.scenic"
    )
    print(json.dumps(g, indent=2, default=str))
