"""Scenic-backend budget sweep via continuous run + periodic checkpoints.

One simulation per method (compositional, monolithic) runs to ``--max_budget``;
every ``--snapshot_every`` seconds the script records each primitive's
trace count. After the run, for every snapshot ``(elapsed, counts)`` the
CSV is filtered to "first N traces per primitive", relabeled with the DFA
spec, and analyzed → one ``(rho, eps)`` point per checkpoint per method.

The result is a dense convergence curve from ``snapshot_every`` to
``max_budget`` for the wall-clock cost of one long simulation per method,
not N separate runs. Each MetaDrive subprocess pays Scenic startup
(~15–30s) once — checkpoints earlier than that record ``no_traces`` (which
*is* the data: it shows when each method "wakes up").

Outputs:
  - ``results.csv`` with one row per (method, checkpoint)
  - ``plots/eps_vs_budget.png``       — log-log Hoeffding CI half-width
  - ``plots/rho_vs_budget.png``       — rho ± eps convergence
  - ``plots/throughput.png``          — completed traces vs. checkpoint
  - ``plots/speedup_vs_budget.png``   — eps_mono / eps_comp at matched T

The DFA spec (``CONFIG["spec_module"]``) exposes ``make_spec()`` and works for
both markovian and non-markovian DFAs — the compositional engine tracks DFA
state across primitive segments, the monolithic side evaluates the whole
trace (see the note in ``default_spec`` on absolute-step labeling).

There is no CLI. The ``EXPERIMENTS`` list at the top of this file defines the
configs (each with its own scenario, monolithic counterpart, and DFA spec); the
script runs them one after another into ``storage/budget_sweep/<name>/``. Two
module-level toggles, ``USE_WANDB`` and ``REUSE_RESULTS``, are the only switches.

    python compare_budget_sweep.py          # run every experiment in EXPERIMENTS
"""

import csv
import importlib.util
import multiprocessing as mp
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from verifai.compositional_analysis import (
    CompositionalAnalysisEngine,
    ScenarioBase,
    relabel_traces,
)
from verifai.generate_graph_traces import (
    _worker_generate_scenario,
    default_mode2d_for_backend,
    resolve_backend,
)
from verifai.monitor import automaton_specification
from verifai.scenic_composition_analysis import (
    analyze_scenic_composition,
    build_partner_format,
)
from verifai.scenic_parser import get_primitives, parse_scenic_spec


METHOD_MONO = "monolithic"
METHOD_COMP = "compositional"

FIELDS = [
    "scenic_file",
    "method",
    "budget",
    "elapsed",
    "rho",
    "eps",
    "n_traces",
    "n_full_traces",
    "n_traces_breakdown",
    "graph_build_s",
    "status",
    "note",
]

WARMUP_STEPS = 25
DEFAULT_MAX_SPEED = 5.5
HOEFFDING_DELTA = 0.05

REPO_ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
#
# No CLI. The sweep runs every entry in EXPERIMENTS, one after another, writing
# each to storage/budget_sweep/<name>/. To add or change an experiment, edit the
# list below. Each entry is (name, config); config["spec"] is a zero-arg factory
# returning an automaton_specification (so each experiment picks its own DFA).
#
# The two module-level toggles below are the only run-mode switches.

USE_WANDB = True  # push plots + results CSV to Weights & Biases
REUSE_RESULTS = False  # skip simulation; just replot from existing results.csv
WANDB_PROJECT = "verifai-compositional-analysis"

SCENIC_DIR = (
    REPO_ROOT
    / "examples/compositional_analysis/dfa_tests/e2e_4way_example"
    / "4_way_intersection_scenic"
)
SPEC_AT_MOST_ONE_BRAKE = (
    Path(__file__).resolve().parent / "specs" / "at_most_one_brake.py"
)


def _safety_spec(max_speed):
    """Factory for the tollgate safety DFA (speed never exceeds max_speed)."""
    return lambda: default_spec(max_speed=max_speed)


# Defaults shared by every experiment; each entry overrides what it needs.
_BASE = {
    "max_budget": 1800.0,  # seconds per method (mono + comp run in turn)
    "snapshot_every": 30.0,  # checkpoint cadence
    "features": ["speed"],
    "center_feat_idx": [],
    "max_steps_overrides": {},  # {primitive: max_steps}
    "prewarm_trim": {},  # {primitive: rows to drop from each trace}
}

# Shared config for the N=5 wander-over-scenarios setup; each experiment below
# reuses it under a different DFA spec.
# (from test_4way_intersection_wander_scenarios.py)
_WANDER_SCEN = {
    **_BASE,
    "scenic_file": str(SCENIC_DIR / "wander_scenarios.scenic"),
    "composite_name": "Main",
    "monolithic_name": "MonolithicWander",
    "max_steps_primitive": 75,
    "max_steps_mono": 200,
    "prewarm_trim": {
        p: 35
        for p in (
            "BrakeScenario",
            "GoStraightScenario",
            "TurnLeftScenario",
            "TurnRightScenario",
        )
    },
}

EXPERIMENTS = [
    # --- wander_scenarios under several specs (1 markovian + 3 non-markovian) ---
    (
        "wander_at_most_one_brake",
        {**_WANDER_SCEN, "spec": lambda: load_spec(str(SPEC_AT_MOST_ONE_BRAKE))},
    ),  # 5-state
    (
        "wander_at_most_two_brake",
        {**_WANDER_SCEN, "spec": lambda: spec_at_most_k_brake(2)},
    ),  # 7-state
    (
        "wander_k_consec_slow_K2",
        {**_WANDER_SCEN, "spec": lambda: spec_k_consec_slow(2)},
    ),  # 4-state
    (
        "wander_k_consec_fast_K10",
        {**_WANDER_SCEN, "spec": lambda: spec_k_consec_fast(10)},
    ),  # 12-state
    # N=5 wander over bare behaviors, safety DFA. (test_4way_intersection_wander.py)
    (
        "composed_wander",
        {
            **_BASE,
            "scenic_file": str(SCENIC_DIR / "composed_wander.scenic"),
            "composite_name": "Main",
            "monolithic_name": "MonolithicWander",
            "spec": _safety_spec(5.5),
            "max_steps_primitive": 75,
            "max_steps_mono": 375,
            "prewarm_trim": {
                p: 35 for p in ("Brake", "GoStraight", "TurnLeft", "TurnRight")
            },
        },
    ),
    # 2-step intersection composition, safety DFA. Sub2* carry a cruise prewarm
    # and need extra raw ticks. (test_4way_intersection_scenarios.py)
    (
        "composed_scenarios",
        {
            **_BASE,
            "scenic_file": str(SCENIC_DIR / "composed_scenarios.scenic"),
            "composite_name": "Main",
            "monolithic_name": "MonolithicMain",
            "spec": _safety_spec(5.5),
            "max_steps_primitive": 85,
            "max_steps_mono": 170,
            "max_steps_overrides": {
                p: 110 for p in ("Subscenario2L", "Subscenario2R", "Subscenario2S")
            },
            "prewarm_trim": {
                p: 25 for p in ("Subscenario2L", "Subscenario2R", "Subscenario2S")
            },
        },
    ),
    # 10-step traversal (approach + turn) chain, safety DFA.
    # (test_4way_intersection_traversal_wander.py)
    (
        "traversal_wander",
        {
            **_BASE,
            "scenic_file": str(SCENIC_DIR / "traversal_wander.scenic"),
            "composite_name": "Main",
            "monolithic_name": "Monolithic5",
            "spec": _safety_spec(7.5),
            "max_steps_primitive": 100,
            "max_steps_mono": 1000,
        },
    ),
]


def load_env_file(env_path):
    """Populate ``os.environ`` from a simple KEY=VALUE ``.env`` file.

    Manual parser so we don't depend on python-dotenv. Existing environment
    variables are never overwritten. Missing file is a no-op.
    """
    env_path = Path(env_path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


# ----------------------------------------------------------------------------
# CSV helpers
# ----------------------------------------------------------------------------


def _count_trace_ids(csv_path):
    if not os.path.exists(csv_path):
        return 0
    seen = set()
    with open(csv_path) as f:
        next(f, None)  # header
        for line in f:
            parts = line.split(",", 1)
            if parts and parts[0]:
                seen.add(parts[0])
    return len(seen)


def _trim_partial(csv_path, keep):
    """Drop rows with trace_id >= keep (partial trace from a hard stop)."""
    if not os.path.exists(csv_path) or keep <= 0:
        return
    with open(csv_path) as f:
        lines = f.readlines()
    if len(lines) <= 1:
        return
    with open(csv_path, "w") as f:
        f.write(lines[0])
        for line in lines[1:]:
            parts = line.split(",", 1)
            if not parts:
                continue
            try:
                tid = int(parts[0])
            except ValueError:
                continue
            if tid < keep:
                f.write(line)


def _filter_csv_first_n_traces(src, dst, n):
    """Copy src → dst, keeping only rows with trace_id < n."""
    with open(src) as fin, open(dst, "w") as fout:
        header = fin.readline()
        fout.write(header)
        for line in fin:
            parts = line.split(",", 1)
            if not parts or not parts[0]:
                continue
            try:
                tid = int(parts[0])
            except ValueError:
                continue
            if tid < n:
                fout.write(line)


def trim_prewarm(csv_path, n):
    """Drop the first ``n`` rows of each trace in-place and renumber step
    to ``0..M-1``. Idempotent on the first call only (assumes the CSV
    still has its raw prewarm prefix). Lifted from
    ``test_4way_intersection_wander.py::trim_prewarm`` so the wall-clock
    sweep tool can produce post-trim primitive CSVs that match the
    handoff distribution the monolithic counterpart sees."""
    df = pd.read_csv(csv_path).sort_values(["trace_id", "step"])
    trimmed = []
    for tid, grp in df.groupby("trace_id"):
        kept = grp.iloc[n:].copy()
        kept["step"] = range(len(kept))
        trimmed.append(kept)
    if trimmed:
        pd.concat(trimmed, ignore_index=True).to_csv(csv_path, index=False)


# ----------------------------------------------------------------------------
# Simulation runner with periodic snapshots
# ----------------------------------------------------------------------------


def simulate_with_snapshots(jobs, max_budget, snapshot_every, save_dir):
    """Spawn one ``_worker_generate_scenario`` per job; hard-stop at
    ``max_budget``; every ``snapshot_every`` seconds record
    ``(elapsed, {primitive: trace_count})``.

    Returns ``(logs, timeline)`` where:
      - logs    : ``{name: csv_path}`` for primitives with >=1 complete trace
      - timeline: ``[(elapsed_s, {name: count}), ...]`` snapshots + final
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    processes = []
    for idx, job in enumerate(jobs):
        full_job = {
            "scenic_file": str(job["scenic_file"]),
            "scenario_name": str(job["scenario_name"]),
            "save_dir": str(save_dir),
            "n": int(job.get("n") or 10**9),
            "max_steps": job.get("max_steps"),
            "mode2d": job.get("mode2d", True),
            "model": job.get("model"),
            "max_iterations": int(job.get("max_iterations", 2000)),
            "position": idx,
        }
        scen_dir = save_dir / full_job["scenario_name"]
        scen_dir.mkdir(parents=True, exist_ok=True)
        csv_path = scen_dir / "traces.csv"
        if csv_path.exists():
            csv_path.unlink()

        p = mp.Process(target=_worker_generate_scenario, args=(full_job,))
        p.start()
        processes.append((full_job["scenario_name"], p, csv_path))

    def _all_counts():
        return {n: _count_trace_ids(str(c)) for n, _, c in processes}

    timeline = []
    start = time.time()
    next_snapshot = snapshot_every
    hard_stopped = False
    counts_at_stop = {}

    poll_sleep = max(0.25, min(snapshot_every / 4.0, 2.0))

    while True:
        elapsed = time.time() - start

        if elapsed >= next_snapshot:
            timeline.append((elapsed, _all_counts()))
            next_snapshot += snapshot_every

        if elapsed >= max_budget:
            counts_at_stop = _all_counts()
            for name, p, _ in processes:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                        p.kill()
                        p.join()
            # Avoid a duplicate final checkpoint when this iteration already
            # recorded a snapshot at the same elapsed (snapshot boundary
            # coinciding with max_budget). counts are effectively identical.
            if not timeline or (elapsed - timeline[-1][0]) > poll_sleep:
                timeline.append((elapsed, counts_at_stop))
            hard_stopped = True
            print(f"[HARD STOP] {elapsed:.1f}s >= max_budget {max_budget:.1f}s")
            break

        if all(not p.is_alive() for _, p, _ in processes):
            final_counts = _all_counts()
            timeline.append((elapsed, final_counts))
            print(f"[finished naturally] at {elapsed:.1f}s")
            break

        time.sleep(poll_sleep)

    logs = {}
    for name, _p, csv_path in processes:
        if not csv_path.exists():
            continue
        if hard_stopped:
            keep = counts_at_stop.get(name, 0)
            _trim_partial(str(csv_path), keep)
            n = keep
        else:
            n = _count_trace_ids(str(csv_path))
        if n > 0:
            logs[name] = str(csv_path)

    return logs, timeline


# ----------------------------------------------------------------------------
# Default DFA spec (matches e2e_4way test)
# ----------------------------------------------------------------------------


def default_spec(max_speed=DEFAULT_MAX_SPEED, warmup_steps=WARMUP_STEPS):
    # Caveat for both markovian and non-markovian specs: the compositional
    # method evaluates each primitive segment independently with its own
    # ``step`` restarting at 0, while the monolithic method sees one
    # continuous trace. Any labeling logic keyed on the *absolute* step
    # index (like ``warmup_steps`` below) therefore fires once per segment
    # compositionally but once overall monolithically, and the two methods
    # will measure different rho. Specs intended to compare both methods
    # should label on state/feature values, not absolute step position.
    def transition(state, sym):
        if state == "bad":
            return "bad"
        return "bad" if sym == "high" else "ok"

    def label_row(row):
        if row["step"] < warmup_steps:
            return "low"
        return "high" if row["speed"] > max_speed else "low"

    return automaton_specification(
        start="ok",
        inputs={"high", "low"},
        transition=transition,
        label=lambda s: s == "ok",
        labeling_function=label_row,
    )


def load_spec(spec_module_path):
    spec_path = Path(spec_module_path).resolve()
    if not spec_path.exists():
        raise FileNotFoundError(f"spec module not found: {spec_path}")
    spec = importlib.util.spec_from_file_location("user_spec", str(spec_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "make_spec"):
        raise AttributeError(f"{spec_path} must define make_spec()")
    return mod.make_spec()


# Additional non-markovian spec factories, ported verbatim from
# test_4way_intersection_wander_scenarios.py. All read the `speed` feature with
# a 5-step warmup and have one absorbing reject state.
_SPEC_WARMUP = 5
_STOP_THRESHOLD = 0.5  # m/s, "slow" boundary
_FAST_THRESHOLD = 1.5  # m/s, "fast" boundary for k_consec_fast


def spec_k_consec_slow(K, threshold=_STOP_THRESHOLD):
    """Never more than K consecutive slow steps post-warmup."""

    def transition(state, sym):
        if state == "ok_run":
            return "slow_1" if sym == "slow" else "ok_run"
        if state == "bad":
            return "bad"
        idx = int(state.split("_")[1])
        if sym == "fast":
            return "ok_run"
        return "bad" if idx >= K else f"slow_{idx + 1}"

    def label_row(row):
        if row["step"] < _SPEC_WARMUP:
            return "fast"
        return "slow" if row["speed"] < threshold else "fast"

    return automaton_specification(
        start="ok_run",
        inputs={"slow", "fast"},
        transition=transition,
        label=lambda s: s != "bad",
        labeling_function=label_row,
    )


def spec_k_consec_fast(K, threshold=_FAST_THRESHOLD):
    """Never more than K consecutive fast steps post-warmup (mirror of slow)."""

    def transition(state, sym):
        if state == "ok":
            return "fast_1" if sym == "fast" else "ok"
        if state == "bad":
            return "bad"
        idx = int(state.split("_")[1])
        if sym == "slow":
            return "ok"
        return "bad" if idx >= K else f"fast_{idx + 1}"

    def label_row(row):
        if row["step"] < _SPEC_WARMUP:
            return "slow"
        return "fast" if row["speed"] >= threshold else "slow"

    return automaton_specification(
        start="ok",
        inputs={"slow", "fast"},
        transition=transition,
        label=lambda s: s != "bad",
        labeling_function=label_row,
    )


def spec_at_most_k_brake(K, threshold=_STOP_THRESHOLD):
    """At most K debounced slow->fast brake episodes. Same DFA family as
    specs/at_most_one_brake.py, generalized to arbitrary K."""

    def transition(state, sym):
        if state == "violated":
            return "violated"
        if state.endswith("_in_slow"):
            n = int(state[1 : state.index("_")])
            if sym == "slow":
                return state
            n += 1
            return "violated" if n > K else f"q{n}"
        n = int(state[1:])
        return f"q{n}_in_slow" if sym == "slow" else state

    def label_row(row):
        if row["step"] < _SPEC_WARMUP:
            return "fast"
        return "slow" if row["speed"] < threshold else "fast"

    return automaton_specification(
        start="q0",
        inputs={"slow", "fast"},
        transition=transition,
        label=lambda s: s != "violated",
        labeling_function=label_row,
    )


def hoeffding_eps(n, delta=HOEFFDING_DELTA):
    return float(np.sqrt(np.log(2 / delta) / (2 * max(n, 1))))


def composition_length(paths) -> int:
    """Number of sequential composition steps in a path.

    Accepts either the wrapped ``[(prob, composition), ...]`` form or a flat
    ``List[CompositionStep]``. Returns the max step count across paths — i.e.
    how many primitive segments make up one full monolithic episode. Used to
    (a) put compositional throughput on a full-episode-equivalent axis and
    (b) decide whether KDE (which needs >=2 traces) is even involved.
    """
    if not paths:
        return 1
    first = paths[0]
    is_wrapped = (
        isinstance(first, tuple)
        and len(first) == 2
        and isinstance(first[0], (int, float))
    )
    comps = [c for _, c in paths] if is_wrapped else [list(paths)]
    return max((len(c) for c in comps), default=1)


# ----------------------------------------------------------------------------
# Per-checkpoint analysis
# ----------------------------------------------------------------------------


def analyze_compositional_at(
    elapsed,
    counts,
    final_logs,
    paths,
    primitives,
    spec,
    features,
    center_feat_idx,
    temp_dir,
):
    breakdown = ";".join(f"{k}={counts.get(k, 0)}" for k in primitives)
    # KDE only enters for multi-step compositions; a single-step composition is
    # a direct DFA-label estimate that needs just 1 trace (matching monolithic).
    # Requiring >=2 everywhere would blank out single-primitive compositions at
    # early budgets while monolithic already reports a rho.
    min_traces = 2 if composition_length(paths) > 1 else 1
    if any(counts.get(p, 0) < min_traces for p in primitives):
        return {
            "rho": None,
            "eps": None,
            "n_traces": sum(counts.get(p, 0) for p in primitives),
            "n_traces_breakdown": breakdown,
            "elapsed": elapsed,
            "status": "insufficient_data",
            "note": f"need >={min_traces} trace(s) per primitive",
        }
    if any(p not in final_logs for p in primitives):
        return {
            "rho": None,
            "eps": None,
            "n_traces": sum(counts.get(p, 0) for p in primitives),
            "n_traces_breakdown": breakdown,
            "elapsed": elapsed,
            "status": "missing_primitives",
            "note": f"final_logs missing: {sorted(set(primitives) - set(final_logs))}",
        }

    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_logs = {}
    for name in primitives:
        n = counts[name]
        dst = temp_dir / f"{name}_t{int(elapsed)}_n{n}.csv"
        _filter_csv_first_n_traces(final_logs[name], str(dst), n)
        temp_logs[name] = str(dst)

    try:
        for name in temp_logs:
            relabel_traces(temp_logs[name], spec)
        base = ScenarioBase(temp_logs)
        engine = CompositionalAnalysisEngine(base)
        rho, eps = engine.check_with_dfa_scenic(
            paths,
            spec,
            features=features,
            center_feat_idx=center_feat_idx,
        )
    except Exception as e:
        result = {
            "rho": None,
            "eps": None,
            "n_traces": sum(counts.values()),
            "n_traces_breakdown": breakdown,
            "elapsed": elapsed,
            "status": "analysis_error",
            "note": repr(e),
        }
    else:
        result = {
            "rho": float(rho),
            "eps": float(eps),
            "n_traces": sum(counts.get(p, 0) for p in primitives),
            "n_traces_breakdown": breakdown,
            "elapsed": elapsed,
            "status": "ok",
            "note": "",
        }
    finally:
        for p in temp_logs.values():
            try:
                Path(p).unlink()
            except Exception:
                pass

    return result


def analyze_monolithic_at(elapsed, count, final_log, mono_name, spec, temp_dir):
    if count < 1:
        return {
            "rho": None,
            "eps": None,
            "n_traces": 0,
            "n_traces_breakdown": f"{mono_name}=0",
            "elapsed": elapsed,
            "status": "no_traces",
            "note": "no traces yet at this checkpoint",
        }
    if final_log is None:
        return {
            "rho": None,
            "eps": None,
            "n_traces": count,
            "n_traces_breakdown": f"{mono_name}={count}",
            "elapsed": elapsed,
            "status": "no_traces",
            "note": "no monolithic CSV path",
        }

    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    dst = temp_dir / f"{mono_name}_t{int(elapsed)}_n{count}.csv"
    _filter_csv_first_n_traces(final_log, str(dst), count)

    try:
        rho = relabel_traces(str(dst), spec)
    except Exception as e:
        result = {
            "rho": None,
            "eps": None,
            "n_traces": count,
            "n_traces_breakdown": f"{mono_name}={count}",
            "elapsed": elapsed,
            "status": "analysis_error",
            "note": repr(e),
        }
    else:
        result = {
            "rho": float(rho),
            "eps": hoeffding_eps(count),
            "n_traces": count,
            "n_traces_breakdown": f"{mono_name}={count}",
            "elapsed": elapsed,
            "status": "ok",
            "note": "",
        }
    finally:
        try:
            dst.unlink()
        except Exception:
            pass

    return result


# ----------------------------------------------------------------------------
# Sweep orchestrator
# ----------------------------------------------------------------------------


def sweep_snapshot(
    scenic_file,
    monolithic_name,
    paths,
    primitives,
    spec,
    max_budget,
    snapshot_every,
    save_dir,
    features,
    center_feat_idx,
    max_steps_mono,
    max_steps_primitive,
    max_steps_overrides,
    csv_path,
    graph_build_s,
    scenic_model,
    mode2d,
    prewarm_trim_overrides=None,
):
    records = []
    # One full monolithic episode == this many primitive segments; used to put
    # compositional throughput on a full-episode-equivalent axis.
    n_steps = composition_length(paths)
    temp_dir = Path(save_dir) / "_temp_filtered"
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        # ---- Compositional ----
        print(
            f"\n=== Compositional: max_budget={max_budget:.0f}s, "
            f"snapshot_every={snapshot_every:.0f}s ==="
        )
        comp_dir = Path(save_dir) / METHOD_COMP
        comp_jobs = [
            {
                "scenic_file": scenic_file,
                "scenario_name": name,
                "max_steps": max_steps_overrides.get(name, max_steps_primitive),
                "model": scenic_model,
                "mode2d": mode2d,
            }
            for name in primitives
        ]
        t0 = time.time()
        logs_comp, timeline_comp = simulate_with_snapshots(
            comp_jobs, max_budget, snapshot_every, comp_dir
        )
        sim_comp_s = time.time() - t0
        print(
            f"  simulation: {sim_comp_s:.1f}s, "
            f"{len(timeline_comp)} snapshots, "
            f"primitives with traces: {sorted(logs_comp)}"
        )

        # Apply prewarm trim to each primitive CSV before any checkpoint
        # analysis. Trim is per-row (drops first N rows of each trace and
        # renumbers ``step``); trace_id count is preserved, so the timeline
        # counts recorded above remain valid.
        if prewarm_trim_overrides:
            for primitive, csv_path_p in logs_comp.items():
                n_trim = prewarm_trim_overrides.get(primitive, 0)
                if n_trim and n_trim > 0:
                    trim_prewarm(csv_path_p, n_trim)
                    print(f"  trimmed {n_trim} prewarm rows from {primitive}")

        print("  analyzing checkpoints …")
        ok_comp = 0
        for elapsed, counts in timeline_comp:
            r = analyze_compositional_at(
                elapsed,
                counts,
                logs_comp,
                paths,
                primitives,
                spec,
                features,
                center_feat_idx,
                temp_dir,
            )
            rec = {
                "scenic_file": str(scenic_file),
                "method": METHOD_COMP,
                # Charge the one-time graph-build cost to the compositional
                # budget axis — it's a compositional-only setup the monolithic
                # method never pays.
                "budget": elapsed + graph_build_s,
                "graph_build_s": graph_build_s,
                **r,
                # Full-episode-equivalent count: comp segments / steps-per-episode.
                "n_full_traces": r["n_traces"] / n_steps if n_steps else r["n_traces"],
            }
            writer.writerow(rec)
            f.flush()
            records.append(rec)
            if rec["status"] == "ok":
                ok_comp += 1
        print(f"  → {ok_comp}/{len(timeline_comp)} checkpoints produced (rho, eps)")

        # ---- Monolithic ----
        print(
            f"\n=== Monolithic: max_budget={max_budget:.0f}s, "
            f"snapshot_every={snapshot_every:.0f}s ==="
        )
        mono_dir = Path(save_dir) / METHOD_MONO
        mono_jobs = [
            {
                "scenic_file": scenic_file,
                "scenario_name": monolithic_name,
                "max_steps": max_steps_mono,
                "model": scenic_model,
                "mode2d": mode2d,
            }
        ]
        t0 = time.time()
        logs_mono, timeline_mono = simulate_with_snapshots(
            mono_jobs, max_budget, snapshot_every, mono_dir
        )
        sim_mono_s = time.time() - t0
        print(
            f"  simulation: {sim_mono_s:.1f}s, "
            f"{len(timeline_mono)} snapshots, "
            f"mono CSV: {logs_mono.get(monolithic_name, 'NONE')}"
        )

        mono_log = logs_mono.get(monolithic_name)
        print("  analyzing checkpoints …")
        ok_mono = 0
        for elapsed, counts in timeline_mono:
            count = counts.get(monolithic_name, 0)
            r = analyze_monolithic_at(
                elapsed, count, mono_log, monolithic_name, spec, temp_dir
            )
            rec = {
                "scenic_file": str(scenic_file),
                "method": METHOD_MONO,
                "budget": elapsed,
                "graph_build_s": graph_build_s,
                **r,
                # A monolithic trace already is one full episode.
                "n_full_traces": r["n_traces"],
            }
            writer.writerow(rec)
            f.flush()
            records.append(rec)
            if rec["status"] == "ok":
                ok_mono += 1
        print(f"  → {ok_mono}/{len(timeline_mono)} checkpoints produced (rho, eps)")

    shutil.rmtree(temp_dir, ignore_errors=True)
    return records


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------


def _series(records, key):
    out = defaultdict(list)
    for r in records:
        v = r.get(key)
        if v is None:
            continue
        out[r["method"]].append((float(r["budget"]), float(v)))
    for m in out:
        out[m].sort()
    return out


def plot_eps_vs_budget(records, out_path):
    series = _series(records, "eps")
    if not series:
        print("[plot eps] no points; skipping")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, pts in series.items():
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=".", linewidth=1.2, label=m)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time budget (s)")
    ax.set_ylabel("eps (Hoeffding CI half-width)")
    ax.set_title("Uncertainty vs. budget")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_rho_vs_budget(records, out_path):
    by = defaultdict(list)
    for r in records:
        if r.get("rho") is None or r.get("eps") is None:
            continue
        by[r["method"]].append((float(r["budget"]), float(r["rho"]), float(r["eps"])))
    for m in by:
        by[m].sort()
    if not by:
        print("[plot rho] no points; skipping")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, pts in by.items():
        xs = [p[0] for p in pts]
        rho = np.array([p[1] for p in pts])
        eps = np.array([p[2] for p in pts])
        ax.plot(xs, rho, marker=".", linewidth=1.2, label=m)
        ax.fill_between(xs, rho - eps, rho + eps, alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("time budget (s)")
    ax.set_ylabel("rho ± eps")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Estimate convergence")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_throughput(records, out_path):
    # n_full_traces normalizes compositional segment counts to full-episode
    # equivalents so the two methods share a unit (a comp "trace" is one
    # primitive segment; a mono "trace" is a full multi-segment episode).
    series = _series(records, "n_full_traces")
    if not series:
        print("[plot throughput] no points; skipping")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, pts in series.items():
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=".", linewidth=1.2, label=m)
    ax.set_xscale("log")
    ax.set_yscale("symlog")
    ax.set_xlabel("time budget (s)")
    ax.set_ylabel("# full-episode-equivalent traces")
    ax.set_title("Trace throughput vs. budget")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_wallclock_combo(records, out_path):
    """Side-by-side figure for the paper (fig:wallclock):
    left panel is ``eps`` vs.~``T`` on log-log axes, right panel is
    ``rho`` with a shaded ``+/- eps`` band on semi-log-x axes. Both
    panels read from the same records as the individual plot functions.
    """
    eps_series = _series(records, "eps")

    rho_by = defaultdict(list)
    for r in records:
        if r.get("rho") is None or r.get("eps") is None:
            continue
        rho_by[r["method"]].append(
            (float(r["budget"]), float(r["rho"]), float(r["eps"]))
        )
    for m in rho_by:
        rho_by[m].sort()

    if not eps_series and not rho_by:
        print("[plot wallclock_combo] no points; skipping")
        return

    fig, (ax_eps, ax_rho) = plt.subplots(1, 2, figsize=(12, 4.5))

    for m, pts in eps_series.items():
        xs, ys = zip(*pts)
        ax_eps.plot(xs, ys, marker=".", linewidth=1.2, label=m)
    ax_eps.set_xscale("log")
    ax_eps.set_yscale("log")
    ax_eps.set_xlabel("time budget (s)")
    ax_eps.set_ylabel(r"$\hat\varepsilon$ (Hoeffding CI half-width)")
    ax_eps.set_title(r"Uncertainty vs. budget")
    ax_eps.grid(True, alpha=0.3)
    ax_eps.legend()

    for m, pts in rho_by.items():
        xs = [p[0] for p in pts]
        rho = np.array([p[1] for p in pts])
        eps = np.array([p[2] for p in pts])
        ax_rho.plot(xs, rho, marker=".", linewidth=1.2, label=m)
        ax_rho.fill_between(xs, rho - eps, rho + eps, alpha=0.2)
    ax_rho.set_xscale("log")
    ax_rho.set_xlabel("time budget (s)")
    ax_rho.set_ylabel(r"$\hat\rho \pm \hat\varepsilon$")
    ax_rho.set_ylim(0.0, 1.0)
    ax_rho.set_title(r"Estimate convergence")
    ax_rho.grid(True, alpha=0.3)
    ax_rho.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_speedup_vs_budget(records, out_path):
    """eps_mono / eps_comp at matched budget. >1 means compositional wins.

    Snapshot times are not exactly aligned between methods; bucket budgets
    to the nearest log-spaced bin and pair within bin.
    """
    rows = [r for r in records if r.get("eps") is not None and r["eps"] > 0]
    if not rows:
        print("[plot speedup] no points; skipping")
        return

    budgets = sorted({r["budget"] for r in rows})
    if len(budgets) < 2:
        print("[plot speedup] need >=2 budgets; skipping")
        return

    # Pair by closest budget within ±20% relative tolerance.
    mono = sorted(
        [r for r in rows if r["method"] == METHOD_MONO], key=lambda r: r["budget"]
    )
    comp = sorted(
        [r for r in rows if r["method"] == METHOD_COMP], key=lambda r: r["budget"]
    )
    pairs = []
    for rm in mono:
        # Find nearest comp by budget within 20% relative tol.
        candidates = [
            rc
            for rc in comp
            if abs(rc["budget"] - rm["budget"]) / max(rm["budget"], 1e-9) < 0.2
        ]
        if not candidates:
            continue
        rc = min(candidates, key=lambda c: abs(c["budget"] - rm["budget"]))
        T = 0.5 * (rm["budget"] + rc["budget"])
        pairs.append((T, rm["eps"] / rc["eps"]))

    if not pairs:
        print("[plot speedup] no matched (mono, comp) pairs; skipping")
        return

    xs, ys = zip(*pairs)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, ys, marker=".", linewidth=1.2, color="#3c5b8a")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("time budget (s)")
    ax.set_ylabel("eps_mono / eps_comp")
    ax.set_title("Compositional precision advantage at matched budget")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def load_records(csv_path):
    out = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in ("budget", "elapsed", "rho", "eps", "graph_build_s"):
                if row.get(k) in (None, "", "None"):
                    row[k] = None
                else:
                    try:
                        row[k] = float(row[k])
                    except ValueError:
                        row[k] = None
            row["n_traces"] = int(row.get("n_traces") or 0)
            # Older CSVs predate n_full_traces; fall back to the raw count so
            # the throughput plot still renders (unnormalized for those rows).
            nft = row.get("n_full_traces")
            if nft in (None, "", "None"):
                row["n_full_traces"] = float(row["n_traces"])
            else:
                try:
                    row["n_full_traces"] = float(nft)
                except ValueError:
                    row["n_full_traces"] = float(row["n_traces"])
            out.append(row)
    return out


PLOT_FILES = [
    ("wallclock", "wallclock.png"),
    ("eps_vs_budget", "eps_vs_budget.png"),
    ("rho_vs_budget", "rho_vs_budget.png"),
    ("throughput", "throughput.png"),
    ("speedup_vs_budget", "speedup_vs_budget.png"),
]


def render_plots(records, plots_dir):
    plot_eps_vs_budget(records, str(plots_dir / "eps_vs_budget.png"))
    plot_rho_vs_budget(records, str(plots_dir / "rho_vs_budget.png"))
    plot_throughput(records, str(plots_dir / "throughput.png"))
    plot_speedup_vs_budget(records, str(plots_dir / "speedup_vs_budget.png"))
    plot_wallclock_combo(records, str(plots_dir / "wallclock.png"))


def log_to_wandb(project, name, config, plots_dir, csv_path):
    """Open a W&B run, push whichever figures + CSV exist, then close it.

    Defensive against partial/empty sweeps (no traces -> no plots): only
    files that were actually written are logged.
    """
    import wandb

    if os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.init(project=project, name=name, config=config)

    images = {}
    for key, fname in PLOT_FILES:
        p = plots_dir / fname
        if p.exists():
            images[key] = wandb.Image(str(p))
    if images:
        wandb.log(images)
    if csv_path.exists():
        artifact = wandb.Artifact("budget_sweep_results", type="dataset")
        artifact.add_file(str(csv_path))
        wandb.log_artifact(artifact)
    wandb.finish()


def run_experiment(name, cfg):
    """Run one budget sweep (or replot it) into storage/budget_sweep/<name>/."""
    save_dir = Path("storage/budget_sweep") / name
    save_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = save_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    csv_path = save_dir / "results.csv"

    print(f"\n{'=' * 70}\nEXPERIMENT: {name}\n{'=' * 70}")

    if REUSE_RESULTS:
        if not csv_path.exists():
            print(f"[skip] REUSE_RESULTS set but {csv_path} missing")
            return
        records = load_records(str(csv_path))
        print(f"[reuse] loaded {len(records)} records")
        render_plots(records, plots_dir)
        print(f"Results: {csv_path}\nPlots:   {plots_dir}/")
        return

    scenic_file = str(Path(cfg["scenic_file"]).resolve())
    spec = cfg["spec"]()
    source_text = Path(scenic_file).read_text(encoding="utf-8")
    backend_name, scenic_model = resolve_backend(None, None, source_text)
    mode2d = default_mode2d_for_backend(backend_name)

    print(f"Parsing {scenic_file} …")
    t0 = time.time()
    graph = analyze_scenic_composition(scenic_file)
    partner = build_partner_format(graph)
    paths = parse_scenic_spec(partner)[cfg["composite_name"]]
    primitives = sorted(get_primitives(paths))
    graph_build_s = time.time() - t0
    print(f"  graph build : {graph_build_s:.3f}s")
    print(f"  backend     : {backend_name}")
    print(f"  primitives  : {primitives}")
    print(
        f"  monolithic  : {cfg['monolithic_name']} (max_steps={cfg['max_steps_mono']})"
    )
    print(
        f"  Plan: 2 simulations × {cfg['max_budget']:.0f}s = "
        f"{2 * cfg['max_budget']:.0f}s wall time."
    )

    records = sweep_snapshot(
        scenic_file,
        cfg["monolithic_name"],
        paths,
        primitives,
        spec,
        cfg["max_budget"],
        cfg["snapshot_every"],
        str(save_dir),
        cfg["features"],
        cfg["center_feat_idx"],
        cfg["max_steps_mono"],
        cfg["max_steps_primitive"],
        cfg["max_steps_overrides"],
        str(csv_path),
        graph_build_s,
        scenic_model,
        mode2d,
        cfg["prewarm_trim"],
    )

    render_plots(records, plots_dir)

    if USE_WANDB:
        log_to_wandb(
            WANDB_PROJECT,
            f"budget_sweep_{name}",
            {k: v for k, v in cfg.items() if k != "spec"}
            | {
                "experiment": name,
                "primitives": primitives,
                "backend": backend_name,
                "scenic_model": scenic_model,
            },
            plots_dir,
            csv_path,
        )

    print(f"Results: {csv_path}\nPlots:   {plots_dir}/")


def main():
    for name, cfg in EXPERIMENTS:
        run_experiment(name, cfg)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    load_env_file(REPO_ROOT / ".env")
    main()
