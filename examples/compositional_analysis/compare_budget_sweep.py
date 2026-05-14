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

DFA spec: pass ``--spec_module path/to/spec.py`` exposing ``make_spec()``.
Default is a tollgate-style safety spec (speed never exceeds MAX_SPEED
post-warmup), matching the e2e_4way test.
"""

import argparse
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
    "n_traces_breakdown",
    "graph_build_s",
    "status",
    "note",
]

WARMUP_STEPS = 25
DEFAULT_MAX_SPEED = 5.5
HOEFFDING_DELTA = 0.05


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


def hoeffding_eps(n, delta=HOEFFDING_DELTA):
    return float(np.sqrt(np.log(2 / delta) / (2 * max(n, 1))))


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
    if any(counts.get(p, 0) < 2 for p in primitives):
        return {
            "rho": None,
            "eps": None,
            "n_traces": sum(counts.get(p, 0) for p in primitives),
            "n_traces_breakdown": breakdown,
            "elapsed": elapsed,
            "status": "insufficient_data",
            "note": "need >=2 traces per primitive for KDE",
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
                "budget": elapsed,
                "graph_build_s": graph_build_s,
                **r,
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
    series = _series(records, "n_traces")
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
    ax.set_ylabel("# trace episodes completed")
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


def parse_max_steps_overrides(items):
    out = {}
    for it in items or ():
        if "=" not in it:
            raise ValueError(f"--max_steps_override expects NAME=INT, got: {it}")
        k, v = it.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


def parse_prewarm_trim(items):
    out = {}
    for it in items or ():
        if "=" not in it:
            raise ValueError(f"--prewarm_trim expects NAME=INT, got: {it}")
        k, v = it.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


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
            out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scenic backend: monolithic vs. compositional SMC over a "
            "continuous run with periodic checkpoints."
        )
    )
    parser.add_argument(
        "--scenic_file",
        required=True,
        help="Composite .scenic source (must declare composite + monolithic).",
    )
    parser.add_argument(
        "--composite_name",
        default="Main",
        help="Composite scenario name (default: Main).",
    )
    parser.add_argument(
        "--monolithic_name",
        default="MonolithicMain",
        help="Monolithic counterpart scenario (default: MonolithicMain).",
    )
    parser.add_argument(
        "--max_budget",
        type=float,
        default=1800.0,
        help="Total wall-clock per method (default: 1800s = 30min).",
    )
    parser.add_argument(
        "--snapshot_every",
        type=float,
        default=30.0,
        help="Checkpoint interval in seconds (default: 30).",
    )
    parser.add_argument(
        "--save_dir",
        default="storage/scenic_budget_sweep",
    )
    parser.add_argument(
        "--spec_module",
        default=None,
        help="Path to .py with make_spec(); else default safety spec.",
    )
    parser.add_argument(
        "--max_speed",
        type=float,
        default=DEFAULT_MAX_SPEED,
    )
    parser.add_argument(
        "--max_steps_primitive",
        type=int,
        default=85,
    )
    parser.add_argument(
        "--max_steps_mono",
        type=int,
        default=170,
    )
    parser.add_argument(
        "--max_steps_override",
        nargs="+",
        default=[],
        help="Per-primitive max_steps override (e.g. Subscenario2L=110).",
    )
    parser.add_argument(
        "--prewarm_trim",
        nargs="+",
        default=[],
        help="Per-primitive prewarm trim count, applied post-generation "
        "(e.g. GoStraight=35 TurnLeft=35 TurnRight=35). Drops the first "
        "N rows of each trace and renumbers ``step``.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=["speed"],
    )
    parser.add_argument(
        "--center_feat_idx",
        nargs="*",
        type=int,
        default=[],
    )
    parser.add_argument("--backend", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--reuse_results",
        action="store_true",
        help="Skip sim; replot from results.csv.",
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = save_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    csv_path = save_dir / "results.csv"

    if args.reuse_results:
        if not csv_path.exists():
            raise FileNotFoundError(f"--reuse_results given but {csv_path} missing")
        records = load_records(str(csv_path))
        print(f"Loaded {len(records)} records.")
    else:
        scenic_file = str(Path(args.scenic_file).resolve())
        spec = (
            load_spec(args.spec_module)
            if args.spec_module
            else default_spec(max_speed=args.max_speed)
        )
        max_steps_overrides = parse_max_steps_overrides(args.max_steps_override)
        prewarm_trim_overrides = parse_prewarm_trim(args.prewarm_trim)

        source_text = Path(scenic_file).read_text(encoding="utf-8")
        backend_name, scenic_model = resolve_backend(
            args.backend, args.model, source_text
        )
        mode2d = default_mode2d_for_backend(backend_name)

        print(f"Parsing {scenic_file} …")
        t0 = time.time()
        graph = analyze_scenic_composition(scenic_file)
        partner = build_partner_format(graph)
        paths = parse_scenic_spec(partner)[args.composite_name]
        primitives = sorted(get_primitives(paths))
        graph_build_s = time.time() - t0
        print(f"  graph build : {graph_build_s:.3f}s")
        print(f"  backend     : {backend_name}")
        print(f"  model       : {scenic_model}")
        print(f"  mode2d      : {mode2d}")
        print(f"  primitives  : {primitives}")
        print(f"  paths       : {paths}")
        print(
            f"  monolithic  : {args.monolithic_name} (max_steps={args.max_steps_mono})"
        )
        n_checkpoints = max(1, int(args.max_budget // args.snapshot_every))
        print(
            f"\nPlan: 2 simulations × {args.max_budget:.0f}s = "
            f"{2 * args.max_budget:.0f}s wall time, "
            f"{n_checkpoints} checkpoints × 2 methods."
        )

        records = sweep_snapshot(
            scenic_file,
            args.monolithic_name,
            paths,
            primitives,
            spec,
            args.max_budget,
            args.snapshot_every,
            str(save_dir),
            args.features,
            args.center_feat_idx,
            args.max_steps_mono,
            args.max_steps_primitive,
            max_steps_overrides,
            str(csv_path),
            graph_build_s,
            scenic_model,
            mode2d,
            prewarm_trim_overrides,
        )

    plot_eps_vs_budget(records, str(plots_dir / "eps_vs_budget.png"))
    plot_rho_vs_budget(records, str(plots_dir / "rho_vs_budget.png"))
    plot_throughput(records, str(plots_dir / "throughput.png"))
    plot_speedup_vs_budget(records, str(plots_dir / "speedup_vs_budget.png"))
    plot_wallclock_combo(records, str(plots_dir / "wallclock.png"))

    print(f"\nResults: {csv_path}")
    print(f"Plots:   {plots_dir}/")


if __name__ == "__main__":
    mp.set_start_method("spawn")

    # ----------------------------------------------------------------
    # Load WANDB_API_KEY (and any other secrets) from .env at the repo
    # root. Manual parser so we don't pull in python-dotenv.
    # ----------------------------------------------------------------
    ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
    if ENV_PATH.exists():
        with open(ENV_PATH) as _envf:
            for _line in _envf:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _k, _, _v = _line.partition("=")
                _k = _k.strip()
                _v = _v.strip().strip('"').strip("'")
                if _k and _k not in os.environ:
                    os.environ[_k] = _v

    # ----------------------------------------------------------------
    # Hard-coded 1-hour sweep configuration: Set~B (N=5 wander) under
    # phi_brake. main()'s argparse plumbing remains available for
    # alternate configurations.
    # ----------------------------------------------------------------
    SCENIC_FILE = str(
        Path(
            "examples/compositional_analysis/dfa_tests/e2e_4way_example/"
            "4_way_intersection_scenic/composed_wander.scenic"
        ).resolve()
    )
    SAVE_DIR = Path("storage/scenic_budget_sweep_wander")
    MAX_BUDGET = 1800.0  # seconds per method (mono + comp run sequentially)
    SNAPSHOT_EVERY = 30.0  # checkpoint cadence -> 60 points per method
    COMPOSITE_NAME = "Main"
    MONOLITHIC_NAME = "MonolithicWander"
    MAX_STEPS_PRIMITIVE = 75  # matches test_4way_intersection_wander.py
    MAX_STEPS_MONO = 375  # 5 * MAX_STEPS_PRIMITIVE
    MAX_STEPS_OVERRIDES = {}  # all primitives use MAX_STEPS_PRIMITIVE
    PREWARM_TRIM_OVERRIDES = {
        "GoStraight": 35,
        "TurnLeft": 35,
        "TurnRight": 35,
    }
    FEATURES = ["x", "y", "speed"]
    CENTER_FEAT_IDX = [0, 1]
    SPEC_MODULE_PATH = str(
        Path(__file__).resolve().parent / "specs" / "at_most_one_brake.py"
    )

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = SAVE_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    csv_path = SAVE_DIR / "results.csv"
    spec = load_spec(SPEC_MODULE_PATH)

    source_text = Path(SCENIC_FILE).read_text(encoding="utf-8")
    backend_name, scenic_model = resolve_backend(None, None, source_text)
    mode2d = default_mode2d_for_backend(backend_name)

    print(f"Parsing {SCENIC_FILE} …")
    t0 = time.time()
    graph = analyze_scenic_composition(SCENIC_FILE)
    partner = build_partner_format(graph)
    paths = parse_scenic_spec(partner)[COMPOSITE_NAME]
    primitives = sorted(get_primitives(paths))
    graph_build_s = time.time() - t0
    print(f"  graph build : {graph_build_s:.3f}s")
    print(f"  backend     : {backend_name}")
    print(f"  primitives  : {primitives}")
    print(f"  paths       : {paths}")
    print(
        f"  Plan: 2 simulations × {MAX_BUDGET:.0f}s = "
        f"{2 * MAX_BUDGET:.0f}s wall time (~1 hour)."
    )

    # ----------------------------------------------------------------
    # Initialize Weights & Biases. API key is expected in env (loaded
    # from .env above). wandb.login() returns silently if already
    # authenticated; wandb.init() opens the run for this sweep.
    # ----------------------------------------------------------------
    import wandb

    if os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb_run = wandb.init(
        project="verifai-compositional-analysis",
        name="scenic_budget_sweep_set_b_n5_brake",
        config={
            "scenic_file": SCENIC_FILE,
            "composite_name": COMPOSITE_NAME,
            "monolithic_name": MONOLITHIC_NAME,
            "max_budget_s": MAX_BUDGET,
            "snapshot_every_s": SNAPSHOT_EVERY,
            "max_steps_primitive": MAX_STEPS_PRIMITIVE,
            "max_steps_mono": MAX_STEPS_MONO,
            "max_steps_overrides": MAX_STEPS_OVERRIDES,
            "prewarm_trim_overrides": PREWARM_TRIM_OVERRIDES,
            "features": FEATURES,
            "center_feat_idx": CENTER_FEAT_IDX,
            "spec_module": SPEC_MODULE_PATH,
            "primitives": primitives,
            "backend": backend_name,
            "scenic_model": scenic_model,
        },
    )

    records = sweep_snapshot(
        SCENIC_FILE,
        MONOLITHIC_NAME,
        paths,
        primitives,
        spec,
        MAX_BUDGET,
        SNAPSHOT_EVERY,
        str(SAVE_DIR),
        FEATURES,
        CENTER_FEAT_IDX,
        MAX_STEPS_MONO,
        MAX_STEPS_PRIMITIVE,
        MAX_STEPS_OVERRIDES,
        str(csv_path),
        graph_build_s,
        scenic_model,
        mode2d,
        PREWARM_TRIM_OVERRIDES,
    )

    plot_eps_vs_budget(records, str(plots_dir / "eps_vs_budget.png"))
    plot_rho_vs_budget(records, str(plots_dir / "rho_vs_budget.png"))
    plot_throughput(records, str(plots_dir / "throughput.png"))
    plot_speedup_vs_budget(records, str(plots_dir / "speedup_vs_budget.png"))
    plot_wallclock_combo(records, str(plots_dir / "wallclock.png"))

    # ----------------------------------------------------------------
    # Push whichever figures + CSV actually exist to wandb, then close
    # the run. Defensive against partial/empty sweeps (no traces -> no
    # plots): only log files that were written.
    # ----------------------------------------------------------------
    _log = {}
    for _name, _file in [
        ("wallclock", "wallclock.png"),
        ("eps_vs_budget", "eps_vs_budget.png"),
        ("rho_vs_budget", "rho_vs_budget.png"),
        ("throughput", "throughput.png"),
        ("speedup_vs_budget", "speedup_vs_budget.png"),
    ]:
        _p = plots_dir / _file
        if _p.exists():
            _log[_name] = wandb.Image(str(_p))
    if _log:
        wandb.log(_log)
    if csv_path.exists():
        _artifact = wandb.Artifact("budget_sweep_results", type="dataset")
        _artifact.add_file(str(csv_path))
        wandb.log_artifact(_artifact)
    wandb.finish()

    print(f"\nResults: {csv_path}")
    print(f"Plots:   {plots_dir}/")
