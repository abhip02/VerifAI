from __future__ import annotations

import csv
import multiprocessing as mp
import shutil
import time
from math import log, sqrt
from pathlib import Path
from typing import Any

import pandas as pd

from verifai.compositional_analysis import (
    CompositionalAnalysisEngine,
    ScenarioBase,
    relabel_traces,
)
from verifai.generate_graph_traces import (
    _count_trace_ids,
    _trim_partial_trace,
    _worker_generate_scenario,
    default_mode2d_for_backend,
    resolve_backend,
)
from verifai.scenic_composition_analysis import (
    analyze_scenic_composition,
    build_partner_format,
)
from verifai.scenic_parser import get_primitives, parse_scenic_spec

from . import checks
from .config import Record, Snapshot, SweepConfig


_RESULTS_FIELDS = [
    "method",
    "budget",
    "rho",
    "eps",
    "n_traces",
    "n_traces_breakdown",
    "status",
    "note",
]


# ---------------------------------------------------------------------------
# Small CSV / composition helpers (the only logic not already in verifai)
# ---------------------------------------------------------------------------


def _filter_csv_first_n_traces(src: str, dst: Path, n: int) -> None:
    """Copy ``src`` → ``dst`` keeping only rows with ``trace_id < n``."""
    with open(src) as fin, open(dst, "w") as fout:
        fout.write(fin.readline())  # header
        for line in fin:
            head, _, _ = line.partition(",")
            try:
                tid = int(head)
            except ValueError:
                continue
            if tid < n:
                fout.write(line)


def _trim_prewarm_rows(csv_path: str, n: int) -> None:
    """Drop the first ``n`` rows of every trace in-place and renumber ``step``.

    Paper §3.2 Challenge 2: prewarming primitives so successive KDE
    supports overlap. The Scenic worker writes raw simulator rows
    including the simulator's own settle-in steps; this drops them so
    the analyzer sees only the steady-state portion.
    """
    df = pd.read_csv(csv_path).sort_values(["trace_id", "step"])
    trimmed = []
    for _, grp in df.groupby("trace_id"):
        kept = grp.iloc[n:].copy()
        kept["step"] = range(len(kept))
        trimmed.append(kept)
    if trimmed:
        pd.concat(trimmed, ignore_index=True).to_csv(csv_path, index=False)


def _composition_length(paths) -> int:
    """Length of the longest linearized path (Alg. 1 output).

    A length-1 path is a single-primitive composite (no KDE handoff →
    direct DFA-label estimate, needs only >=1 trace). Length >=2 needs
    >=2 traces per primitive because the KDE handoff fit requires it.
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


def _hoeffding_eps(n: int, delta: float) -> float:
    """Monolithic CI half-width: ``sqrt(log(2/δ) / (2N))``."""
    return float(sqrt(log(2 / delta) / (2 * max(n, 1))))


# ---------------------------------------------------------------------------
# BudgetSweep
# ---------------------------------------------------------------------------


class BudgetSweep:
    """Drives one compositional+monolithic budget sweep end-to-end.

    Holds the parsed scenic graph (``paths``, ``primitives``) so the three
    workhorse methods share state instead of threading it through long
    argument lists. Construct once, call :meth:`run`.

    Attributes:
        cfg: The :class:`SweepConfig` this sweep was constructed with.
        paths: Composition paths for ``cfg.composite_name``.
        primitives: Sorted list of primitive scenario names in ``paths``.
        csv_path: Path to the ``results.csv`` file this sweep writes.
    """

    def __init__(self, cfg: SweepConfig) -> None:
        """Validate ``cfg`` and prepare output paths. Scenic parsing
        happens lazily in :meth:`run` so ``__init__`` stays cheap and
        the experiment list can be constructed without I/O.
        """
        checks.check_config(cfg)
        cfg.scenic_file = Path(cfg.scenic_file).resolve()
        self.cfg: SweepConfig = cfg
        self.paths: list = []
        self.primitives: list[str] = []
        self.csv_path: Path = cfg.save_dir / "results.csv"

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_with_snapshots(
        self,
        jobs: list[dict[str, Any]],
        method_dir: Path,
    ) -> tuple[dict[str, str], list[Snapshot]]:
        """Spawn one worker per job, hard-stop at ``cfg.max_budget``,
        snapshot every ``cfg.snapshot_every`` seconds.

        Returns:
            ``(logs, timeline)`` — ``logs`` maps each scenario name with
            >=1 complete trace to its final CSV path; ``timeline`` is the
            chronological list of snapshots (including the final
            post-stop snapshot).
        """
        method_dir = Path(method_dir)
        method_dir.mkdir(parents=True, exist_ok=True)
        scenario_names = checks.check_jobs(jobs, method_dir)

        # Launch workers.
        processes: list[tuple[str, mp.Process, Path]] = []
        for idx, job in enumerate(jobs):
            full_job = {
                "scenic_file": str(job["scenic_file"]),
                "scenario_name": str(job["scenario_name"]),
                "save_dir": str(method_dir),
                "n": int(job.get("n") or 10**9),
                "max_steps": int(job["max_steps"]),
                "mode2d": bool(job.get("mode2d", True)),
                "model": job.get("model"),
                "max_iterations": int(job.get("max_iterations", 2000)),
                "position": idx,
            }
            scen_dir = method_dir / full_job["scenario_name"]
            scen_dir.mkdir(parents=True, exist_ok=True)
            csv_path = scen_dir / "traces.csv"
            if csv_path.exists():
                csv_path.unlink()
            proc = mp.Process(target=_worker_generate_scenario, args=(full_job,))
            proc.start()
            processes.append((full_job["scenario_name"], proc, csv_path))

        def _all_counts() -> dict[str, int]:
            return {name: _count_trace_ids(path) for name, _, path in processes}

        timeline: list[Snapshot] = []
        start = time.time()
        next_snap = self.cfg.snapshot_every
        poll_sleep = max(0.25, min(self.cfg.snapshot_every / 4.0, 2.0))
        hard_stopped = False
        counts_at_stop: dict[str, int] = {}

        while True:
            elapsed = time.time() - start
            if elapsed >= next_snap:
                timeline.append((elapsed, _all_counts()))
                next_snap += self.cfg.snapshot_every
            if elapsed >= self.cfg.max_budget:
                counts_at_stop = _all_counts()
                for _, proc, _ in processes:
                    if proc.is_alive():
                        proc.terminate()
                        proc.join(timeout=5)
                        if proc.is_alive():
                            proc.kill()
                            proc.join()
                # Avoid a duplicate final checkpoint if a snapshot was
                # just taken in this same poll iteration.
                if not timeline or (elapsed - timeline[-1][0]) > poll_sleep:
                    timeline.append((elapsed, counts_at_stop))
                hard_stopped = True
                print(
                    f"[hard stop] {elapsed:.1f}s >= max_budget {self.cfg.max_budget:.1f}s"
                )
                break
            if all(not proc.is_alive() for _, proc, _ in processes):
                timeline.append((elapsed, _all_counts()))
                print(f"[natural finish] at {elapsed:.1f}s")
                break
            time.sleep(poll_sleep)

        # Trim partial trailing traces (hard-stop case) and build the
        # ``logs`` dict, only including scenarios that produced >=1 trace.
        logs: dict[str, str] = {}
        for name, _proc, csv_path in processes:
            if not csv_path.exists():
                continue
            if hard_stopped:
                keep = counts_at_stop.get(name, 0)
                _trim_partial_trace(csv_path, keep)
                n = keep
            else:
                n = _count_trace_ids(csv_path)
            if n > 0:
                logs[name] = str(csv_path)

        # Apply prewarm trim (compositional primitives only — monolithic
        # name typically isn't in cfg.prewarm_trim, so this is a no-op
        # on the mono call).
        for name, log_path in logs.items():
            trim = self.cfg.prewarm_trim.get(name, 0)
            if trim and trim > 0:
                _trim_prewarm_rows(log_path, trim)
                print(f"[prewarm] trimmed {trim} rows per trace from {name}")

        checks.check_simulation_result(
            logs,
            timeline,
            scenario_names,
            self.cfg.max_budget,
            self.cfg.snapshot_every,
        )
        checks.check_throughput(timeline, scenario_names, self.cfg.max_budget)
        for name, log_path in logs.items():
            checks.check_trace_csv(log_path, self.cfg.features, name)
        return logs, timeline

    # ------------------------------------------------------------------
    # Per-checkpoint analyzers
    # ------------------------------------------------------------------

    def analyze_compositional(
        self,
        snapshot: Snapshot,
        logs: dict[str, str],
    ) -> Record:
        """Filter each primitive CSV to its first-N traces, relabel
        with the DFA spec, run ``check_with_dfa_scenic``. Returns a
        status record (``missing_primitives`` / ``insufficient_data``)
        during the Fig. 4 startup window."""
        checks.check_compositional_inputs(self.primitives, snapshot, logs)
        elapsed, counts = snapshot
        n_traces = sum(counts.get(p, 0) for p in self.primitives)
        breakdown = ";".join(f"{p}={counts.get(p, 0)}" for p in self.primitives)

        n_steps = _composition_length(self.paths)
        min_traces = 2 if n_steps > 1 else 1

        base_record: Record = {
            "method": "compositional",
            "budget": elapsed,
            "rho": None,
            "eps": None,
            "n_traces": n_traces,
            "n_traces_breakdown": breakdown,
            "status": "ok",
            "note": "",
        }

        missing = sorted(set(self.primitives) - set(logs))
        if missing:
            base_record["status"] = "missing_primitives"
            base_record["note"] = f"no traces yet for: {missing}"
            checks.check_record(base_record, "compositional", elapsed, n_traces)
            checks.check_rho_signal(base_record, self.cfg.composite_name)
            return base_record

        if any(counts.get(p, 0) < min_traces for p in self.primitives):
            base_record["status"] = "insufficient_data"
            base_record["note"] = f"need >={min_traces} trace(s) per primitive"
            checks.check_record(base_record, "compositional", elapsed, n_traces)
            checks.check_rho_signal(base_record, self.cfg.composite_name)
            return base_record

        temp_dir = self.cfg.save_dir / "_temp" / f"comp_t{int(elapsed)}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            temp_logs: dict[str, str] = {}
            for p in self.primitives:
                n = counts[p]
                dst = temp_dir / f"{p}_n{n}.csv"
                _filter_csv_first_n_traces(logs[p], dst, n)
                relabel_traces(str(dst), self.cfg.spec)
                temp_logs[p] = str(dst)

            try:
                base = ScenarioBase(temp_logs, delta=self.cfg.delta)
                engine = CompositionalAnalysisEngine(base)
                rho, eps = engine.check_with_dfa_scenic(
                    self.paths,
                    self.cfg.spec,
                    features=self.cfg.features,
                    center_feat_idx=self.cfg.center_feat_idx,
                )
                base_record["rho"] = float(rho)
                base_record["eps"] = float(eps)
            except Exception as exc:
                base_record["status"] = "analysis_error"
                base_record["note"] = repr(exc)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        checks.check_record(base_record, "compositional", elapsed, n_traces)
        checks.check_rho_signal(base_record, self.cfg.composite_name)
        return base_record

    def analyze_monolithic(
        self,
        snapshot: Snapshot,
        log: str | None,
    ) -> Record:
        """Filter mono CSV to first-N traces, relabel (returns ρ),
        compute eps = sqrt(log(2/δ)/(2N))."""
        n = checks.check_monolithic_inputs(self.cfg.monolithic_name, snapshot, log)
        elapsed, _ = snapshot

        base_record: Record = {
            "method": "monolithic",
            "budget": elapsed,
            "rho": None,
            "eps": None,
            "n_traces": n,
            "n_traces_breakdown": f"{self.cfg.monolithic_name}={n}",
            "status": "ok",
            "note": "",
        }

        if n < 1 or log is None:
            base_record["status"] = "no_traces"
            base_record["note"] = "no monolithic traces at this checkpoint"
            checks.check_record(base_record, "monolithic", elapsed, n)
            checks.check_rho_signal(base_record, self.cfg.monolithic_name)
            return base_record

        temp_dir = self.cfg.save_dir / "_temp" / f"mono_t{int(elapsed)}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            dst = temp_dir / f"{self.cfg.monolithic_name}_n{n}.csv"
            _filter_csv_first_n_traces(log, dst, n)
            try:
                rho = relabel_traces(str(dst), self.cfg.spec)
                base_record["rho"] = float(rho)
                base_record["eps"] = _hoeffding_eps(n, self.cfg.delta)
            except Exception as exc:
                base_record["status"] = "analysis_error"
                base_record["note"] = repr(exc)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        checks.check_record(base_record, "monolithic", elapsed, n)
        checks.check_rho_signal(base_record, self.cfg.monolithic_name)
        return base_record

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run(self) -> list[Record]:
        """Parse scenic, run both methods sequentially, write CSV."""
        # 1. Parse scenic source → composition paths + primitive set.
        print(f"[parse] {self.cfg.scenic_file}")
        t0 = time.time()
        graph = analyze_scenic_composition(self.cfg.scenic_file)
        partner = build_partner_format(graph)
        self.paths = parse_scenic_spec(partner)[self.cfg.composite_name]
        self.primitives = sorted(get_primitives(self.paths))
        print(f"[parse] {time.time() - t0:.2f}s — primitives: {self.primitives}")
        checks.check_parsed_graph(self.paths, self.primitives)
        checks.check_run_preconditions(self.paths, self.primitives)

        # 2. Resolve backend (simulator + 2D mode + scenic model).
        source_text = self.cfg.scenic_file.read_text(encoding="utf-8")
        backend_name, scenic_model = resolve_backend(None, None, source_text)
        mode2d = default_mode2d_for_backend(backend_name)
        print(f"[backend] {backend_name} (mode2d={mode2d})")

        self.cfg.save_dir.mkdir(parents=True, exist_ok=True)

        # 3. Compositional simulation.
        print(f"\n=== Compositional sim (max_budget={self.cfg.max_budget:.0f}s) ===")
        comp_jobs = [
            {
                "scenic_file": str(self.cfg.scenic_file),
                "scenario_name": p,
                "max_steps": self.cfg.max_steps_overrides.get(
                    p, self.cfg.max_steps_primitive
                ),
                "model": scenic_model,
                "mode2d": mode2d,
            }
            for p in self.primitives
        ]
        logs_comp, timeline_comp = self.simulate_with_snapshots(
            comp_jobs, self.cfg.save_dir / "compositional"
        )

        # 4. Compositional per-checkpoint analysis.
        print(f"[analyze] {len(timeline_comp)} compositional checkpoints …")
        records: list[Record] = []
        for snap in timeline_comp:
            records.append(self.analyze_compositional(snap, logs_comp))

        # 5. Monolithic simulation.
        print(f"\n=== Monolithic sim (max_budget={self.cfg.max_budget:.0f}s) ===")
        mono_jobs = [
            {
                "scenic_file": str(self.cfg.scenic_file),
                "scenario_name": self.cfg.monolithic_name,
                "max_steps": self.cfg.max_steps_mono,
                "model": scenic_model,
                "mode2d": mode2d,
            }
        ]
        logs_mono, timeline_mono = self.simulate_with_snapshots(
            mono_jobs, self.cfg.save_dir / "monolithic"
        )
        mono_log = logs_mono.get(self.cfg.monolithic_name)

        # 6. Monolithic per-checkpoint analysis.
        print(f"[analyze] {len(timeline_mono)} monolithic checkpoints …")
        for snap in timeline_mono:
            records.append(self.analyze_monolithic(snap, mono_log))

        # 7. Write results.csv.
        with self.csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_RESULTS_FIELDS)
            writer.writeheader()
            for r in records:
                writer.writerow({k: r.get(k, "") for k in _RESULTS_FIELDS})

        # 8. Clean up the analyzer's temp scratch dir (best-effort).
        shutil.rmtree(self.cfg.save_dir / "_temp", ignore_errors=True)

        checks.check_run_result(records, self.csv_path)
        checks.check_method_agreement(records)
        return records
