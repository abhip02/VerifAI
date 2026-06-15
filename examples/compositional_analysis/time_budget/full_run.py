"""Full re-analysis pipeline: the three analyses in parallel, then the
eps-variant rendering once all three have finished.

  Stage 1 (parallel):
    - 15-minute full grid       (main --time-budget 15 --reuse-traces)
    - 60-minute Scenic grid     (main --time-budget 60 --only scenic --reuse-traces)
  Stage 2 (after the 15-minute grid, whose results it uses as reference):
    - compute-matched ablation  (ablation_compute_matched)
  Stage 3:
    - comp-eps=0 variants       (render_eps_variants)

Reuses the existing trace stores (no simulation). Child output is logged
to storage/budget_sweep_v4/full_run_logs/.

Run:
  python -m examples.compositional_analysis.time_budget.full_run
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LOG_DIR = REPO / "storage/budget_sweep_v4/full_run_logs"

PKG = "examples.compositional_analysis.time_budget"

STAGE1 = [
    (
        "grid_15min",
        [
            "-m",
            f"{PKG}.main",
            "--time-budget",
            "15",
            "--reuse-traces",
            "--workers",
            "7",
        ],
    ),
    (
        "scenic_60min",
        [
            "-m",
            f"{PKG}.main",
            "--time-budget",
            "60",
            "--only",
            "scenic",
            "--reuse-traces",
            "--workers",
            "4",
        ],
    ),
]
# The ablation reads the newest 15-minute convergence results as its
# full-pool / monolithic reference, so it must run after grid_15min.
STAGE2 = [("ablation", ["-m", f"{PKG}.ablation_compute_matched"])]
STAGE3 = [("render_eps_variants", ["-m", f"{PKG}.render_eps_variants"])]


def _run_stage(jobs, t0: float) -> list[str]:
    procs = []
    for name, argv in jobs:
        log = (LOG_DIR / f"{name}.log").open("w")
        p = subprocess.Popen(
            [sys.executable, *argv],
            cwd=REPO,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        procs.append((name, p, log))
        print(f"[full_run] started {name} (pid={p.pid}) -> {LOG_DIR / f'{name}.log'}")

    failed = []
    for name, p, log in procs:
        rc = p.wait()
        log.close()
        mins = (time.time() - t0) / 60
        status = "ok" if rc == 0 else f"FAILED (rc={rc})"
        print(f"[full_run] {name}: {status} after {mins:.0f} min")
        if rc != 0:
            failed.append(name)
    return failed


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for stage in (STAGE1, STAGE2, STAGE3):
        failed = _run_stage(stage, t0)
        if failed:
            raise SystemExit(
                f"[full_run] aborting; failed jobs: {failed} (see logs in {LOG_DIR})"
            )

    print(f"[full_run] all done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
