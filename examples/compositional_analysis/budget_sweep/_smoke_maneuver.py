"""Short-budget smoke runner for the two maneuver_choose cells.

Overrides max_budget to 120s per cell so each run lands in ~3 minutes.
Writes to storage/budget_sweep_v3_smoke_maneuver/. Disables W&B push.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

os.environ["WANDB_DISABLED"] = "1"

from examples.compositional_analysis.budget_sweep import checks  # noqa: E402
from examples.compositional_analysis.budget_sweep.main_v3 import (  # noqa: E402
    _build_curated_experiments,
)
from examples.compositional_analysis.budget_sweep.sweep import BudgetSweep  # noqa: E402


SMOKE_BUDGET = 120.0
SMOKE_SNAPSHOT = 30.0
SMOKE_ROOT = Path("storage/budget_sweep_v3_smoke_maneuver")


def main() -> None:
    cells = _build_curated_experiments()
    only = os.environ.get("SMOKE_CELL")
    if only:
        cells = [c for c in cells if only in c[0]]
    print(f"[smoke] running {len(cells)} cells at budget={SMOKE_BUDGET}s")
    for name, cfg in cells:
        cfg.max_budget = SMOKE_BUDGET
        cfg.snapshot_every = SMOKE_SNAPSHOT
        cfg.save_dir = SMOKE_ROOT / name
        print(f"\n{'=' * 70}\nSMOKE: {name}\n{'=' * 70}")
        sweep = BudgetSweep(cfg)
        records = sweep.run()
        print(f"[{name}] wrote {len(records)} records to {sweep.csv_path}")
        # Print last comp and mono row for quick eyeballing.
        comp_last = [r for r in records if r.get("method") == "compositional"]
        mono_last = [r for r in records if r.get("method") == "monolithic"]
        if comp_last:
            r = comp_last[-1]
            print(
                f"  comp final: rho={r.get('rho')!r:>22}  "
                f"eps={r.get('eps')!r:>22}  n={r.get('n_traces')!r}"
            )
        if mono_last:
            r = mono_last[-1]
            print(
                f"  mono final: rho={r.get('rho')!r:>22}  "
                f"eps={r.get('eps')!r:>22}  n={r.get('n_traces')!r}"
            )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
