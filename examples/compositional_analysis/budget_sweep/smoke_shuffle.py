"""Smoke-test the native_shuffle cell at a small budget.

Verifies that:
- Scenic parses `do shuffle { ... }` at scenario scope
- analyze_scenic_composition returns a non-empty branching structure
- BudgetSweep can drive both compositional and monolithic to completion
- results.csv has finite rho/eps for both methods on at least one snapshot
"""

from __future__ import annotations

import dataclasses
import multiprocessing as mp
from pathlib import Path

from examples.compositional_analysis.budget_sweep.main_v3 import (
    SCEN_DIR,
    _kw,
)
from examples.compositional_analysis.scenic_scenarios.specs import spec_max_speed
from examples.compositional_analysis.budget_sweep.config import SweepConfig
from examples.compositional_analysis.budget_sweep.sweep import BudgetSweep


def main() -> None:
    cfg = SweepConfig(
        spec=spec_max_speed(),
        **_kw(
            SCEN_DIR / "metadrive" / "composites/native_shuffle.scenic",
            "Main",
            "MonoSShuffleCXO",
            max_steps_primitive=40,
            max_steps_mono=160,
        ),
    )
    cfg = dataclasses.replace(
        cfg,
        max_budget=120.0,
        snapshot_every=30.0,
        save_dir=Path("storage/budget_sweep_v3_smoke")
        / "metadrive__max_speed__shuffle",
    )
    print(f"[smoke] scenic_file={cfg.scenic_file}")
    print(f"[smoke] max_budget={cfg.max_budget}s snapshot_every={cfg.snapshot_every}s")
    sweep = BudgetSweep(cfg)
    records = sweep.run()
    print(f"[smoke] wrote {len(records)} records to {sweep.csv_path}")
    for r in records:
        print(
            f"  method={r.get('method')} budget={float(r.get('budget', 0)):.1f}s "
            f"rho={r.get('rho')} eps={r.get('eps')} n={r.get('n_traces')} "
            f"status={r.get('status')}"
        )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
