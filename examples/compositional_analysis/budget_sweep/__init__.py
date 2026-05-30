"""Budget-sweep package — reproduces §4.3 (Q2: efficiency at matched
wall-clock budget) from our paper:

    Compositional_Analysis_for_Safety_Specifications___RV26.pdf
    (Pomalapally & Raeesi, RV26 submission)

Public API:
    - :class:`SweepConfig`   — inputs for one sweep (paths, spec, budgets).
    - :class:`BudgetSweep`   — driver: parses scenic, runs both methods
      sequentially, analyzes every checkpoint, writes ``results.csv``.
    - :data:`Snapshot`, :data:`Record` — type aliases used across modules.

Run via ``python -m budget_sweep`` (entry point in :mod:`budget_sweep.main`).

Module layout:
    config.py  — :class:`SweepConfig` + type aliases.
    sweep.py   — :class:`BudgetSweep` (the four workhorse methods).
    checks.py  — runtime/scenario-health checks that **log** (don't raise).
    main.py    — ``EXPERIMENTS`` list + ``main()`` entry point.
"""

from .config import Record, Snapshot, SweepConfig
from .sweep import BudgetSweep

__all__ = ["BudgetSweep", "Record", "Snapshot", "SweepConfig"]
