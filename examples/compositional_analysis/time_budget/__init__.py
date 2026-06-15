"""Budget-sweep package — reproduces §4.3 (Q2: efficiency at matched
wall-clock budget) from our paper:

    Compositional_Analysis_for_Safety_Specifications___RV26.pdf
    (Pomalapally & Raeesi, RV26 submission)

The trace-replay grid (4 specs × 7 scenarios = 28 cells) is driven from
:mod:`budget_sweep.main`; :mod:`budget_sweep.full_run` orchestrates the
full re-analysis (grid → compute-matched ablation → eps variants).

Module layout:
    main.py                      — 28-cell trace-replay grid + ``main()`` CLI.
    full_run.py                  — multi-stage re-analysis orchestrator.
    ablation_compute_matched.py  — compute-matched ablation.
    render_eps_variants.py       — comp-eps=0 presentation variants.
    plots.py                     — figure rendering.
    config.py                    — :data:`Record` type alias.

Run the grid with::

    python -m examples.compositional_analysis.budget_sweep.main
"""

from .config import Record

__all__ = ["Record"]
