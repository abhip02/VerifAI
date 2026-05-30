"""Entry point: ``EXPERIMENTS`` list + ``main()``.

No CLI — configure by editing :data:`EXPERIMENTS` below. Each entry is
a ``(name, SweepConfig)`` pair; experiments execute one after another
(never in parallel) so each owns the machine during its wall-clock
window, which §4.3's matched-T axis requires.

Run with::

    python -m budget_sweep
"""

from __future__ import annotations

import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path

from verifai.monitor import automaton_specification

from . import checks
from .config import Record, SweepConfig
from .sweep import BudgetSweep

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "verifai-budget-sweep-v2")

# Repo root is four parents up from this file:
#   .../VerifAI/examples/compositional_analysis/budget_sweep/main.py
#   parents[0] = budget_sweep/
#   parents[1] = compositional_analysis/
#   parents[2] = examples/
#   parents[3] = VerifAI/
REPO_ROOT = Path(__file__).resolve().parents[3]
SCENIC_DIR = (
    REPO_ROOT
    / "examples/compositional_analysis/dfa_tests/e2e_4way_example"
    / "4_way_intersection_scenic"
)


# The DFA factories themselves live with the e2e tests so the budget
# sweep doesn't fork the spec definitions. ``make_spec_safe_under_max``
# is the Markovian "did speed stay ≤ MAX for the whole post-warmup
# segment?" automaton used to produce Fig. 4 (cf. §4.3). Each call
# returns a fresh ``automaton_specification`` so experiments don't
# share mutable DFA state.
from examples.compositional_analysis.dfa_tests.e2e_4way_example.test_4way_intersection_wander import (  # noqa: E402
    make_spec_safe_under_max,
)


def _make_safe_under_max_spec() -> automaton_specification:
    """Factory for §4.3's ``safe_under_max`` Markovian DFA.

    Delegates to :func:`make_spec_safe_under_max` in
    ``dfa_tests/e2e_4way_example/test_4way_intersection_wander.py`` —
    same DFA used to produce Fig. 4. Threshold (``MAX_SPEED``) and
    warmup (``WARMUP_STEPS``) constants live in that module.
    """
    return make_spec_safe_under_max()


# ---------------------------------------------------------------------------
# Experiments to run, sequentially, one per :class:`BudgetSweep` instance.
# Add or comment out entries to change what ``main()`` executes. Each entry
# is ``(name, cfg)``; results land in ``cfg.save_dir / name``.
# ---------------------------------------------------------------------------

EXPERIMENTS: list[tuple[str, SweepConfig]] = [
    # Paper §4.3 / Fig. 4 — Set C under safe_under_max.
    # Composite: Sub1 ; choose{Sub2L, Sub2R, Sub2S} (two-step approach-
    # then-turn). Spec: safe_under_max (Markovian speed-threshold safety).
    # Features = ["speed"] only → center_feat_idx=[] (no position centering
    # needed for an intrinsic signal).
    (
        "set_c_safe_under_max",
        SweepConfig(
            scenic_file=SCENIC_DIR / "composed_scenarios.scenic",
            composite_name="Main",
            monolithic_name="MonolithicMain",
            spec=_make_safe_under_max_spec(),
            max_budget=1800.0,
            snapshot_every=30.0,
            max_steps_primitive=85,
            max_steps_mono=170,
            features=["speed"],
            center_feat_idx=[],
            delta=0.05,
            max_steps_overrides={
                p: 110 for p in ("Subscenario2L", "Subscenario2R", "Subscenario2S")
            },
            prewarm_trim={
                p: 25 for p in ("Subscenario2L", "Subscenario2R", "Subscenario2S")
            },
            save_dir=Path("storage/budget_sweep_v2"),
        ),
    ),
]


def _push_to_wandb(
    results_per_experiment: dict[str, tuple[Path, list[Record]]],
) -> None:
    """Upload the checks log file + each experiment's ``results.csv``
    to Weights & Biases, plus a per-method (T, ρ̂, ε̂) series for the
    W&B UI to plot. Skipped silently if wandb is not installed or
    ``WANDB_DISABLED`` is set in the environment.
    """
    if os.environ.get("WANDB_DISABLED"):
        print("[wandb] WANDB_DISABLED set; skipping upload")
        return
    try:
        import wandb
    except ImportError:
        print("[wandb] wandb not installed; skipping upload")
        return

    run_name = f"budget_sweep_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project=WANDB_PROJECT, name=run_name, reinit=True)

    # 1. Whole log file as an artifact (the main ask).
    log_path = checks._LOG_PATH
    if log_path.exists():
        art = wandb.Artifact("budget_sweep_checks_log", type="log")
        art.add_file(str(log_path))
        wandb.log_artifact(art)
        wandb.save(str(log_path), base_path=str(log_path.parent), policy="now")

    # 2. Per-experiment results.csv + (T, ρ̂, ε̂) line series.
    for name, (csv_path, records) in results_per_experiment.items():
        if csv_path.exists():
            art = wandb.Artifact(f"{name}_results", type="results")
            art.add_file(str(csv_path))
            wandb.log_artifact(art)

        # Step the series by budget seconds so the W&B chart x-axis is T.
        for r in records:
            if r.get("rho") is None or r.get("eps") is None:
                continue
            wandb.log(
                {
                    f"{name}/{r['method']}/rho": float(r["rho"]),
                    f"{name}/{r['method']}/eps": float(r["eps"]),
                    f"{name}/{r['method']}/n_traces": int(r["n_traces"]),
                },
                step=int(float(r["budget"])),
            )

    wandb.finish()


def main() -> None:
    """Run every :class:`SweepConfig` in :data:`EXPERIMENTS` sequentially,
    then upload the checks log + all results to W&B (best-effort)."""
    seen_names: set[str] = set()
    results_per_experiment: dict[str, tuple[Path, list[Record]]] = {}
    for name, cfg in EXPERIMENTS:
        if not name or name in seen_names:
            checks._log.warning("duplicate or empty experiment name: %r", name)
        seen_names.add(name)

        # Per-experiment save_dir nesting so results don't clobber each other.
        cfg.save_dir = cfg.save_dir / name

        print(f"\n{'=' * 70}\nEXPERIMENT: {name}\n{'=' * 70}")
        sweep = BudgetSweep(cfg)
        records = sweep.run()
        print(f"[{name}] wrote {len(records)} records to {sweep.csv_path}")
        results_per_experiment[name] = (sweep.csv_path, records)

    _push_to_wandb(results_per_experiment)


if __name__ == "__main__":
    # MetaDrive + Scenic require the "spawn" start method on macOS and
    # are safer with it on Linux too — fork-after-import can deadlock.
    mp.set_start_method("spawn", force=True)
    main()
