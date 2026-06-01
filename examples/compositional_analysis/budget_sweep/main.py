from __future__ import annotations

import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path

from . import checks
from .config import Record, SweepConfig
from .plots import PLOT_FILES, render_plots
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


# Non-markovian DFA factories from the v1 comparison script. Only the
# count/consecutive specs are exercised here; the markovian
# ``safe_under_max`` / ``default_spec`` experiments have been removed.
from examples.compositional_analysis.compare_budget_sweep import (  # noqa: E402
    spec_at_most_k_brake,
)


# ---------------------------------------------------------------------------
# Experiments to run, sequentially, one per :class:`BudgetSweep` instance.
# Add or comment out entries to change what ``main()`` executes. Each entry
# is ``(name, cfg)``; results land in ``cfg.save_dir / name``.
# ---------------------------------------------------------------------------

_V2_SAVE_ROOT = Path("storage/budget_sweep_v2")

# Shared kwargs for the N=5 wander_scenarios setup. All four non-markovian
# DFAs below reuse it; useful comp length = 5 × (75 − 35) = 200 = max_steps_mono.
_WANDER_SCEN_KW: dict[str, object] = dict(
    scenic_file=SCENIC_DIR / "wander_scenarios.scenic",
    composite_name="Main",
    monolithic_name="MonolithicWander",
    max_budget=1800.0,
    snapshot_every=30.0,
    max_steps_primitive=75,
    max_steps_mono=200,
    features=["speed"],
    center_feat_idx=[],
    delta=0.05,
    prewarm_trim={
        p: 35
        for p in (
            "BrakeScenario",
            "GoStraightScenario",
            "TurnLeftScenario",
            "TurnRightScenario",
        )
    },
    save_dir=_V2_SAVE_ROOT,
)


EXPERIMENTS: list[tuple[str, SweepConfig]] = [
    # --- wander_scenarios under two non-markovian brake-episode specs ---
    # K tuned so mono ρ̂ lands in (0.05, 0.95). At N=5 segments the brake-
    # episode count is bounded by ⌊5/2⌋=2, so K∈{2,3,...} is a deterministic
    # accept; only K∈{0, 1} actually discriminate.
    (
        "wander_no_brake",
        SweepConfig(spec=spec_at_most_k_brake(0), **_WANDER_SCEN_KW),
    ),
    (
        "wander_at_most_one_brake",
        SweepConfig(spec=spec_at_most_k_brake(1), **_WANDER_SCEN_KW),
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

    # 2. Per-experiment results.csv + figures + (T, ρ̂, ε̂) line series.
    for name, (csv_path, records) in results_per_experiment.items():
        if csv_path.exists():
            art = wandb.Artifact(f"{name}_results", type="results")
            art.add_file(str(csv_path))
            wandb.log_artifact(art)

        plots_dir = csv_path.parent / "plots"
        if plots_dir.exists():
            images = {
                f"{name}/{key}": wandb.Image(str(plots_dir / fname))
                for key, fname in PLOT_FILES
                if (plots_dir / fname).exists()
            }
            if images:
                wandb.log(images)

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
        plots_dir = sweep.csv_path.parent / "plots"
        try:
            written = render_plots(records, plots_dir)
            print(f"[{name}] rendered {len(written)} plots in {plots_dir}/")
        except (
            Exception
        ) as exc:  # pragma: no cover — plotting must never abort the sweep
            checks._log.warning("[%s] plot rendering failed: %s", name, exc)
        results_per_experiment[name] = (sweep.csv_path, records)

    _push_to_wandb(results_per_experiment)


if __name__ == "__main__":
    # MetaDrive + Scenic require the "spawn" start method on macOS and
    # are safer with it on Linux too — fork-after-import can deadlock.
    mp.set_start_method("spawn", force=True)
    main()
