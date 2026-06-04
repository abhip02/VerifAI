"""v3 budget sweep — 4 specs × 7 composites = 28 cells on the new
Scenic v3 scenario family (see SCENIC_SCENARIOS.md §6).

3 specs run on MetaDrive (Town07); the 1 Webots spec row is skipped
when the Webots binary is not on PATH — per SCENIC_SCENARIOS.md §8
risk #4 and CLAUDE.md's guardrails.

Sibling of main.py (which holds the v2 wander record); intentionally
duplicates the runner/W&B-push wiring so the two sweeps stay
independent.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
from datetime import datetime
from pathlib import Path

from . import checks
from .config import Record, SweepConfig
from .plots import PLOT_FILES, render_plots
from .sweep import BudgetSweep

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "verifai-budget-sweep-v3")

# REPO_ROOT = .../VerifAI
REPO_ROOT = Path(__file__).resolve().parents[3]
SCEN_DIR = REPO_ROOT / "examples/compositional_analysis/scenic_scenarios"

from examples.compositional_analysis.scenic_scenarios.specs import (  # noqa: E402
    spec_tollgate,
    spec_two_stops,
    spec_fast_twice,
    spec_slow2_accel,
    spec_max_speed,
)


_V3_SAVE_ROOT = Path("storage/budget_sweep_v3")


# Per-primitive prewarm trim: drop the first N rows of each leaf trace so
# the analyzer sees only the steady-state segment. Empirically (40-tick
# trace, PID + with-speed-Range(0,10) spawn): X needs ~12 ticks to settle
# at 9 m/s, O ~12 to settle at 8, C ~8 to settle at 5, S ~5 to settle at
# ~0. Without this the first segment of each primitive trace is a
# slow→fast ramp that pollutes the DFA labelling at every handoff.
PREWARM_TRIM = {"S": 5, "X": 12, "C": 8, "O": 12}


def _kw(
    scenic_file: Path,
    composite: str,
    monolithic: str,
    max_steps_primitive: int,
    max_steps_mono: int,
    max_budget: float = 1800.0,
    snapshot_every: float = 30.0,
) -> dict:
    return dict(
        scenic_file=scenic_file,
        composite_name=composite,
        monolithic_name=monolithic,
        max_budget=max_budget,
        snapshot_every=snapshot_every,
        max_steps_primitive=max_steps_primitive,
        max_steps_mono=max_steps_mono,
        features=["x", "y", "speed"],
        center_feat_idx=[0, 1],
        delta=0.05,
        prewarm_trim=dict(PREWARM_TRIM),
        save_dir=_V3_SAVE_ROOT,
    )


# (composite-file relpath within a backend subtree, composite name, mono name,
#  per-primitive max_steps, monolithic max_steps)
COMPOSITES: list[tuple[str, str, str, int, int]] = [
    ("composites/seq_SX.scenic",         "Main", "MonoSX",          40,  80),
    ("composites/seq_SXS.scenic",        "Main", "MonoSXS",         40, 120),
    ("composites/seq_SOC.scenic",        "Main", "MonoSOC",         40, 120),
    ("composites/seq_CSXS.scenic",       "Main", "MonoCSXS",        40, 160),
    ("composites/seq_CXSXC.scenic",      "Main", "MonoCXSXC",       40, 200),
    ("composites/native_choose.scenic",  "Main", "MonoSChooseCXO",  40,  80),
    ("composites/native_shuffle.scenic", "Main", "MonoSShuffleCXO", 40, 160),
]

# 3 non-Markovian specs on MetaDrive + 1 Markovian agreement-baseline spec
# on MetaDrive + 1 non-Markovian spec on Webots → 5×7 = 35 cells total.
# The Markovian baseline (spec_max_speed) is expected to show
# ρ̂_comp ≈ ρ̂_mono across all 7 composites — a gap there would indicate
# a pipeline bug; agreement there validates the rest of the methodology.
ASSIGNMENTS: list[tuple[str, str, callable]] = [
    ("metadrive", "tollgate",    spec_tollgate),
    ("metadrive", "two_stops",   spec_two_stops),
    ("metadrive", "fast_twice",  spec_fast_twice),
    ("metadrive", "max_speed",   spec_max_speed),
    ("webots",    "slow2_accel", spec_slow2_accel),
]


def _build_experiments() -> list[tuple[str, SweepConfig]]:
    have_webots = shutil.which("webots") is not None
    out: list[tuple[str, SweepConfig]] = []
    for backend, spec_name, spec_factory in ASSIGNMENTS:
        if backend == "webots" and not have_webots:
            print(f"[main_v3] skipping webots row '{spec_name}': "
                  "`webots` binary not on PATH")
            continue
        for file_rel, comp_name, mono_name, prim_steps, mono_steps in COMPOSITES:
            name = f"{backend}__{spec_name}__{_short_comp_name(file_rel)}"
            cfg = SweepConfig(
                spec=spec_factory(),
                **_kw(
                    SCEN_DIR / backend / file_rel,
                    comp_name,
                    mono_name,
                    prim_steps,
                    mono_steps,
                ),
            )
            out.append((name, cfg))
    return out


def _short_comp_name(file_rel: str) -> str:
    base = Path(file_rel).stem  # e.g. seq_SX or native_choose
    if base.startswith("seq_"):
        return base[len("seq_"):]
    if base.startswith("native_"):
        return base[len("native_"):]
    return base


EXPERIMENTS: list[tuple[str, SweepConfig]] = _build_experiments()


def _push_to_wandb(
    results_per_experiment: dict[str, tuple[Path, list[Record]]],
) -> None:
    if os.environ.get("WANDB_DISABLED"):
        print("[wandb] WANDB_DISABLED set; skipping upload")
        return
    try:
        import wandb
    except ImportError:
        print("[wandb] wandb not installed; skipping upload")
        return

    run_name = f"budget_sweep_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project=WANDB_PROJECT, name=run_name, reinit=True)

    log_path = checks._LOG_PATH
    if log_path.exists():
        art = wandb.Artifact("budget_sweep_checks_log_v3", type="log")
        art.add_file(str(log_path))
        wandb.log_artifact(art)
        wandb.save(str(log_path), base_path=str(log_path.parent), policy="now")

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
    seen: set[str] = set()
    results: dict[str, tuple[Path, list[Record]]] = {}
    print(f"[main_v3] running {len(EXPERIMENTS)} cells")
    for name, cfg in EXPERIMENTS:
        if not name or name in seen:
            checks._log.warning("duplicate or empty experiment name: %r", name)
        seen.add(name)
        cfg.save_dir = cfg.save_dir / name
        print(f"\n{'=' * 70}\nEXPERIMENT: {name}\n{'=' * 70}")
        sweep = BudgetSweep(cfg)
        records = sweep.run()
        print(f"[{name}] wrote {len(records)} records to {sweep.csv_path}")
        plots_dir = sweep.csv_path.parent / "plots"
        try:
            written = render_plots(records, plots_dir)
            print(f"[{name}] rendered {len(written)} plots in {plots_dir}/")
        except Exception as exc:  # pragma: no cover
            checks._log.warning("[%s] plot rendering failed: %s", name, exc)
        results[name] = (sweep.csv_path, records)

    _push_to_wandb(results)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
