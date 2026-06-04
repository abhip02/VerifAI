"""v3 budget sweep — MetaDrive only, curated 3-cell experiment list.

Active EXPERIMENTS is a focused subset of 3 (spec, composite) pairs
chosen from the 4×7=28 cross-product based on smoke evidence in
PRELIM_RESULTS.md. Webots is deferred — its scenic backend requires
running inside a Webots Supervisor controller and is incompatible
with the BudgetSweep worker model; see PRELIM_RESULTS.md §"Open
issues" #3 for the integration scope.

Sibling of main.py (which holds the v2 wander record); intentionally
duplicates the runner/W&B-push wiring so the two sweeps stay
independent.
"""

from __future__ import annotations

import multiprocessing as mp
import os
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
    ("composites/seq_SX.scenic", "Main", "MonoSX", 40, 80),
    ("composites/seq_SXS.scenic", "Main", "MonoSXS", 40, 120),
    ("composites/seq_SOC.scenic", "Main", "MonoSOC", 40, 120),
    ("composites/seq_CSXS.scenic", "Main", "MonoCSXS", 40, 160),
    ("composites/seq_CXSXC.scenic", "Main", "MonoCXSXC", 40, 200),
    ("composites/native_choose.scenic", "Main", "MonoSChooseCXO", 40, 80),
    ("composites/native_shuffle.scenic", "Main", "MonoSShuffleCXO", 40, 160),
]

# MetaDrive-only assignments. The Markovian baseline (spec_max_speed) is
# expected to show ρ̂_comp ≈ ρ̂_mono across all composites — a gap there
# would indicate a pipeline bug; agreement there validates the rest of
# the methodology.
ASSIGNMENTS: list[tuple[str, str, callable]] = [
    ("metadrive", "tollgate", spec_tollgate),
    ("metadrive", "two_stops", spec_two_stops),
    ("metadrive", "fast_twice", spec_fast_twice),
    ("metadrive", "max_speed", spec_max_speed),
]


def _build_experiments() -> list[tuple[str, SweepConfig]]:
    out: list[tuple[str, SweepConfig]] = []
    for backend, spec_name, spec_factory in ASSIGNMENTS:
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
        return base[len("seq_") :]
    if base.startswith("native_"):
        return base[len("native_") :]
    return base


# ---------------------------------------------------------------------------
# Curated cells — the 2 best non-Markovian + 1 Markovian-baseline (spec,
# composite) pairs identified by smoke testing (see PRELIM_RESULTS.md).
# These three are the only cells we have empirical evidence land in the
# discriminating (0.1, 0.9) band on at least one method, so they're the
# right targets for the fixed-time-budget figure. Use these instead of
# the full 28-cell cross-product (built by _build_experiments) to keep
# the unattended sweep focused (~3 CPU-h instead of 14).
# ---------------------------------------------------------------------------


def _build_curated_experiments() -> list[tuple[str, SweepConfig]]:
    cells: list[tuple[str, str, str, str, str, int]] = [
        # (cell-name suffix, spec_name, scenic_file, composite, mono_name, mono_steps)
        # ★ Headline non-Markovian × sequential composition: handoff-state
        # stitching exercised. comp=0.667, mono=0.600, |Δρ̂|=0.07 at T=180s.
        # (
        #    "fast_twice__CSXS",
        #    "fast_twice",
        #    "composites/seq_CSXS.scenic",
        #    "Main",
        #    "MonoCSXS",
        #    160,
        # ),
        # ★ Native `do choose` — exercises path-weighted reuse across 3 branches.
        # Paired with Markovian max_speed because fast_twice/tollgate saturate
        # structurally on a 2-segment S→one-of-{C,X,O} trace (no fast→slow→fast
        # possible; S already satisfies tollgate K=1). max_speed lands in (0,1)
        # because X∈[7,10] and O∈[6.5,9] sometimes exceed the 8.5 m/s threshold.
        (
            "max_speed__choose",
            "max_speed",
            "composites/native_choose.scenic",
            "Main",
            "MonoSChooseCXO",
            80,
        ),
        # ★ Native `do shuffle` — exercises path-weighted reuse across the 6
        # permutations of {C, X, O}. Same Markovian spec for the same
        # saturation reason; 4-segment trace amplifies the comp speedup over
        # mono (mono must draw a full S+perm trajectory per sample).
        (
            "max_speed__shuffle",
            "max_speed",
            "composites/native_shuffle.scenic",
            "Main",
            "MonoSShuffleCXO",
            160,
        ),
        # ★ Non-Markovian × `do choose`. Structurally saturated: only one
        # follow-on segment, so the fast→slow→fast pattern is impossible
        # and ρ̂≈0 on both methods. Kept as a pipeline-agreement check —
        # if comp and mono disagree here, something is wrong with the
        # comp branch-weighting because mono can only produce ρ̂=0.
        (
            "fast_twice__choose",
            "fast_twice",
            "composites/native_choose.scenic",
            "Main",
            "MonoSChooseCXO",
            80,
        ),
        # ★ Non-Markovian × `do shuffle` — the real second non-Markovian
        # cell. Of the 6 perms of {C,X,O}, exactly X→C→O and O→C→X
        # produce fast→slow→fast, so the spec is non-trivially
        # satisfied with nominal weight 2/6 ≈ 0.33. Discriminating both
        # on the branching structure and on the temporal pattern.
        (
            "fast_twice__shuffle",
            "fast_twice",
            "composites/native_shuffle.scenic",
            "Main",
            "MonoSShuffleCXO",
            160,
        ),
    ]
    spec_map = {
        "tollgate": spec_tollgate,
        "two_stops": spec_two_stops,
        "fast_twice": spec_fast_twice,
        "max_speed": spec_max_speed,
    }
    out: list[tuple[str, SweepConfig]] = []
    for suffix, spec_name, file_rel, comp_name, mono_name, mono_steps in cells:
        backend = "metadrive"  # all curated cells are metadrive
        name = f"{backend}__{suffix}"
        cfg = SweepConfig(
            spec=spec_map[spec_name](),
            **_kw(
                SCEN_DIR / backend / file_rel,
                comp_name,
                mono_name,
                max_steps_primitive=40,
                max_steps_mono=mono_steps,
            ),
        )
        out.append((name, cfg))
    return out


# Active list run by main(). Swap to `_build_experiments()` for the full
# 4×7=28 cross-product (mostly saturated-by-construction; see PRELIM_RESULTS.md).
EXPERIMENTS: list[tuple[str, SweepConfig]] = _build_curated_experiments()

# Full cross-product kept available but inactive (call _build_experiments()
# at module level to populate). The 28 cells include many saturated-by-
# construction cells (e.g. fast_twice__CXSXC is always-violated, structural).
ALL_EXPERIMENTS: list[tuple[str, SweepConfig]] = _build_experiments()


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
