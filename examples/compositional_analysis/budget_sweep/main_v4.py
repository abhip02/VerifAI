"""v4 budget sweep — same 4 specs × 7 scenarios grid as v3, three changes:

1. Configurable time anchor (default 15 min, ``--time-budget`` minutes):
   every scenario is generated under that wall-clock cap (the largest
   budget point), and the convergence sweep runs a 30 s grid up to it.
   Caps are derived from the freshly generated CSVs (each CSV *is*
   exactly one budget's worth), not from historical runs.
2. No trace reuse: the v4 trace store (storage/budget_sweep_v4/traces/)
   is wiped and regenerated on every run — including the three Scenic
   ``Subscenario2*_far`` steer primitives, whose scenario definitions were
   ported into composed_scenarios.scenic from the compositional-analysis
   branch (the storage_paper_steer_fix generation). Like the originals,
   the _far CSVs are raw (steps 0..200, no prewarm trim): the steer
   spec's index-based warmup masks rows 0-24 at evaluation time.
3. Streaming W&B: the run is logged live — generation counts as soon as
   generation finishes, then each cell's convergence series + plot the
   moment that cell completes — instead of one push at the end.

Everything else (grid, specs, engines, ground-truth construction,
ShuffleMain cleaning, plots) is identical to main_v3.
"""

from __future__ import annotations

import csv
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

from verifai.compositional_analysis import (
    CompositionalAnalysisEngine,
    ScenarioBase,
    relabel_traces,
)
from verifai.generate_graph_traces import generate_graph_scenarios

from examples.compositional_analysis.scenic_scenarios.specs import add_dh_column

from .main_v3 import (
    COMP_DIR,
    EXPERIMENTS,
    Cell,
    MD_COMBOS,
    MD_PRIMITIVES,
    SCENIC_FILE,
    SCENIC_MAX_STEPS_MONO_MAIN,
    SCENIC_MAX_STEPS_MONO_SHUF,
    SCENIC_MAX_STEPS_PRIMITIVE,
    SCENIC_MAX_STEPS_SUB2,
    SCENIC_MONO,
    SCENIC_PREWARM_TRIM,
    SCENIC_PRIM_DIRS_DEFAULT,
    SCENIC_PRIM_DIRS_STEER,
    SCENIC_PRIMITIVES,
    SCENIC_SUB2_NAMES,
    SCENIC_WARMUP_STEPS,
    SHUFFLE_BOUNDARY_TRIM,
    SHUFFLEMAIN_CLEAN_DIR,
    SHUFFLEMAIN_DIR,
    _RESULT_FIELDS,
    _count_distinct_traces,
    _error_row,
    _mono_scenario_name,
    _normal_ci_half,
    _report_rho,
    _resolve_cap,
    _scale_caps,
    _scenic_paths,
    _trim_prewarm,
)

# ---------------------------------------------------------------------------
# v4 paths + budget constants
# ---------------------------------------------------------------------------

_V4_SAVE_ROOT = Path("storage/budget_sweep_v4")
# Trace store + filter cache are scoped by generation budget (set in
# _set_run_paths) so e.g. a 15-min and a 30-min sweep can run
# concurrently without wiping each other's traces. Two simultaneous runs
# at the *same* budget would still collide — don't do that.
MD_BASE_V4 = _V4_SAVE_ROOT / "b900s/traces/metadrive"
SCENIC_BASE_V4 = _V4_SAVE_ROOT / "b900s/traces/scenic"
_FILTER_CACHE_V4 = _V4_SAVE_ROOT / "b900s/_filtered_traces"


def _set_run_paths(gen_time_budget: float) -> Path:
    """Point the module's trace-store globals at the per-budget subtree."""
    global MD_BASE_V4, SCENIC_BASE_V4, _FILTER_CACHE_V4
    run_root = _V4_SAVE_ROOT / f"b{int(gen_time_budget)}s"
    MD_BASE_V4 = run_root / "traces/metadrive"
    SCENIC_BASE_V4 = run_root / "traces/scenic"
    _FILTER_CACHE_V4 = run_root / "_filtered_traces"
    return run_root

_CALIB_BUDGET_S = 900.0  # default wall-clock anchor: one 15-min generation run

_FAR_PRIMS = ("Subscenario2L_far", "Subscenario2R_far", "Subscenario2S_far")
# Matches the storage_paper_steer_fix generation: raw 201-row traces, the
# turn landing past the index-warmup window. No prewarm trim afterwards.
SCENIC_MAX_STEPS_FAR = 200
# ShuffleMain respawns the ego per segment, so each of the 4 segments pays
# the Sub2-style prewarm — budget 4 full Sub2 windows.
SCENIC_MAX_STEPS_SHUFFLEMAIN = 4 * SCENIC_MAX_STEPS_SUB2
# The scenario actually simulated for the shuffle ground truth. ShuffleMain
# itself is parser-only (its leaves never terminate, so a `do` chain over
# them can't advance); ShuffleMainExec composes the *Seg leaf variants whose
# behaviors `terminate` at trajectory completion. The generated directory is
# renamed to SHUFFLEMAIN_DIR so everything downstream (cleaning, caps,
# results keying) keeps using the ShuffleMain name.
SHUFFLEMAIN_SCENARIO = "ShuffleMainExec"

# Generation never targets a trace count — the wall-clock budget is the
# stop condition. n is just a ceiling so the worker loop has a bound.
_N_TRACES_CEILING = 100_000


def _filtered_csv(src: Path, max_traces: int | None) -> str:
    """v3's _filtered_csv against the v4 cache root."""
    if not max_traces:
        return str(src)
    import pandas as pd

    _FILTER_CACHE_V4.mkdir(parents=True, exist_ok=True)
    dst = _FILTER_CACHE_V4 / f"{src.parent.name}__top{max_traces}.csv"
    if not dst.is_file() or dst.stat().st_mtime < src.stat().st_mtime:
        df = pd.read_csv(src, on_bad_lines="skip", low_memory=False)
        keep = df["trace_id"].drop_duplicates().head(max_traces)
        df[df["trace_id"].isin(keep)].to_csv(dst, index=False)
    return str(dst)


def _ensure_clean_shufflemain() -> Path:
    """v3's ShuffleMain boundary cleaning against the v4 trace store."""
    src = SCENIC_BASE_V4 / SHUFFLEMAIN_DIR / "traces.csv"
    dst = SCENIC_BASE_V4 / SHUFFLEMAIN_CLEAN_DIR / "traces.csv"
    if dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    import numpy as np
    import pandas as pd

    df = pd.read_csv(src, low_memory=False).sort_values(["trace_id", "step"])
    dx = df.groupby("trace_id")["x"].diff().abs()
    dy = df.groupby("trace_id")["y"].diff().abs()
    spike = ((dx.fillna(0) ** 2 + dy.fillna(0) ** 2) ** 0.5 > 5) | (df["speed"] > 9)
    drop = spike.copy()
    by_trace = spike.groupby(df["trace_id"])
    for i in range(1, SHUFFLE_BOUNDARY_TRIM + 1):
        drop = drop | by_trace.shift(i).fillna(False)
    out = df[~drop].copy()

    def _dh(grp):
        d = grp["heading"].diff().abs()
        d = d.apply(
            lambda v: min(v, 2 * np.pi - v) if (pd.notna(v) and v <= 2 * np.pi) else 0.0
        )
        d[grp["step"].diff() > 1] = 0.0
        return d.fillna(0.0)

    out["dh"] = out.groupby("trace_id", group_keys=False).apply(
        _dh, include_groups=False
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    print(f"[clean] ShuffleMain → {dst} ({len(out)}/{len(df)} rows kept)")
    return dst


# ---------------------------------------------------------------------------
# Fresh generation — always wipe + regenerate (no reuse)
# ---------------------------------------------------------------------------


# MetaDrive expert-policy generation (compositional-analysis branch
# pipeline): S/X/C/O are MetaDrive PG map BLOCKS (Straight, X-intersection,
# Curve, rOundabout) and the combos are multi-block maps, driven by
# MetaDrive's built-in ExpertPolicy via
# examples/compositional_analysis/utils.py::generate_traces. Seeds match
# dfa_tests/test_check_with_dfa_cosafety_fast_twice.py — the generation
# that produced the storage/vshape_speed* stores the paper table used.
_MD_SEEDS = {
    "S": 0, "X": 1, "C": 2, "O": 8,
    "SX": 3, "SXS": 4, "SOC": 10, "CSXS": 11, "CXSXC": 12,
}


def _md_expert_worker(name: str, seed: int, save_dir: str, n: int,
                      time_budget: float) -> str:
    """Generate one MetaDrive expert-policy scenario (runs in a subprocess —
    MetaDrive supports only one engine per process)."""
    import sys as _sys

    comp_dir = str(COMP_DIR)
    if comp_dir not in _sys.path:
        _sys.path.insert(0, comp_dir)  # utils.py does `from train import make_env`
    from utils import generate_traces

    generate_traces(
        seed=seed,
        save_dir=save_dir,
        expert=True,
        n=n,
        scenario=name,
        time_budget=time_budget,
    )
    return name


def _generate_md(combos: set[str], time_budget: float, gen_workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor

    def _run_batch(names: list[str]) -> None:
        workers = max(1, min(gen_workers, len(names)))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [
                pool.submit(
                    _md_expert_worker,
                    name,
                    _MD_SEEDS[name],
                    str(MD_BASE_V4),
                    _N_TRACES_CEILING,
                    time_budget,
                )
                for name in names
            ]
            for f in futs:
                print(f"[gen] MD expert scenario {f.result()} done")

    print(f"[gen] MD primitives {list(MD_PRIMITIVES)} ({time_budget:.0f}s each)")
    _run_batch(list(MD_PRIMITIVES))

    todo = sorted(combos)
    if todo:
        print(f"[gen] MD monoliths {todo} ({time_budget:.0f}s each)")
        _run_batch(todo)


def _generate_scenic(combos: set[str], need_far: bool, time_budget: float) -> None:
    print(
        f"[gen] Scenic primitives {list(SCENIC_PRIMITIVES)} ({time_budget:.0f}s each)"
    )
    generate_graph_scenarios(
        str(SCENIC_FILE),
        list(SCENIC_PRIMITIVES),
        n=_N_TRACES_CEILING,
        save_dir=str(SCENIC_BASE_V4),
        max_steps={
            p: (
                SCENIC_MAX_STEPS_SUB2
                if p in SCENIC_SUB2_NAMES
                else SCENIC_MAX_STEPS_PRIMITIVE
            )
            for p in SCENIC_PRIMITIVES
        },
        time_budget=time_budget,
    )
    for p in SCENIC_PRIMITIVES:
        if p in SCENIC_SUB2_NAMES:
            _trim_prewarm(
                SCENIC_BASE_V4 / p / "traces.csv",
                SCENIC_PREWARM_TRIM,
                step_offset=SCENIC_WARMUP_STEPS,
            )

    monos: list[str] = []
    max_steps: dict[str, int] = {}
    if "choose" in combos:
        monos.append("MonolithicMain")
        max_steps["MonolithicMain"] = SCENIC_MAX_STEPS_MONO_MAIN
    if "shuffle" in combos:
        # The shuffle ground truth is the cleaned ShuffleMain (simulated as
        # ShuffleMainExec, see SHUFFLEMAIN_SCENARIO). "MonolithicShuffle"
        # survives only as the cap-dict key — its scenario is not generated.
        monos.append(SHUFFLEMAIN_SCENARIO)
        max_steps[SHUFFLEMAIN_SCENARIO] = SCENIC_MAX_STEPS_SHUFFLEMAIN
    if monos:
        print(f"[gen] Scenic monoliths {monos} ({time_budget:.0f}s each)")
        generate_graph_scenarios(
            str(SCENIC_FILE),
            monos,
            n=_N_TRACES_CEILING,
            save_dir=str(SCENIC_BASE_V4),
            max_steps=max_steps,
            time_budget=time_budget,
        )
        src_dir = SCENIC_BASE_V4 / SHUFFLEMAIN_SCENARIO
        dst_dir = SCENIC_BASE_V4 / SHUFFLEMAIN_DIR
        if src_dir.is_dir() and not dst_dir.exists():
            src_dir.rename(dst_dir)

    if need_far:
        print(
            f"[gen] Scenic _far steer primitives {list(_FAR_PRIMS)} ({time_budget:.0f}s each)"
        )
        generate_graph_scenarios(
            str(SCENIC_FILE),
            list(_FAR_PRIMS),
            n=_N_TRACES_CEILING,
            save_dir=str(SCENIC_BASE_V4),
            max_steps=SCENIC_MAX_STEPS_FAR,
            time_budget=time_budget,
        )
        # No prewarm trim: the steer spec's index warmup masks rows 0-24.

    add_dh_column(SCENIC_BASE_V4)
    if "shuffle" in combos:
        _ensure_clean_shufflemain()


def generate_fresh_traces(
    cells: list[Cell], time_budget: float, gen_workers: int
) -> None:
    """Wipe the v4 trace store and regenerate everything `cells` needs."""
    shutil.rmtree(MD_BASE_V4, ignore_errors=True)
    shutil.rmtree(SCENIC_BASE_V4, ignore_errors=True)
    shutil.rmtree(_FILTER_CACHE_V4, ignore_errors=True)
    MD_BASE_V4.mkdir(parents=True, exist_ok=True)
    SCENIC_BASE_V4.mkdir(parents=True, exist_ok=True)

    md_combos = {c.combo for c in cells if c.backend == "metadrive"}
    scenic_combos = {c.combo for c in cells if c.backend == "scenic"}
    need_far = any(
        c.backend == "scenic" and c.spec_name == "sustained_steer" for c in cells
    )

    if md_combos:
        _generate_md(md_combos, time_budget, gen_workers)
    if scenic_combos or need_far:
        _generate_scenic(scenic_combos, need_far, time_budget)


def caps_from_fresh_csvs(cells: list[Cell]) -> tuple[dict[str, int], dict[str, int]]:
    """Per-scenario caps = distinct-trace counts of the just-generated CSVs.

    Each CSV holds exactly one ``_CALIB_BUDGET_S`` budget's worth of traces,
    so the full count *is* the anchor; smaller budgets scale linearly.
    Scenic comp caps are keyed by *directory* name (regular and _far pools
    have different per-trace costs, so each scales by its own count).
    """
    comp_caps: dict[str, int] = {}
    mono_caps: dict[str, int] = {}

    if any(c.backend == "metadrive" for c in cells):
        for p in MD_PRIMITIVES:
            comp_caps[p] = _count_distinct_traces(MD_BASE_V4 / p / "traces.csv")
        for combo in {c.combo for c in cells if c.backend == "metadrive"}:
            mono_caps[combo] = _count_distinct_traces(MD_BASE_V4 / combo / "traces.csv")

    scenic_combos = {c.combo for c in cells if c.backend == "scenic"}
    if scenic_combos:
        need_far = any(
            c.backend == "scenic" and c.spec_name == "sustained_steer" for c in cells
        )
        scenic_dirs = list(SCENIC_PRIMITIVES) + (list(_FAR_PRIMS) if need_far else [])
        for p in scenic_dirs:
            comp_caps[p] = _count_distinct_traces(SCENIC_BASE_V4 / p / "traces.csv")
        if "choose" in scenic_combos:
            mono_caps["MonolithicMain"] = _count_distinct_traces(
                SCENIC_BASE_V4 / "MonolithicMain" / "traces.csv"
            )
        if "shuffle" in scenic_combos:
            # The shuffle ground truth is ShuffleMain; key by MonolithicShuffle
            # to match _mono_scenario_name's lookup.
            mono_caps["MonolithicShuffle"] = _count_distinct_traces(
                SCENIC_BASE_V4 / SHUFFLEMAIN_DIR / "traces.csv"
            )

    return comp_caps, mono_caps


# ---------------------------------------------------------------------------
# Engine runners — v3's, pointed at the v4 trace store
# ---------------------------------------------------------------------------


def _run_metadrive(
    spec, spec_name: str, combo: str, max_traces_comp, max_traces_mono
) -> tuple[float, float, float]:
    prim_paths = {
        p: _filtered_csv(
            MD_BASE_V4 / p / "traces.csv", _resolve_cap(p, max_traces_comp)
        )
        for p in MD_PRIMITIVES
    }
    engine = CompositionalAnalysisEngine(ScenarioBase(prim_paths))
    rho_safe_comp, eps = engine.check_with_dfa(
        MD_COMBOS[combo], spec, features=["speed"], center_feat_idx=[]
    )
    rho_safe_mono = relabel_traces(
        _filtered_csv(
            MD_BASE_V4 / combo / "traces.csv", _resolve_cap(combo, max_traces_mono)
        ),
        spec,
    )
    return float(rho_safe_comp), float(eps), float(rho_safe_mono)


def _run_scenic(
    spec, spec_name: str, combo: str, max_traces_comp, max_traces_mono
) -> tuple[float, float, float]:
    prim_dirs = (
        SCENIC_PRIM_DIRS_STEER
        if spec_name == "sustained_steer"
        else SCENIC_PRIM_DIRS_DEFAULT
    )
    logs = {
        # Cap lookup keyed by directory (d), not logical name (p): the
        # _far steer pools scale by their own measured 15-min throughput.
        p: _filtered_csv(
            SCENIC_BASE_V4 / d / "traces.csv", _resolve_cap(d, max_traces_comp)
        )
        for p, d in prim_dirs.items()
    }
    engine = CompositionalAnalysisEngine(ScenarioBase(logs))
    rho_safe_comp, eps = engine.check_with_dfa_scenic(
        _scenic_paths(combo), spec, features=["speed"], center_feat_idx=[]
    )
    mono_name = SCENIC_MONO[combo]
    mono_src = (
        _ensure_clean_shufflemain()
        if combo == "shuffle"
        else SCENIC_BASE_V4 / mono_name / "traces.csv"
    )
    rho_safe_mono = relabel_traces(
        _filtered_csv(mono_src, _resolve_cap(mono_name, max_traces_mono)),
        spec,
    )
    return float(rho_safe_comp), float(eps), float(rho_safe_mono)


_RUNNERS: dict[str, Callable] = {
    "metadrive": _run_metadrive,
    "scenic": _run_scenic,
}


def run_cell_convergence(
    cell: Cell,
    budgets_s,
    comp_caps_calib: dict[str, int],
    mono_caps_calib: dict[str, int],
    calib_budget_s: float = _CALIB_BUDGET_S,
) -> list[dict]:
    """v3's convergence walk with the generation-budget anchor and v4 runners."""
    runner = _RUNNERS[cell.backend]
    spec = cell.spec_factory()
    mono_name = _mono_scenario_name(cell)
    records: list[dict] = []

    for t in budgets_s:
        factor = float(t) / calib_budget_s
        cc = _scale_caps(comp_caps_calib, factor)
        mc = _scale_caps(mono_caps_calib, factor)
        n_mono = _resolve_cap(mono_name, mc)

        rho_safe_c, eps_c, rho_safe_m = runner(spec, cell.spec_name, cell.combo, cc, mc)
        rho_c = _report_rho(cell.spec_name, rho_safe_c)
        rho_m = _report_rho(cell.spec_name, rho_safe_m)
        eps_m = _normal_ci_half(rho_m, n_mono) if n_mono else float("nan")

        records.append(
            {
                "method": "compositional",
                "budget": float(t),
                "rho": float(rho_c),
                "eps": float(eps_c),
                "n_traces": n_mono,
            }
        )
        records.append(
            {
                "method": "monolithic",
                "budget": float(t),
                "rho": float(rho_m),
                "eps": float(eps_m),
                "n_traces": n_mono,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Streaming W&B
# ---------------------------------------------------------------------------

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "verifai-budget-sweep-v4")


def _wandb_start(n_cells: int, budgets_s, gen_time_budget: float):
    """Open the streaming run up front; return the module or None."""
    if os.environ.get("WANDB_DISABLED"):
        print("[wandb] WANDB_DISABLED set; running without streaming")
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] wandb not installed; running without streaming")
        return None

    wandb.init(
        project=WANDB_PROJECT,
        name=f"budget_sweep_v4_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "n_cells": n_cells,
            "calib_budget_s": gen_time_budget,
            "gen_time_budget_s": gen_time_budget,
            "budgets_s": list(budgets_s),
            "trace_reuse": "none",
        },
    )
    # Custom step metric so per-budget series stream out of order across
    # cells without tripping W&B's monotonic-step requirement.
    wandb.define_metric("budget_s")
    wandb.define_metric("conv/*", step_metric="budget_s")
    return wandb


def _wandb_log_cell(wandb, cell: Cell, recs: list[dict], png: Path) -> None:
    for r in recs:
        wandb.log(
            {
                "budget_s": float(r["budget"]),
                f"conv/{cell.name}/{r['method']}/rho": float(r["rho"]),
                f"conv/{cell.name}/{r['method']}/eps": float(r["eps"]),
            }
        )
    payload: dict = {}
    if png.is_file():
        payload[f"plots/{cell.name}"] = wandb.Image(str(png))
    comp_pt = max(
        (r for r in recs if r["method"] == "compositional"), key=lambda r: r["budget"]
    )
    mono_pt = max(
        (r for r in recs if r["method"] == "monolithic"), key=lambda r: r["budget"]
    )
    payload[f"{cell.name}/rho_comp"] = float(comp_pt["rho"])
    payload[f"{cell.name}/rho_mono"] = float(mono_pt["rho"])
    payload[f"{cell.name}/abs_diff"] = abs(comp_pt["rho"] - mono_pt["rho"])
    payload[f"{cell.name}/eps_comp"] = float(comp_pt["eps"])
    wandb.log(payload)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    *,
    only: str | None = None,
    list_only: bool = False,
    budgets_s: tuple[float, ...] | None = None,
    gen_time_budget: float = _CALIB_BUDGET_S,
    gen_workers: int = 5,
) -> None:
    # Default sweep grid: 30 s steps up to the generation budget (which is
    # also the calibration anchor), final point pinned to the full budget.
    if budgets_s is None:
        budgets_s = tuple(float(t) for t in range(30, int(gen_time_budget) + 1, 30))
        if not budgets_s or budgets_s[-1] != gen_time_budget:
            budgets_s = budgets_s + (float(gen_time_budget),)

    cells = EXPERIMENTS
    if only:
        cells = [c for c in cells if only in c.name]

    if list_only:
        print(f"{len(cells)} cells:")
        for c in cells:
            print(f"  {c.name}")
        return

    _set_run_paths(gen_time_budget)
    stamp = f"b{int(gen_time_budget)}s_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _V4_SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    out_csv = _V4_SAVE_ROOT / f"results_{stamp}.csv"
    per_cell_dir = _V4_SAVE_ROOT / f"convergence_{stamp}"
    per_cell_dir.mkdir(parents=True, exist_ok=True)

    wandb = _wandb_start(len(cells), budgets_s, gen_time_budget)

    print(f"[main_v4] generating fresh traces ({gen_time_budget:.0f}s per scenario)")
    generate_fresh_traces(cells, gen_time_budget, gen_workers)

    comp_caps, mono_caps = caps_from_fresh_csvs(cells)
    print(f"[calibration] fresh {gen_time_budget / 60:.0f}-min trace counts:")
    print(f"  comp = {comp_caps}")
    print(f"  mono = {mono_caps}")
    if wandb:
        wandb.log(
            {f"gen/comp/{k}": v for k, v in comp_caps.items()}
            | {f"gen/mono/{k}": v for k, v in mono_caps.items()}
        )

    from .plots import plot_rho_vs_budget

    rows: list[dict] = []
    print(f"[main_v4] running {len(cells)} cells → {out_csv}")

    for cell in cells:
        print(f"\n{'=' * 70}\nCELL: {cell.name}\n{'=' * 70}")
        try:
            recs = run_cell_convergence(
                cell, budgets_s, comp_caps, mono_caps,
                calib_budget_s=gen_time_budget,
            )
        except Exception as exc:
            print(f"[{cell.name}] ERROR: {exc}", file=sys.stderr)
            rows.append(_error_row(cell, exc))
            continue

        cell_csv = per_cell_dir / f"{cell.name}.csv"
        with cell_csv.open("w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["method", "budget", "rho", "eps", "n_traces"]
            )
            w.writeheader()
            w.writerows(recs)
        png = per_cell_dir / f"{cell.name}.png"
        try:
            plot_rho_vs_budget(recs, png)
        except Exception as exc:
            print(f"[{cell.name}] plot failed: {exc}", file=sys.stderr)

        comp_pt = max(
            (r for r in recs if r["method"] == "compositional"),
            key=lambda r: r["budget"],
        )
        mono_pt = max(
            (r for r in recs if r["method"] == "monolithic"),
            key=lambda r: r["budget"],
        )
        rows.append(
            {
                "cell": cell.name,
                "backend": cell.backend,
                "spec": cell.spec_name,
                "combo": cell.combo,
                "rho_comp": comp_pt["rho"],
                "rho_mono": mono_pt["rho"],
                "abs_diff": abs(comp_pt["rho"] - mono_pt["rho"]),
                "eps_comp": comp_pt["eps"],
                "elapsed_s": None,
                "error": "",
            }
        )
        print(
            f"[{cell.name}] @t={comp_pt['budget']:.0f}s: "
            f"rho_comp={comp_pt['rho']:.3f}±{comp_pt['eps']:.3f}  "
            f"rho_mono={mono_pt['rho']:.3f}±{mono_pt['eps']:.3f}"
        )

        if wandb:
            try:
                _wandb_log_cell(wandb, cell, recs, png)
            except Exception as exc:
                print(f"[{cell.name}] wandb stream failed: {exc}", file=sys.stderr)

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_RESULT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[main_v4] wrote {len(rows)} summary rows → {out_csv}")
    print(f"[main_v4] convergence CSVs + plots → {per_cell_dir}")

    # Final image bundle: one folder with every cell plot, named by the
    # budget marker (plots_15min/, plots_30min/, ...) — the drop-in source
    # for the paper's figures/time-budget/ directory.
    minutes = int(round(gen_time_budget / 60))
    plots_dir = _V4_SAVE_ROOT / f"plots_{minutes}min"
    shutil.rmtree(plots_dir, ignore_errors=True)
    plots_dir.mkdir(parents=True)
    pngs = sorted(per_cell_dir.glob("*.png"))
    for png in pngs:
        shutil.copy2(png, plots_dir / png.name)
    print(f"[main_v4] {len(pngs)} plots bundled → {plots_dir}")

    if wandb:
        table = wandb.Table(columns=_RESULT_FIELDS)
        for r in rows:
            table.add_data(*[r.get(k) for k in _RESULT_FIELDS])
        wandb.log({"results_table": table})

        # Image gallery keyed by the budget marker, then the bundle as a
        # named artifact (download with: wandb artifact get
        # budget_sweep_v4_plots_15min).
        wandb.log(
            {
                f"final_plots_{minutes}min/{png.stem}": wandb.Image(str(png))
                for png in pngs
            }
        )
        art = wandb.Artifact(f"budget_sweep_v4_plots_{minutes}min", type="plots")
        art.add_dir(str(plots_dir))
        wandb.log_artifact(art)

        art = wandb.Artifact("budget_sweep_v4_results", type="results")
        art.add_file(str(out_csv))
        for f in per_cell_dir.iterdir():
            if f.is_file():
                art.add_file(str(f))
        wandb.log_artifact(art)
        wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "v4 budget sweep: generate every scenario fresh under a "
            "per-scenario wall-clock cap, then run the 4x7 convergence "
            "grid with the cap as the largest budget point."
        )
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=_CALIB_BUDGET_S / 60.0,
        metavar="MINUTES",
        help=(
            "Per-scenario generation time limit in minutes; also the "
            "calibration anchor and the largest budget point of the "
            "sweep (default: %(default).0f)"
        ),
    )
    parser.add_argument(
        "--only",
        default=None,
        metavar="SUBSTR",
        help="Only run cells whose name contains SUBSTR "
        "(e.g. 'metadrive' or 'tollgate'); default: all 28 cells",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="Print the selected cell names and exit",
    )
    parser.add_argument(
        "--gen-workers",
        type=int,
        default=5,
        metavar="N",
        help="Parallel monolith generations (default: %(default)s)",
    )
    args = parser.parse_args()

    main(
        only=args.only,
        list_only=args.list_only,
        gen_time_budget=args.time_budget * 60.0,
        gen_workers=args.gen_workers,
    )
