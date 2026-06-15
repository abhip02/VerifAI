"""Render the comp-eps=0 presentation variant of existing results.

The partner wants every result in two presentations:
  1. compositional epsilon = 0 (bare point estimate; monolithic eps normal)
  2. compositional epsilon as computed by the engine

The epsilon computation does not affect rho, so both variants come from a
single analysis run: this script post-processes a results directory and
emits a sibling `<dir>_eps0/` with the compositional eps zeroed in every
cell CSV and the plots re-rendered. Monolithic records are untouched.

Handles:
  - convergence_b<N>s_<stamp>/   (main output: per-cell CSV + PNG;
        also writes a plots_<minutes>min_eps0/ bundle next to the
        existing plots_<minutes>min/)
  - ablation_compute_matched/    (cell curve CSVs + table.csv; the table
        variant zeroes the compositional eps columns and recomputes the
        agreement flag against the monolithic bound alone)

Usage:
  python -m examples.compositional_analysis.budget_sweep.render_eps_variants \
      [DIR ...]
  With no arguments, auto-discovers the latest convergence_b900s_*,
  convergence_b3600s_*, and ablation_compute_matched under
  storage/budget_sweep_v4/.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "storage/budget_sweep_v4"

_CONV_COLS = {"method", "budget", "rho", "eps"}


def _is_cell_csv(path: Path) -> bool:
    try:
        head = pd.read_csv(path, nrows=1)
    except Exception:
        return False
    return _CONV_COLS.issubset(head.columns)


def _render_cell(src_csv: Path, out_dir: Path) -> None:
    from examples.compositional_analysis.budget_sweep.plots import (
        cell_title,
        legend_loc_for,
        plot_rho_vs_budget,
    )

    df = pd.read_csv(src_csv)
    df.loc[df.method == "compositional", "eps"] = 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / src_csv.name
    df.to_csv(out_csv, index=False)
    try:
        plot_rho_vs_budget(df.to_dict("records"), out_dir / f"{src_csv.stem}.png",
                           title=cell_title(src_csv.stem),
                           legend_loc=legend_loc_for(src_csv.stem))
    except Exception as exc:
        print(f"  [warn] plot failed for {src_csv.stem}: {exc}", file=sys.stderr)


def _render_ablation_table(src_csv: Path, out_dir: Path) -> None:
    t = pd.read_csv(src_csv)
    for col in ("eps_comp_full", "eps_comp_matched"):
        if col in t.columns:
            t[col] = 0.0
    if {"rho_comp_matched", "rho_mono", "eps_mono"}.issubset(t.columns):
        t["agree_matched"] = (
            (t.rho_comp_matched - t.rho_mono).abs() <= t.eps_mono
        )
    if {"rho_comp_full", "rho_mono", "eps_mono"}.issubset(t.columns):
        t["agree_full"] = (t.rho_comp_full - t.rho_mono).abs() <= t.eps_mono
    out_dir.mkdir(parents=True, exist_ok=True)
    t.to_csv(out_dir / src_csv.name, index=False)


def render_dir(src: Path) -> Path:
    out = src.parent / f"{src.name}_eps0"
    cells = 0
    for csv in sorted(src.glob("*.csv")):
        if csv.name == "table.csv":
            _render_ablation_table(csv, out)
            print(f"  table.csv -> {out / 'table.csv'} (comp eps zeroed, "
                  f"agreement vs mono bound only)")
            continue
        if _is_cell_csv(csv):
            _render_cell(csv, out)
            cells += 1
    print(f"[{src.name}] {cells} cells -> {out}")

    # plots bundle for main convergence dirs (plots_<minutes>min_eps0)
    m = re.match(r"convergence_b(\d+)s_", src.name)
    if m and cells:
        minutes = int(round(int(m.group(1)) / 60))
        bundle = src.parent / f"plots_{minutes}min_eps0"
        bundle.mkdir(exist_ok=True)
        import shutil

        for png in sorted(out.glob("*.png")):
            shutil.copy2(png, bundle / png.name)
        print(f"[{src.name}] plots bundle -> {bundle}")
    return out


def _latest(pattern: str) -> Path | None:
    dirs = sorted(ROOT.glob(pattern), key=lambda p: p.name)
    return dirs[-1] if dirs else None


def main(args: list[str]) -> None:
    if args:
        targets = [Path(a).resolve() for a in args]
    else:
        targets = [
            d for d in (
                _latest("convergence_b900s_*"),
                _latest("convergence_b3600s_*"),
                ROOT / "ablation_compute_matched",
            )
            if d is not None and d.is_dir()
        ]
    if not targets:
        raise SystemExit("no result directories found")
    for t in targets:
        if not t.is_dir():
            raise SystemExit(f"not a directory: {t}")
        render_dir(t)


if __name__ == "__main__":
    main(sys.argv[1:])
