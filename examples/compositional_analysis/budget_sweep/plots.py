"""Matplotlib figure generation for the v2 budget sweep.

Ported verbatim from ``compare_budget_sweep.py`` (the v1 script) so the
paper figures continue to render from v2 results without changes to the
v2 methodology. Reads :class:`~budget_sweep.config.Record` rows produced
by :class:`~budget_sweep.sweep.BudgetSweep` and writes PNGs into a
``plots/`` subdirectory.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import Record

METHOD_MONO = "monolithic"
METHOD_COMP = "compositional"


PLOT_FILES = [
    ("wallclock", "wallclock.png"),
    ("eps_vs_budget", "eps_vs_budget.png"),
    ("rho_vs_budget", "rho_vs_budget.png"),
    ("throughput", "throughput.png"),
    ("speedup_vs_budget", "speedup_vs_budget.png"),
]


def _series(records, key):
    out = defaultdict(list)
    for r in records:
        v = r.get(key)
        if v is None:
            continue
        out[r["method"]].append((float(r["budget"]), float(v)))
    for m in out:
        out[m].sort()
    return out


def plot_eps_vs_budget(records, out_path):
    series = _series(records, "eps")
    if not series:
        print("[plot eps] no points; skipping")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, pts in series.items():
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=".", linewidth=1.2, label=m)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time budget (s)")
    ax.set_ylabel("eps (Hoeffding CI half-width)")
    ax.set_title("Uncertainty vs. budget")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_rho_vs_budget(records, out_path):
    by = defaultdict(list)
    for r in records:
        if r.get("rho") is None or r.get("eps") is None:
            continue
        by[r["method"]].append((float(r["budget"]), float(r["rho"]), float(r["eps"])))
    for m in by:
        by[m].sort()
    if not by:
        print("[plot rho] no points; skipping")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, pts in by.items():
        xs = [p[0] for p in pts]
        rho = np.array([p[1] for p in pts])
        eps = np.array([p[2] for p in pts])
        ax.plot(xs, rho, marker=".", linewidth=1.2, label=m)
        ax.fill_between(xs, rho - eps, rho + eps, alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("time budget (s)")
    ax.set_ylabel("rho ± eps")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Estimate convergence")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_throughput(records, out_path):
    series = _series(records, "n_full_traces")
    if not series:
        print("[plot throughput] no points; skipping")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, pts in series.items():
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=".", linewidth=1.2, label=m)
    ax.set_xscale("log")
    ax.set_yscale("symlog")
    ax.set_xlabel("time budget (s)")
    ax.set_ylabel("# full-episode-equivalent traces")
    ax.set_title("Trace throughput vs. budget")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_wallclock_combo(records, out_path):
    """Side-by-side figure for the paper (fig:wallclock):
    left panel is ``eps`` vs.~``T`` on log-log axes, right panel is
    ``rho`` with a shaded ``+/- eps`` band on semi-log-x axes.
    """
    eps_series = _series(records, "eps")

    rho_by = defaultdict(list)
    for r in records:
        if r.get("rho") is None or r.get("eps") is None:
            continue
        rho_by[r["method"]].append(
            (float(r["budget"]), float(r["rho"]), float(r["eps"]))
        )
    for m in rho_by:
        rho_by[m].sort()

    if not eps_series and not rho_by:
        print("[plot wallclock_combo] no points; skipping")
        return

    fig, (ax_eps, ax_rho) = plt.subplots(1, 2, figsize=(12, 4.5))

    for m, pts in eps_series.items():
        xs, ys = zip(*pts)
        ax_eps.plot(xs, ys, marker=".", linewidth=1.2, label=m)
    ax_eps.set_xscale("log")
    ax_eps.set_yscale("log")
    ax_eps.set_xlabel("time budget (s)")
    ax_eps.set_ylabel(r"$\hat\varepsilon$ (Hoeffding CI half-width)")
    ax_eps.set_title(r"Uncertainty vs. budget")
    ax_eps.grid(True, alpha=0.3)
    ax_eps.legend()

    for m, pts in rho_by.items():
        xs = [p[0] for p in pts]
        rho = np.array([p[1] for p in pts])
        eps = np.array([p[2] for p in pts])
        ax_rho.plot(xs, rho, marker=".", linewidth=1.2, label=m)
        ax_rho.fill_between(xs, rho - eps, rho + eps, alpha=0.2)
    ax_rho.set_xscale("log")
    ax_rho.set_xlabel("time budget (s)")
    ax_rho.set_ylabel(r"$\hat\rho \pm \hat\varepsilon$")
    ax_rho.set_ylim(0.0, 1.0)
    ax_rho.set_title(r"Estimate convergence")
    ax_rho.grid(True, alpha=0.3)
    ax_rho.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_speedup_vs_budget(records, out_path):
    """eps_mono / eps_comp at matched budget. >1 means compositional wins.

    Snapshot times are not exactly aligned between methods; bucket budgets
    to the nearest log-spaced bin and pair within bin.
    """
    rows = [r for r in records if r.get("eps") is not None and r["eps"] > 0]
    if not rows:
        print("[plot speedup] no points; skipping")
        return

    budgets = sorted({r["budget"] for r in rows})
    if len(budgets) < 2:
        print("[plot speedup] need >=2 budgets; skipping")
        return

    mono = sorted(
        [r for r in rows if r["method"] == METHOD_MONO], key=lambda r: r["budget"]
    )
    comp = sorted(
        [r for r in rows if r["method"] == METHOD_COMP], key=lambda r: r["budget"]
    )
    pairs = []
    for rm in mono:
        candidates = [
            rc
            for rc in comp
            if abs(rc["budget"] - rm["budget"]) / max(rm["budget"], 1e-9) < 0.2
        ]
        if not candidates:
            continue
        rc = min(candidates, key=lambda c: abs(c["budget"] - rm["budget"]))
        T = 0.5 * (rm["budget"] + rc["budget"])
        pairs.append((T, rm["eps"] / rc["eps"]))

    if not pairs:
        print("[plot speedup] no matched (mono, comp) pairs; skipping")
        return

    xs, ys = zip(*pairs)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, ys, marker=".", linewidth=1.2, color="#3c5b8a")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("time budget (s)")
    ax.set_ylabel("eps_mono / eps_comp")
    ax.set_title("Compositional precision advantage at matched budget")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _coerce_records(records: list[Record]) -> list[Record]:
    """Normalize numeric fields the plotters expect to be floats/None.

    v2 CSVs don't carry ``n_full_traces`` (the v1 throughput
    normalization column); fall back to ``n_traces`` so the throughput
    plot still renders, mirroring v1's ``load_records`` behavior.
    """
    out = []
    for row in records:
        r = dict(row)
        for k in ("budget", "rho", "eps"):
            v = r.get(k)
            if v in (None, "", "None"):
                r[k] = None
            else:
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    r[k] = None
        try:
            r["n_traces"] = int(r.get("n_traces") or 0)
        except (TypeError, ValueError):
            r["n_traces"] = 0
        nft = r.get("n_full_traces")
        if nft in (None, "", "None"):
            r["n_full_traces"] = float(r["n_traces"])
        else:
            try:
                r["n_full_traces"] = float(nft)
            except (TypeError, ValueError):
                r["n_full_traces"] = float(r["n_traces"])
        out.append(r)
    return out


def render_plots(records: list[Record], plots_dir: Path) -> dict[str, Path]:
    """Render all paper figures; return ``{key: path}`` for whatever
    was actually written (plotters skip when no data is available).
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    recs = _coerce_records(records)
    plot_eps_vs_budget(recs, str(plots_dir / "eps_vs_budget.png"))
    plot_rho_vs_budget(recs, str(plots_dir / "rho_vs_budget.png"))
    plot_throughput(recs, str(plots_dir / "throughput.png"))
    plot_speedup_vs_budget(recs, str(plots_dir / "speedup_vs_budget.png"))
    plot_wallclock_combo(recs, str(plots_dir / "wallclock.png"))
    return {key: plots_dir / fname for key, fname in PLOT_FILES if (plots_dir / fname).exists()}


def load_records_csv(csv_path: Path) -> list[Record]:
    """Load a v2 ``results.csv`` back into the record format the
    plotters consume. Useful for re-rendering without re-running."""
    with open(csv_path) as f:
        return list(csv.DictReader(f))
