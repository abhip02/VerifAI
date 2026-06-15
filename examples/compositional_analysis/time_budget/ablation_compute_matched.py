"""Compute-matched ablation for the time-budget experiment.

Pre-empts the objection that the compositional side used k x 15 min of
generation (one 15-min process per primitive). The ENTIRE primitive
library of a backend is truncated to what a single 15-minute budget,
split equally across its four pools (225 s each), could generate; the
truncated library is then shared across all compositions as in the main
experiment. This matches the compositional side's total generation cost
to that of ONE monolithic scenario (each monolithic scenario kept its own
full 15-minute pool), i.e. it is stricter than a per-scenario split over
distinct primitives.

Truncation basis: pools carry no wall-clock timestamps, but trace_id is
strictly sequential generation order and each pool's total wall-clock is
exactly 900 s, so a t-second sub-budget is approximated by the first
floor(N * t / 900) traces (average per-trace cost) -- the same linearity
assumption the budget sweep itself uses for its budget axis.

Outputs (under storage/budget_sweep_v4/ablation_compute_matched/):
  table.csv             28 rows: full / matched / monolithic rho +- eps,
                        agreement flags, min handoff N_eff, dropped buckets
  <cell>.png + .csv     matched-pool convergence curves for the four
                        main-body cells

Purely post-hoc: no trace generation. Run:
  python -m examples.compositional_analysis.time_budget.ablation_compute_matched
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from verifai.compositional_analysis import (
    CompositionalAnalysisEngine,
    ScenarioBase,
    relabel_traces,
)

from examples.compositional_analysis.time_budget.main import (
    MD_COMBOS,
    SCENIC_PRIM_DIRS_DEFAULT,
    SCENIC_PRIM_DIRS_STEER,
    _normal_ci_half,
    _report_rho,
    _scenic_paths,
)
from examples.compositional_analysis.scenic_scenarios.specs import (
    make_steer_spec_metadrive,
    make_steer_spec_scenic,
    make_tollgate_spec_md,
    make_tollgate_spec_scenic,
    make_two_stops_spec_md,
    make_two_stops_spec_scenic,
    make_vshape_safety_spec_md,
    make_vshape_safety_spec_scenic,
)

REPO = Path(__file__).resolve().parents[3]
B900 = REPO / "storage/budget_sweep_v4/b900s/traces"
OUT = REPO / "storage/budget_sweep_v4/ablation_compute_matched"
CACHE = OUT / "_truncated"


def _official_dir() -> Path:
    """Newest non-empty 15-minute convergence directory: the reference for
    the full-pool and monolithic columns. Must come from the same engine
    version as the matched run, so run the grid analysis first."""
    dirs = sorted(
        d for d in (REPO / "storage/budget_sweep_v4").glob("convergence_b900s_*")
        if any(d.glob("*.csv"))
    )
    if not dirs:
        raise SystemExit("no completed convergence_b900s_* results found")
    return dirs[-1]


OFFICIAL = _official_dir()

BUDGET_S = 900.0
MD_PRIMS = ("S", "X", "C", "O")
SCENIC_PRIMS = ("Subscenario1", "Subscenario2L", "Subscenario2R", "Subscenario2S")

SPECS = [
    ("two_stops", make_two_stops_spec_md, make_two_stops_spec_scenic),
    ("tollgate", make_tollgate_spec_md, make_tollgate_spec_scenic),
    ("vshape", make_vshape_safety_spec_md, make_vshape_safety_spec_scenic),
    ("sustained_steer", make_steer_spec_metadrive, make_steer_spec_scenic),
]

CURVE_CELLS = {  # the four main-body cells
    ("two_stops", "SX"),
    ("tollgate", "SX"),
    ("vshape", "CXSXC"),
    ("sustained_steer", "CSXS"),
}


# ---------------------------------------------------------------------------
# Instrumented engine: wraps forward() to record the N_eff the engine uses
# for each handoff step. The engine code itself is not modified. Per-bucket
# drop counting is not observable from outside the forward() implementation,
# so instr_dropped is always 0 with this engine version.
# ---------------------------------------------------------------------------


class InstrumentedEngine(CompositionalAnalysisEngine):
    def reset_instr(self):
        self.instr_neff: list[float] = []  # per handoff step (as used for eps)
        self.instr_dropped: int = 0  # not observable with forward()-based engine

    def forward(self, prev_step, step, *args, **kwargs):
        out = super().forward(prev_step, step, *args, **kwargs)
        if prev_step is not None:
            _, n_eff = out
            if n_eff:
                self.instr_neff.append(float(n_eff))
        return out


# ---------------------------------------------------------------------------
# Pool truncation
# ---------------------------------------------------------------------------

_pool_n: dict[str, int] = {}


def pool_csv(backend: str, name: str) -> Path:
    return B900 / backend / name / "traces.csv"


def pool_size(backend: str, name: str) -> int:
    key = f"{backend}/{name}"
    if key not in _pool_n:
        _pool_n[key] = pd.read_csv(
            pool_csv(backend, name), usecols=["trace_id"], low_memory=False
        ).trace_id.nunique()
    return _pool_n[key]


def truncated(backend: str, name: str, n_keep: int) -> str:
    """First n_keep trace_ids (= generation prefix) of a pool, cached."""
    n_keep = max(1, int(n_keep))
    if n_keep >= pool_size(backend, name):
        return str(pool_csv(backend, name))
    CACHE.mkdir(parents=True, exist_ok=True)
    dst = CACHE / f"{backend}__{name}__n{n_keep}.csv"
    if not dst.is_file():
        df = pd.read_csv(pool_csv(backend, name), low_memory=False)
        keep = df["trace_id"].drop_duplicates().head(n_keep)
        df[df["trace_id"].isin(keep)].to_csv(dst, index=False)
    return str(dst)


def matched_caps(
    backend: str, prim_dirs: dict[str, str], seconds_total: float
) -> dict[str, int]:
    """Per-primitive trace cap for an equal split of seconds_total over the
    scenario's distinct primitive pools (keyed by directory name)."""
    k = len(set(prim_dirs.values()))
    per = seconds_total / k
    return {
        d: int(pool_size(backend, d) * per / BUDGET_S) for d in set(prim_dirs.values())
    }


# ---------------------------------------------------------------------------
# Cell evaluation
# ---------------------------------------------------------------------------


def run_md_cell(spec_factory, combo: str, budget_s: float):
    prim_dirs = {p: p for p in MD_PRIMS}
    caps = matched_caps("metadrive", prim_dirs, budget_s)
    logs = {p: truncated("metadrive", p, caps[p]) for p in MD_PRIMS}
    engine = InstrumentedEngine(ScenarioBase(logs))
    engine.reset_instr()
    rho, eps = engine.check_with_dfa(
        MD_COMBOS[combo], spec_factory(), features=["speed"], center_feat_idx=[]
    )
    return float(rho), float(eps), engine, caps


def run_scenic_cell(spec_factory, spec_name: str, combo: str, budget_s: float):
    prim_dirs = (
        SCENIC_PRIM_DIRS_STEER
        if spec_name == "sustained_steer"
        else SCENIC_PRIM_DIRS_DEFAULT
    )
    caps = matched_caps("scenic", prim_dirs, budget_s)
    logs = {p: truncated("scenic", d, caps[d]) for p, d in prim_dirs.items()}
    engine = InstrumentedEngine(ScenarioBase(logs))
    engine.reset_instr()
    rho, eps = engine.check_with_dfa_scenic(
        _scenic_paths(combo), spec_factory(), features=["speed"], center_feat_idx=[]
    )
    return float(rho), float(eps), engine, caps


def official_900s_row(cell: str):
    df = pd.read_csv(OFFICIAL / f"{cell}.csv")
    out = {}
    for m in ("compositional", "monolithic"):
        s = df[df.method == m].sort_values("budget")
        out[m] = (float(s.rho.iloc[-1]), float(s.eps.iloc[-1]))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    for spec_name, md_fac, sc_fac in SPECS:
        for combo in list(MD_COMBOS) + ["choose", "shuffle"]:
            backend = "metadrive" if combo in MD_COMBOS else "scenic"
            cell = f"{backend}__{spec_name}__{combo}"
            official = official_900s_row(cell)
            full_rho, full_eps = official["compositional"]
            mono_rho, mono_eps = official["monolithic"]

            if backend == "metadrive":
                m_rho, m_eps, eng, caps = run_md_cell(md_fac, combo, BUDGET_S)
            else:
                m_rho, m_eps, eng, caps = run_scenic_cell(
                    sc_fac, spec_name, combo, BUDGET_S
                )
            m_rho_rep = _report_rho(spec_name, m_rho)

            agree_full = abs(full_rho - mono_rho) <= full_eps + mono_eps
            agree_matched = abs(m_rho_rep - mono_rho) <= m_eps + mono_eps
            min_neff = min(eng.instr_neff) if eng.instr_neff else float("nan")

            rows.append(
                {
                    "cell": cell,
                    "spec": spec_name,
                    "scenario": combo,
                    "k_split": len(set(caps)),
                    "rho_comp_full": round(full_rho, 4),
                    "eps_comp_full": round(full_eps, 4),
                    "rho_comp_matched": round(m_rho_rep, 4),
                    "eps_comp_matched": round(m_eps, 4),
                    "rho_mono": round(mono_rho, 4),
                    "eps_mono": round(mono_eps, 4),
                    "agree_full": agree_full,
                    "agree_matched": agree_matched,
                    "min_neff_handoff": round(min_neff, 1),
                    "dropped_buckets": eng.instr_dropped,
                    "matched_caps": ";".join(
                        f"{k}={v}" for k, v in sorted(caps.items())
                    ),
                }
            )
            print(
                f"[{cell}] matched={m_rho_rep:.3f}±{m_eps:.3f} "
                f"full={full_rho:.3f}±{full_eps:.3f} mono={mono_rho:.3f}±{mono_eps:.3f} "
                f"agree={agree_matched} minNeff={min_neff:.0f} "
                f"dropped={eng.instr_dropped}"
            )

    table = OUT / "table.csv"
    with table.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {table}")

    # Convergence curves for the four main-body cells, matched caps.
    from examples.compositional_analysis.time_budget.plots import (
        cell_title,
        legend_loc_for,
        plot_rho_vs_budget,
    )

    for spec_name, md_fac, _ in SPECS:
        for combo in MD_COMBOS:
            if (spec_name, combo) not in CURVE_CELLS:
                continue
            cell = f"metadrive__{spec_name}__{combo}"
            n_mono_full = pool_size("metadrive", combo)
            spec = md_fac()
            recs = []
            for t in range(30, 901, 30):
                rho_c, eps_c, _, _ = run_md_cell(md_fac, combo, float(t))
                n_m = max(1, int(n_mono_full * t / BUDGET_S))
                rho_m = float(relabel_traces(truncated("metadrive", combo, n_m), spec))
                rc = _report_rho(spec_name, rho_c)
                rm = _report_rho(spec_name, rho_m)
                recs.append(
                    {
                        "method": "compositional",
                        "budget": float(t),
                        "rho": rc,
                        "eps": eps_c,
                        "n_traces": n_m,
                    }
                )
                recs.append(
                    {
                        "method": "monolithic",
                        "budget": float(t),
                        "rho": rm,
                        "eps": _normal_ci_half(rm, n_m),
                        "n_traces": n_m,
                    }
                )
            pd.DataFrame(recs).to_csv(OUT / f"{cell}.csv", index=False)
            plot_rho_vs_budget(recs, OUT / f"{cell}.png", title=cell_title(cell),
                               legend_loc=legend_loc_for(cell))
            print(f"[curve] {cell} done")

    print(f"\nall outputs in {OUT}")


if __name__ == "__main__":
    main()
