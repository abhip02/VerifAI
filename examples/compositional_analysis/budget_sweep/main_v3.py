"""v3 budget sweep — Appendix C grid: 4 specs × 7 scenarios = 28 cells.

Trace-replay sweep over pre-generated CSVs (no Scenic regeneration).
Engine calls go through `check_with_dfa` (MetaDrive) and
`check_with_dfa_scenic` (Scenic choose/shuffle); monolithic ground
truth via `relabel_traces`. The two co-safety specs (V-shape, sustained
steering) run as their absorbing-reject safety complement, with
ρ_cosafety = 1 − ρ_safety.

Coverage:
    Specs (App C §C.1):       two_stops, tollgate, vshape, sustained_steer
    MD scenarios (5 each):    SX, SXS, SOC, CSXS, CXSXC
    Scenic scenarios (2 each):  choose, shuffle
    Total: 4 × (5 + 2) = 28 cells.

Per-spec MD/Scenic parameter pairs live in
`examples.compositional_analysis.scenic_scenarios.specs` and match
App C §C.3 (different signal thresholds / counter windows / warmups
for the two backends, identical DFA structure).
"""

from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from verifai.compositional_analysis import (
    CompositionalAnalysisEngine,
    ScenarioBase,
    relabel_traces,
)
from verifai.scenic_composition_analysis import (
    analyze_scenic_composition,
    build_partner_format,
)
from verifai.generate_graph_traces import generate_graph_scenarios
from verifai.scenic_parser import parse_scenic_spec

from examples.compositional_analysis.scenic_scenarios.specs import (
    add_dh_column,
    make_steer_spec_metadrive,
    make_steer_spec_scenic,
    make_tollgate_spec_md,
    make_tollgate_spec_scenic,
    make_two_stops_spec_md,
    make_two_stops_spec_scenic,
    make_vshape_safety_spec_md,
    make_vshape_safety_spec_scenic,
)

# ---------------------------------------------------------------------------
# Paths and scenario catalogues
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
COMP_DIR = REPO_ROOT / "examples/compositional_analysis"

MD_BASE = COMP_DIR / "dfa_tests/storage/vshape_speed_new"
SCENIC_BASE = COMP_DIR / "dfa_tests/e2e_4way_example/storage_paper"
SCENIC_FILE = (
    COMP_DIR
    / "dfa_tests/e2e_4way_example/4_way_intersection_scenic/composed_scenarios.scenic"
)

MD_PRIMITIVES = ("S", "X", "C", "O")
MD_COMBOS: dict[str, list[str]] = {
    "SX": ["S", "X"],
    "SXS": ["S", "X", "S"],
    "SOC": ["S", "O", "C"],
    "CSXS": ["C", "S", "X", "S"],
    "CXSXC": ["C", "X", "S", "X", "C"],
}

SCENIC_PRIMITIVES = (
    "Subscenario1",
    "Subscenario2L",
    "Subscenario2R",
    "Subscenario2S",
)
SCENIC_MONO = {
    "choose": "MonolithicMain",
    "shuffle": "MonolithicShuffle",
}
SCENIC_ENTRYPOINT = {
    "choose": "Main",
    "shuffle": "ShuffleMain",
}

_V3_SAVE_ROOT = Path("storage/budget_sweep_v3")
_FILTER_CACHE_ROOT = _V3_SAVE_ROOT / "_filtered_traces"


# ---------------------------------------------------------------------------
# 30-min wall-clock calibration from historical sweeps
# ---------------------------------------------------------------------------
#
# Old runs under storage/budget_sweep_v3/<cell>/{compositional,monolithic}/<name>/traces.csv
# were generated under a 30-min per-side wall-clock cap. The CSVs have
# no time column, so to emulate that cap on top of today's
# unlimited-budget CSVs we use the median trace-count per scenario name
# from those historical runs as the per-scenario row cap.
#
# Aliases below map historical scenario names → the names main_v3.py
# expects today (the old runs used a slightly different combo set;
# only overlapping names are useful for calibration).

_HISTORY_ROOT = _V3_SAVE_ROOT


def _count_distinct_traces(csv_path: Path) -> int:
    """Cheap distinct-`trace_id` count: read col 0, dedupe, return len.

    Avoids loading the full CSV into pandas just to count IDs.
    """
    import csv as _csv

    seen: set[str] = set()
    with csv_path.open() as f:
        r = _csv.reader(f)
        header = next(r, None)
        if not header:
            return 0
        for row in r:
            if row:
                seen.add(row[0])
    return len(seen)


def _median(xs: list[int]) -> int:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) // 2


def calibrate_caps_from_history(
    history_root: Path = _HISTORY_ROOT,
) -> tuple[dict[str, int], dict[str, int]]:
    """Walk historical 30-min sweeps and return per-scenario row caps.

    Returns (comp_caps, mono_caps):
      - ``comp_caps[name]`` = median distinct-trace count seen for
        primitive ``name`` across all historical compositional/<name>/
        directories. Keyed by MD primitive name (S/X/C/O).
      - ``mono_caps[combo]`` = median distinct-trace count for the
        monolithic CSV of ``combo``, keyed by the *combo* name
        (e.g. "SX", "CSXS") — Mono prefix stripped.

    Missing names mean "no historical data" and the caller should pick
    a fallback. Today only MD scenarios appear in storage/budget_sweep_v3;
    Scenic primitives/monoliths have no calibration data here.
    """
    comp_buckets: dict[str, list[int]] = {}
    mono_buckets: dict[str, list[int]] = {}

    if not history_root.is_dir():
        return {}, {}

    for cell_dir in history_root.iterdir():
        if not cell_dir.is_dir() or cell_dir.name.startswith("_"):
            continue

        comp_dir = cell_dir / "compositional"
        if comp_dir.is_dir():
            for prim in comp_dir.iterdir():
                csv = prim / "traces.csv"
                if csv.is_file():
                    comp_buckets.setdefault(prim.name, []).append(
                        _count_distinct_traces(csv)
                    )

        mono_dir = cell_dir / "monolithic"
        if mono_dir.is_dir():
            for mono in mono_dir.iterdir():
                csv = mono / "traces.csv"
                if csv.is_file():
                    # Strip "Mono" prefix → combo name (MonoSX → SX,
                    # MonoSChooseCXO → SChooseCXO, etc.)
                    key = mono.name[4:] if mono.name.startswith("Mono") else mono.name
                    mono_buckets.setdefault(key, []).append(_count_distinct_traces(csv))

    comp_caps = {k: _median(v) for k, v in comp_buckets.items()}
    mono_caps = {k: _median(v) for k, v in mono_buckets.items()}
    return comp_caps, mono_caps


# Effective "segment count" for each scenario, used to interpolate
# wall-clock-cap trace counts when history is sparse. Monolithic combos
# run the full segment chain on one ego; their wall-clock per trace
# scales roughly linearly with the segment count. Scenic monoliths are
# mapped to their equivalent MD chain length:
#   MonolithicMain   — choose between 2 sub-paths → ~2 segments
#   MonolithicShuffle — shuffle of 4 maneuvers → ~4 segments
_MONO_SEGMENT_COUNT = {
    "SX": 2,
    "SXS": 3,
    "SOC": 3,
    "CSXS": 4,
    "CXSXC": 5,
    "MonolithicMain": 2,
    "MonolithicShuffle": 4,
}

# Primitives — all run a single FollowLane segment at similar lengths,
# so we collapse them into a single "primitive" bucket for interpolation.
_COMP_PRIMITIVE_NAMES = (
    "S",
    "X",
    "C",
    "O",
    "Subscenario1",
    "Subscenario2L",
    "Subscenario2R",
    "Subscenario2S",
)


def _interp_monolithic_cap(history: dict[str, int], target_segments: int) -> int | None:
    """Estimate a monolithic-side trace cap for a combo with
    ``target_segments`` segments, using historical (segments → count)
    pairs as anchors and linear interpolation. Falls back to a single
    historical median if only one anchor is available, and to None if
    history is empty.
    """
    anchors = sorted(
        (_MONO_SEGMENT_COUNT[k], v)
        for k, v in history.items()
        if k in _MONO_SEGMENT_COUNT
    )
    if not anchors:
        return None
    if len(anchors) == 1:
        return anchors[0][1]
    # Pick the two anchors that bracket target_segments (or the nearest
    # two if target falls outside the anchor range).
    lo = anchors[0]
    hi = anchors[-1]
    for i in range(len(anchors) - 1):
        if anchors[i][0] <= target_segments <= anchors[i + 1][0]:
            lo, hi = anchors[i], anchors[i + 1]
            break
    if hi[0] == lo[0]:
        return lo[1]
    slope = (hi[1] - lo[1]) / (hi[0] - lo[0])
    est = lo[1] + slope * (target_segments - lo[0])
    return max(1, int(round(est)))


def build_auto_caps() -> tuple[dict[str, int], dict[str, int]]:
    """Return (comp_caps, mono_caps) dicts ready to drop into
    ``MAX_TRACES_COMP`` / ``MAX_TRACES_MONO``.

    Walks history once, then fills every scenario in v3's grid:
      - Compositional primitives: historical median if seen, else the
        median across all historical primitives (single bucket).
      - Monolithic combos: historical median if seen, else linear
        interpolation by segment count using SX/CSXS-style anchors.
    """
    raw_comp, raw_mono = calibrate_caps_from_history()

    # Compositional fallback: only consider v3-grid primitives that
    # appear in history. Older sweeps included other primitives (e.g.
    # TurnL/Straight) at very different cadence; mixing them in would
    # bias the Scenic-primitive default. Scenic primitives have no
    # history of their own, so they inherit this MD-only median.
    in_grid = [raw_comp[n] for n in _COMP_PRIMITIVE_NAMES if n in raw_comp]
    prim_default: int | None = _median(in_grid) if in_grid else None

    comp_caps: dict[str, int] = {}
    for name in _COMP_PRIMITIVE_NAMES:
        if name in raw_comp:
            comp_caps[name] = raw_comp[name]
        elif prim_default is not None:
            comp_caps[name] = prim_default

    mono_caps: dict[str, int] = {}
    for name in _MONO_SEGMENT_COUNT:
        if name in raw_mono:
            mono_caps[name] = raw_mono[name]
        else:
            est = _interp_monolithic_cap(raw_mono, _MONO_SEGMENT_COUNT[name])
            if est is not None:
                mono_caps[name] = est

    return comp_caps, mono_caps


def _resolve_cap(name: str, spec: object) -> int | None:
    """Resolve a per-side cap spec down to a single int (or None).

    ``spec`` can be:
      - ``None``                → no cap
      - ``int``                 → same cap for every scenario
      - ``dict[str, int]``      → per-scenario cap, keyed by scenario name
      - ``dict`` with ``"_default"`` key → fallback used for any
        scenario not explicitly listed (handy when calibration data
        only covers part of the grid).
    """
    if spec is None:
        return None
    if isinstance(spec, int):
        return spec if spec > 0 else None
    if isinstance(spec, dict):
        if name in spec:
            v = spec[name]
            return int(v) if v else None
        if "_default" in spec:
            v = spec["_default"]
            return int(v) if v else None
        return None
    raise TypeError(f"Unsupported cap spec type: {type(spec).__name__}")


def _filtered_csv(src: Path, max_traces: int | None) -> str:
    """Return path to a copy of ``src`` containing only the first
    ``max_traces`` trace_ids. Cached under ``_FILTER_CACHE_ROOT`` keyed
    by (scenario dir name, count) so repeated cells share the same file.
    Pass ``None`` to bypass filtering.
    """
    if not max_traces:
        return str(src)
    import pandas as pd

    _FILTER_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    dst = _FILTER_CACHE_ROOT / f"{src.parent.name}__top{max_traces}.csv"
    if not dst.is_file() or dst.stat().st_mtime < src.stat().st_mtime:
        # on_bad_lines='skip' rescues sources with corrupted rows (race-
        # condition writes can splice two records onto one line, e.g.
        # the CXSXC monolith CSV at line 409635). low_memory=False
        # silences the dtype warning on Scenic monoliths whose `label`
        # column mixes ints and strings.
        df = pd.read_csv(src, on_bad_lines="skip", low_memory=False)
        keep = df["trace_id"].drop_duplicates().head(max_traces)
        df[df["trace_id"].isin(keep)].to_csv(dst, index=False)
    return str(dst)


# ---------------------------------------------------------------------------
# Trace sources for on-demand generation
# ---------------------------------------------------------------------------

# Scenic source files that define each backend's primitives + monoliths.
MD_PRIMITIVES_SRC = COMP_DIR / "scenic_scenarios/metadrive/primitives.scenic"
MD_MONO_DIR = COMP_DIR / "scenic_scenarios/metadrive/monolithic"

# combo dir name → (.scenic source, scenario name inside the file)
MD_MONO_SOURCES: dict[str, tuple[Path, str]] = {
    "SX": (MD_MONO_DIR / "mono_SX.scenic", "MonoSX"),
    "SXS": (MD_MONO_DIR / "mono_SXS.scenic", "MonoSXS"),
    "SOC": (MD_MONO_DIR / "mono_SOC.scenic", "MonoSOC"),
    "CSXS": (MD_MONO_DIR / "mono_CSXS.scenic", "MonoCSXS"),
    "CXSXC": (MD_MONO_DIR / "mono_CXSXC.scenic", "MonoCXSXC"),
}

DEFAULT_N_TRACES = 1000
SCENIC_PREWARM_TRIM = 25  # rows trimmed from Sub2* (matches test_4way_paper_specs.py)
SCENIC_WARMUP_STEPS = 25  # step-offset after trim
SCENIC_SUB2_NAMES = {"Subscenario2L", "Subscenario2R", "Subscenario2S"}
SCENIC_MAX_STEPS_PRIMITIVE = 85
SCENIC_MAX_STEPS_SUB2 = SCENIC_MAX_STEPS_PRIMITIVE + SCENIC_PREWARM_TRIM  # 110
SCENIC_MAX_STEPS_MONO_MAIN = SCENIC_MAX_STEPS_PRIMITIVE * 2  # choose
SCENIC_MAX_STEPS_MONO_SHUF = SCENIC_MAX_STEPS_PRIMITIVE * 4  # shuffle


def _trim_prewarm(csv_path: Path, n_trim: int, step_offset: int) -> None:
    """Drop the first n_trim rows per trace and rebase `step` to step_offset.

    Lifted from dfa_tests/e2e_4way_example/test_4way_paper_specs.py so the
    Scenic Sub2 primitive logs land in the same step coordinate the specs
    expect (`step >= WARMUP_STEPS = 25`).
    """
    import pandas as pd  # local to keep import surface small if unused

    df = pd.read_csv(csv_path).sort_values(["trace_id", "step"])
    trimmed = []
    for _, grp in df.groupby("trace_id"):
        kept = grp.iloc[n_trim:].copy()
        kept["step"] = range(step_offset, step_offset + len(kept))
        trimmed.append(kept)
    pd.concat(trimmed, ignore_index=True).to_csv(csv_path, index=False)


def _have_traces(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _ensure_md_traces(
    n_traces: int,
    gen_workers: int = 1,
    time_budget: float = float("inf"),
) -> None:
    """Generate any missing MetaDrive primitive / combo CSVs under MD_BASE.

    Primitives (S/X/C/O) come from primitives.scenic; combos come from
    monolithic/mono_*.scenic with scenario name `MonoXXX`. After generation
    the `MonoXXX/` directory is renamed to `XXX/` to match the layout
    `_run_metadrive` expects (`MD_BASE/{combo}/traces.csv`).

    When ``gen_workers > 1``, primitives are still produced in a single
    `generate_graph_scenarios` call (it already spawns one subprocess per
    primitive in the list — parallel by construction), and the *combo*
    generations are then dispatched onto a thread pool of size
    ``gen_workers`` so all 5 combo subprocesses overlap.
    """
    MD_BASE.mkdir(parents=True, exist_ok=True)

    missing_prims = [
        p for p in MD_PRIMITIVES if not _have_traces(MD_BASE / p / "traces.csv")
    ]
    if missing_prims:
        print(f"[ensure] MD primitives missing: {missing_prims} → generating")
        generate_graph_scenarios(
            str(MD_PRIMITIVES_SRC),
            missing_prims,
            n=n_traces,
            save_dir=str(MD_BASE),
            time_budget=time_budget,
        )

    pending_combos = [
        (combo, src, mono_name)
        for combo, (src, mono_name) in MD_MONO_SOURCES.items()
        if not _have_traces(MD_BASE / combo / "traces.csv")
    ]

    def _gen_one(combo: str, src: Path, mono_name: str) -> None:
        print(f"[ensure] MD combo {combo} missing → generating ({mono_name})")
        generate_graph_scenarios(
            str(src),
            [mono_name],
            n=n_traces,
            save_dir=str(MD_BASE),
            time_budget=time_budget,
        )
        src_dir = MD_BASE / mono_name
        dst_dir = MD_BASE / combo
        if src_dir.is_dir() and not dst_dir.exists():
            src_dir.rename(dst_dir)

    if not pending_combos:
        return
    if gen_workers <= 1:
        for combo, src, mono_name in pending_combos:
            _gen_one(combo, src, mono_name)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=gen_workers) as pool:
            futs = [pool.submit(_gen_one, c, s, m) for c, s, m in pending_combos]
            for f in futs:
                f.result()


def _ensure_scenic_traces(
    n_traces: int,
    gen_workers: int = 1,
    time_budget: float = float("inf"),
) -> None:
    """Generate any missing Scenic primitive / monolithic CSVs under SCENIC_BASE."""
    SCENIC_BASE.mkdir(parents=True, exist_ok=True)

    # Primitives: per-leaf max_steps + Sub2 prewarm trim.
    missing_prims = [
        p for p in SCENIC_PRIMITIVES if not _have_traces(SCENIC_BASE / p / "traces.csv")
    ]
    if missing_prims:
        print(f"[ensure] Scenic primitives missing: {missing_prims} → generating")
        max_steps_map = {
            p: (
                SCENIC_MAX_STEPS_SUB2
                if p in SCENIC_SUB2_NAMES
                else SCENIC_MAX_STEPS_PRIMITIVE
            )
            for p in missing_prims
        }
        generate_graph_scenarios(
            str(SCENIC_FILE),
            missing_prims,
            n=n_traces,
            save_dir=str(SCENIC_BASE),
            max_steps=max_steps_map,
            time_budget=time_budget,
        )
        for p in missing_prims:
            if p in SCENIC_SUB2_NAMES:
                _trim_prewarm(
                    SCENIC_BASE / p / "traces.csv",
                    SCENIC_PREWARM_TRIM,
                    step_offset=SCENIC_WARMUP_STEPS,
                )

    # Monoliths. Parallelizable across the 2 monoliths if requested.
    pending_monos = [
        (combo, mono_name)
        for combo, mono_name in SCENIC_MONO.items()
        if not _have_traces(SCENIC_BASE / mono_name / "traces.csv")
    ]

    def _gen_mono(combo: str, mono_name: str) -> None:
        max_steps = (
            SCENIC_MAX_STEPS_MONO_SHUF
            if combo == "shuffle"
            else SCENIC_MAX_STEPS_MONO_MAIN
        )
        print(f"[ensure] Scenic monolith {mono_name} missing → generating")
        generate_graph_scenarios(
            str(SCENIC_FILE),
            [mono_name],
            n=n_traces,
            save_dir=str(SCENIC_BASE),
            max_steps=max_steps,
            time_budget=time_budget,
        )

    if not pending_monos:
        return
    if gen_workers <= 1:
        for combo, mono_name in pending_monos:
            _gen_mono(combo, mono_name)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=gen_workers) as pool:
            futs = [pool.submit(_gen_mono, c, m) for c, m in pending_monos]
            for f in futs:
                f.result()


def _ensure_traces(
    cells: list["Cell"],
    n_traces: int,
    gen_workers: int = 1,
    time_budget: float = float("inf"),
) -> None:
    """Pre-flight: generate any missing trace CSVs needed by `cells`.

    ``time_budget`` is a per-scenario wall-clock cap (seconds) passed
    through to ``generate_graph_scenarios``: generation hard-stops when
    the budget elapses, capping the trace count even if ``n_traces``
    hasn't been reached.
    """
    backends = {c.backend for c in cells}
    if "metadrive" in backends:
        _ensure_md_traces(n_traces, gen_workers=gen_workers, time_budget=time_budget)
    if "scenic" in backends:
        _ensure_scenic_traces(
            n_traces, gen_workers=gen_workers, time_budget=time_budget
        )


# ---------------------------------------------------------------------------
# Cell definition + experiment list
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    name: str
    spec_name: str
    spec_factory: Callable[[], object]
    backend: str  # "metadrive" | "scenic"
    combo: str  # MD: "SX"... ; Scenic: "choose"|"shuffle"


# App C grid: 4 specs × 7 scenarios = 28 cells. Per §C.3 the DFAs are
# identical across backends; only thresholds/counters/warmup differ, so
# each spec ships as a (MD-factory, Scenic-factory) pair.
_APP_C_SPECS: list[tuple[str, Callable, Callable]] = [
    ("two_stops", make_two_stops_spec_md, make_two_stops_spec_scenic),
    ("tollgate", make_tollgate_spec_md, make_tollgate_spec_scenic),
    ("vshape", make_vshape_safety_spec_md, make_vshape_safety_spec_scenic),
    ("sustained_steer", make_steer_spec_metadrive, make_steer_spec_scenic),
]


def _build_experiments() -> list[Cell]:
    cells: list[Cell] = []

    for spec_name, md_factory, scenic_factory in _APP_C_SPECS:
        for combo in MD_COMBOS:
            cells.append(
                Cell(
                    name=f"metadrive__{spec_name}__{combo}",
                    spec_name=spec_name,
                    spec_factory=md_factory,
                    backend="metadrive",
                    combo=combo,
                )
            )
        for combo in ("choose", "shuffle"):
            cells.append(
                Cell(
                    name=f"scenic__{spec_name}__{combo}",
                    spec_name=spec_name,
                    spec_factory=scenic_factory,
                    backend="scenic",
                    combo=combo,
                )
            )

    return cells


EXPERIMENTS: list[Cell] = _build_experiments()


# ---------------------------------------------------------------------------
# Engine runners — one per backend, both return (rho_safety_comp, eps, rho_safety_mono)
# ---------------------------------------------------------------------------

_scenic_paths_cache: dict[str, list] | None = None


def _scenic_paths(combo: str):
    global _scenic_paths_cache
    if _scenic_paths_cache is None:
        graph = analyze_scenic_composition(str(SCENIC_FILE))
        partner = build_partner_format(graph)
        _scenic_paths_cache = parse_scenic_spec(partner)
    return _scenic_paths_cache[SCENIC_ENTRYPOINT[combo]]


def _run_metadrive(
    spec, combo: str, max_traces_comp, max_traces_mono
) -> tuple[float, float, float]:
    prim_paths = {
        p: _filtered_csv(MD_BASE / p / "traces.csv", _resolve_cap(p, max_traces_comp))
        for p in MD_PRIMITIVES
    }
    engine = CompositionalAnalysisEngine(ScenarioBase(prim_paths))
    rho_safe_comp, eps = engine.check_with_dfa(
        MD_COMBOS[combo], spec, features=["speed"], center_feat_idx=[]
    )
    rho_safe_mono = relabel_traces(
        _filtered_csv(
            MD_BASE / combo / "traces.csv", _resolve_cap(combo, max_traces_mono)
        ),
        spec,
    )
    return float(rho_safe_comp), float(eps), float(rho_safe_mono)


def _run_scenic(
    spec, combo: str, max_traces_comp, max_traces_mono
) -> tuple[float, float, float]:
    logs = {
        p: _filtered_csv(
            SCENIC_BASE / p / "traces.csv", _resolve_cap(p, max_traces_comp)
        )
        for p in SCENIC_PRIMITIVES
    }
    engine = CompositionalAnalysisEngine(ScenarioBase(logs))
    rho_safe_comp, eps = engine.check_with_dfa_scenic(
        _scenic_paths(combo), spec, features=["speed"], center_feat_idx=[]
    )
    mono_name = SCENIC_MONO[combo]
    rho_safe_mono = relabel_traces(
        _filtered_csv(
            SCENIC_BASE / mono_name / "traces.csv",
            _resolve_cap(mono_name, max_traces_mono),
        ),
        spec,
    )
    return float(rho_safe_comp), float(eps), float(rho_safe_mono)


_RUNNERS: dict[str, Callable] = {
    "metadrive": _run_metadrive,
    "scenic": _run_scenic,
}


_CALIB_BUDGET_S = 1800.0  # the wall-clock budget the calibration dicts represent
_CI_Z = 1.96  # 95% normal-approx CI half-width for the monolithic side


def _scale_caps(caps: dict[str, int], factor: float) -> dict[str, int]:
    """Scale every value in a cap dict by ``factor`` (rounded, ≥1)."""
    return {k: max(1, int(round(v * factor))) for k, v in caps.items()}


def _normal_ci_half(p: float, n: int) -> float:
    if not n or n <= 0:
        return float("nan")
    p = max(0.0, min(1.0, p))
    return _CI_Z * (p * (1.0 - p) / n) ** 0.5


def _mono_scenario_name(cell: Cell) -> str:
    """Resolve the lookup key used inside the mono-cap dict for a given cell."""
    if cell.backend == "metadrive":
        return cell.combo
    return SCENIC_MONO[cell.combo]


def run_cell_convergence(
    cell: Cell,
    budgets_s,
    comp_caps_calib: dict[str, int] | None,
    mono_caps_calib: dict[str, int] | None,
) -> list[dict]:
    """Run a single cell at multiple time-budget points (seconds).

    The calibration dicts give the per-scenario trace count that fits in
    ``_CALIB_BUDGET_S`` (default 30 min). For each ``t`` in ``budgets_s``
    we scale every cap by ``t / _CALIB_BUDGET_S`` and run the engine
    once; two ``Record``-shaped dicts (compositional + monolithic) are
    appended per budget point so the rows feed straight into
    ``plots.plot_rho_vs_budget``.

    eps for the compositional side comes from the engine (Hoeffding CI);
    eps for the monolithic side is a 95 % normal-approx half-width
    derived from ``rho_mono`` and the number of monolithic traces kept
    at that budget.
    """
    runner = _RUNNERS[cell.backend]
    spec = cell.spec_factory()
    mono_name = _mono_scenario_name(cell)
    records: list[dict] = []

    for t in budgets_s:
        factor = float(t) / _CALIB_BUDGET_S
        cc = _scale_caps(comp_caps_calib, factor) if comp_caps_calib else None
        mc = _scale_caps(mono_caps_calib, factor) if mono_caps_calib else None
        n_mono = _resolve_cap(mono_name, mc) if mc else None

        rho_safe_c, eps_c, rho_safe_m = runner(spec, cell.combo, cc, mc)
        rho_c = 1.0 - rho_safe_c
        rho_m = 1.0 - rho_safe_m
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


def run_cell(
    cell: Cell,
    max_traces_comp=None,
    max_traces_mono=None,
) -> dict:
    """Run a single (spec, scenario) cell. Returns a row dict.

    ``max_traces_comp`` caps the primitive CSVs fed to the compositional
    engine; ``max_traces_mono`` caps the monolithic-ground-truth CSV.
    Each may be ``None`` (no cap), an ``int`` (same cap for every
    scenario), or a ``dict`` keyed by scenario name (optionally with a
    ``"_default"`` fallback). See ``_resolve_cap``.
    """
    runner = _RUNNERS[cell.backend]
    spec = cell.spec_factory()
    t0 = time.time()
    rho_safe_comp, eps_comp, rho_safe_mono = runner(
        spec, cell.combo, max_traces_comp, max_traces_mono
    )
    rho_comp = 1.0 - rho_safe_comp
    rho_mono = 1.0 - rho_safe_mono
    return {
        "cell": cell.name,
        "backend": cell.backend,
        "spec": cell.spec_name,
        "combo": cell.combo,
        "rho_comp": rho_comp,
        "rho_mono": rho_mono,
        "abs_diff": abs(rho_comp - rho_mono),
        "eps_comp": eps_comp,
        "elapsed_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_RESULT_FIELDS = [
    "cell",
    "backend",
    "spec",
    "combo",
    "rho_comp",
    "rho_mono",
    "abs_diff",
    "eps_comp",
    "elapsed_s",
    "error",
]


def _error_row(cell: Cell, exc: BaseException) -> dict:
    return {
        "cell": cell.name,
        "backend": cell.backend,
        "spec": cell.spec_name,
        "combo": cell.combo,
        "rho_comp": None,
        "rho_mono": None,
        "abs_diff": None,
        "eps_comp": None,
        "elapsed_s": None,
        "error": f"{type(exc).__name__}: {exc}",
    }


def main(
    *,
    only: str | None = None,
    list_only: bool = False,
    max_traces_comp=None,
    max_traces_mono=None,
    budgets_s: tuple[float, ...] | None = None,
    n_traces: int = DEFAULT_N_TRACES,
    skip_generation: bool = False,
    workers: int = 1,
    gen_time_budget: float = float("inf"),
    gen_workers: int = 1,
    add_dh: bool = False,
) -> None:
    cells = EXPERIMENTS
    if only:
        cells = [c for c in cells if only in c.name]

    # Resolve "auto" → calibration dicts derived from historical 30-min
    # sweeps under storage/budget_sweep_v3/. Computed once per run.
    if max_traces_comp == "auto" or max_traces_mono == "auto":
        comp_caps, mono_caps = build_auto_caps()
        if comp_caps or mono_caps:
            print(
                "[calibration] 30-min auto caps "
                "(history + segment-count interpolation):"
            )
            print(f"  comp = {comp_caps or '(none)'}")
            print(f"  mono = {mono_caps or '(none)'}")
        else:
            print(
                "[calibration] WARNING: no historical sweeps found under "
                f"{_HISTORY_ROOT} — 'auto' will fall back to no cap."
            )
        if max_traces_comp == "auto":
            max_traces_comp = comp_caps or None
        if max_traces_mono == "auto":
            max_traces_mono = mono_caps or None

    if list_only:
        print(f"{len(cells)} cells:")
        for c in cells:
            print(f"  {c.name}")
        return

    if not skip_generation:
        _ensure_traces(
            cells,
            n_traces,
            gen_workers=gen_workers,
            time_budget=gen_time_budget,
        )

    if add_dh:
        add_dh_column(SCENIC_BASE)

    save_dir = _V3_SAVE_ROOT
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = save_dir / f"results_{stamp}.csv"

    workers = max(1, min(workers, len(cells)))
    rows: list[dict] = []
    print(f"[main_v3] running {len(cells)} cells (workers={workers}) → {out_csv}")

    # Convergence-sweep mode: walk each cell across multiple time budgets
    # (in seconds), record (rho, eps) per (cell, method, budget), render
    # one rho-vs-budget plot per cell, and aggregate the largest-budget
    # row of each cell into the usual summary CSV.
    if budgets_s:
        if not (isinstance(max_traces_comp, dict) and isinstance(max_traces_mono, dict)):
            raise SystemExit(
                "budgets_s requires dict-valued max_traces_comp / "
                "max_traces_mono (use 'auto' or pass a dict)."
            )
        from .plots import plot_rho_vs_budget

        per_cell_dir = save_dir / f"convergence_{stamp}"
        per_cell_dir.mkdir(parents=True, exist_ok=True)
        all_records: list[dict] = []  # for W&B
        cell_to_records: dict[str, list[dict]] = {}

        for cell in cells:
            print(f"\n{'=' * 70}\nCELL: {cell.name}\n{'=' * 70}")
            try:
                recs = run_cell_convergence(
                    cell,
                    budgets_s,
                    comp_caps_calib=max_traces_comp,
                    mono_caps_calib=max_traces_mono,
                )
            except Exception as exc:
                print(f"[{cell.name}] ERROR: {exc}", file=sys.stderr)
                rows.append(_error_row(cell, exc))
                continue

            cell_to_records[cell.name] = recs
            for r in recs:
                all_records.append({**r, "cell": cell.name, "spec": cell.spec_name,
                                    "backend": cell.backend, "combo": cell.combo})

            # Per-cell convergence CSV + plot.
            cell_csv = per_cell_dir / f"{cell.name}.csv"
            with cell_csv.open("w", newline="") as f:
                w = csv.DictWriter(
                    f, fieldnames=["method", "budget", "rho", "eps", "n_traces"]
                )
                w.writeheader()
                w.writerows(recs)
            try:
                plot_rho_vs_budget(recs, per_cell_dir / f"{cell.name}.png")
            except Exception as exc:
                print(f"[{cell.name}] plot failed: {exc}", file=sys.stderr)

            # Summary row from the largest-budget point.
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

        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_RESULT_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[main_v3] wrote {len(rows)} summary rows → {out_csv}")
        print(f"[main_v3] convergence CSVs + plots → {per_cell_dir}")

        _push_to_wandb(
            out_csv,
            rows,
            max_traces_comp,
            max_traces_mono,
            convergence_records=all_records,
            convergence_dir=per_cell_dir,
        )
        return

    if workers == 1:
        for cell in cells:
            print(f"\n{'=' * 70}\nCELL: {cell.name}\n{'=' * 70}")
            try:
                row = run_cell(
                    cell,
                    max_traces_comp=max_traces_comp,
                    max_traces_mono=max_traces_mono,
                )
                row["error"] = ""
            except Exception as exc:
                row = _error_row(cell, exc)
                print(f"[{cell.name}] ERROR: {row['error']}", file=sys.stderr)
            rows.append(row)
            print(
                f"[{cell.name}] rho_comp={row['rho_comp']} "
                f"rho_mono={row['rho_mono']} |Δ|={row['abs_diff']}"
            )
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        by_name: dict[str, dict] = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(run_cell, c, max_traces_comp, max_traces_mono): c
                for c in cells
            }
            for fut in as_completed(futures):
                cell = futures[fut]
                try:
                    row = fut.result()
                    row["error"] = ""
                except Exception as exc:
                    row = _error_row(cell, exc)
                    print(f"[{cell.name}] ERROR: {row['error']}", file=sys.stderr)
                by_name[cell.name] = row
                print(
                    f"[{cell.name}] rho_comp={row['rho_comp']} "
                    f"rho_mono={row['rho_mono']} |Δ|={row['abs_diff']} "
                    f"({len(by_name)}/{len(cells)} done)"
                )
        rows = [by_name[c.name] for c in cells if c.name in by_name]

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_RESULT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[main_v3] wrote {len(rows)} rows → {out_csv}")

    _push_to_wandb(out_csv, rows, max_traces_comp, max_traces_mono)


# ---------------------------------------------------------------------------
# W&B push
# ---------------------------------------------------------------------------

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "verifai-budget-sweep-v3")


def _push_to_wandb(
    out_csv: Path,
    rows: list[dict],
    max_traces_comp: object,
    max_traces_mono: object,
    convergence_records: list[dict] | None = None,
    convergence_dir: Path | None = None,
) -> None:
    """Log the App C grid results to W&B: the full CSV as an artifact,
    a per-cell table, per-cell scalars, and per-spec aggregate stats.

    Disabled with ``WANDB_DISABLED=1``. Silently skips if ``wandb`` is
    not installed so a missing dep never breaks the sweep.
    """
    if os.environ.get("WANDB_DISABLED"):
        print("[wandb] WANDB_DISABLED set; skipping upload")
        return
    try:
        import wandb
    except ImportError:
        print("[wandb] wandb not installed; skipping upload")
        return

    run_name = f"budget_sweep_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        reinit=True,
        config={
            "n_cells": len(rows),
            "max_traces_comp": str(max_traces_comp),
            "max_traces_mono": str(max_traces_mono),
        },
    )

    if out_csv.exists():
        art = wandb.Artifact("budget_sweep_v3_results", type="results")
        art.add_file(str(out_csv))
        wandb.log_artifact(art)

    table = wandb.Table(columns=_RESULT_FIELDS)
    for r in rows:
        table.add_data(*[r.get(k) for k in _RESULT_FIELDS])
    wandb.log({"results_table": table})

    spec_buckets: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("rho_comp") is None or r.get("rho_mono") is None:
            continue
        spec_buckets.setdefault(r["spec"], []).append(r)
        payload = {
            f"{r['cell']}/rho_comp": float(r["rho_comp"]),
            f"{r['cell']}/rho_mono": float(r["rho_mono"]),
            f"{r['cell']}/abs_diff": float(r["abs_diff"]),
            f"{r['cell']}/eps_comp": float(r["eps_comp"]),
        }
        if r.get("elapsed_s") is not None:
            payload[f"{r['cell']}/elapsed_s"] = float(r["elapsed_s"])
        wandb.log(payload)

    for spec_name, bucket in spec_buckets.items():
        diffs = [float(r["abs_diff"]) for r in bucket]
        wandb.log(
            {
                f"agg/{spec_name}/n_cells": len(bucket),
                f"agg/{spec_name}/mean_abs_diff": sum(diffs) / len(diffs),
                f"agg/{spec_name}/max_abs_diff": max(diffs),
            }
        )

    if convergence_dir and convergence_dir.is_dir():
        plot_imgs: dict[str, "wandb.Image"] = {}
        for png in sorted(convergence_dir.glob("*.png")):
            plot_imgs[f"convergence/{png.stem}"] = wandb.Image(str(png))
        if plot_imgs:
            wandb.log(plot_imgs)
        art = wandb.Artifact("budget_sweep_v3_convergence", type="results")
        for f in convergence_dir.iterdir():
            if f.is_file():
                art.add_file(str(f))
        wandb.log_artifact(art)

    if convergence_records:
        conv_cols = ["cell", "spec", "backend", "combo", "method",
                     "budget", "rho", "eps", "n_traces"]
        conv_table = wandb.Table(columns=conv_cols)
        for r in convergence_records:
            conv_table.add_data(*[r.get(k) for k in conv_cols])
        wandb.log({"convergence_table": conv_table})

        # Per-budget time series so W&B's native scalar panels can
        # render the convergence curve too. step = budget in seconds,
        # which matches the x-axis label downstream.
        by_step: dict[int, dict[str, float]] = {}
        for r in convergence_records:
            step = int(round(float(r["budget"])))
            entry = by_step.setdefault(step, {})
            entry[f"conv/{r['cell']}/{r['method']}/rho"] = float(r["rho"])
            entry[f"conv/{r['cell']}/{r['method']}/eps"] = float(r["eps"])
        for step in sorted(by_step):
            wandb.log(by_step[step], step=step)

    wandb.finish()


if __name__ == "__main__":
    # Edit these to control the run — no CLI flags.
    ONLY = None  # e.g. "metadrive" or "tollgate"; None = all 28 cells
    LIST_ONLY = False  # True → just print cell names and exit
    # Per-side trace caps — emulate a wall-clock budget on top of the
    # pre-generated CSVs. Three accepted forms each:
    #   None              → no cap
    #   int               → same cap for every scenario
    #   dict[str, int]    → per-scenario cap; key by primitive name
    #                       (S, X, C, O, Subscenario1, …) for comp, or
    #                       combo name (SX, SXS, …, MonolithicMain,
    #                       MonolithicShuffle) for mono. Add "_default"
    #                       for a fallback used by any missing scenario.
    #   "auto"            → calibrate from prior 30-min runs under
    #                       storage/budget_sweep_v3/<cell>/{compositional,
    #                       monolithic}/<name>/traces.csv. Falls back to
    #                       None for any scenario with no history.
    #
    # Today's history covers MD only (S/X/C/O ≈ ~280 traces in 30 min;
    # SX ≈ 187, CSXS ≈ 101). Scenic names won't appear in calibration
    # output yet, so pair "auto" with an explicit "_default" if you want
    # Scenic cells capped too:
    #   MAX_TRACES_COMP = {"_default": 250}
    #   MAX_TRACES_MONO = {"SX": 187, "CSXS": 101, "_default": 120}
    MAX_TRACES_COMP = "auto"
    MAX_TRACES_MONO = "auto"

    # Convergence sweep. When set, each cell runs at every budget point
    # below; per-cell rho-vs-budget plots are rendered into
    # storage/budget_sweep_v3/convergence_<stamp>/ and uploaded to W&B.
    # The MAX_TRACES_* dicts (must be dicts in this mode — use "auto"
    # above) are treated as the 30-min anchor and linearly scaled per
    # budget point. Set BUDGETS_S = None to fall back to single-shot.
    # Dense 30-second grid: 30, 60, 90, …, 1800 → 60 points per cell.
    # plot_rho_vs_budget renders the x-axis on a log scale, so the early
    # points still show up; convergence reads as a smooth curve.
    BUDGETS_S: tuple[float, ...] | None = tuple(range(30, 1801, 30))
    N_TRACES = DEFAULT_N_TRACES  # traces requested when *generating* missing CSVs
    SKIP_GENERATION = False  # True → never auto-generate; missing CSVs error out
    WORKERS = 1  # analysis-pass subprocesses
    GEN_TIME_BUDGET = float("inf")  # per-scenario wall-clock cap for generation (s)
    GEN_WORKERS = 1  # parallel scenarios during generation pre-flight
    ADD_DH = False  # Scenic dh-column preprocessing pass

    main(
        only=ONLY,
        list_only=LIST_ONLY,
        max_traces_comp=MAX_TRACES_COMP,
        max_traces_mono=MAX_TRACES_MONO,
        budgets_s=BUDGETS_S,
        n_traces=N_TRACES,
        skip_generation=SKIP_GENERATION,
        workers=WORKERS,
        gen_time_budget=GEN_TIME_BUDGET,
        gen_workers=GEN_WORKERS,
        add_dh=ADD_DH,
    )
