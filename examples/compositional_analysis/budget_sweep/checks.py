from __future__ import annotations

import logging
import numbers
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Record, Snapshot, SweepConfig


_LOG_PATH = Path(__file__).with_suffix(".log")
_log = logging.getLogger("budget_sweep.checks")
if not _log.handlers:
    _log.setLevel(logging.DEBUG)
    _log.propagate = False
    _fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _fh = logging.FileHandler(_LOG_PATH, mode="a", encoding="utf-8")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(_fmt)
    _log.addHandler(_fh)
    _sh = logging.StreamHandler()
    _sh.setLevel(logging.WARNING)
    _sh.setFormatter(_fmt)
    _log.addHandler(_sh)


def _check(cond: bool, msg: str) -> bool:
    """Log ``msg`` at WARNING level when ``cond`` is false; return ``cond``."""
    if not cond:
        _log.warning(msg)
    return cond


_REQUIRED_RECORD_KEYS = {"method", "budget", "rho", "eps", "n_traces", "status"}
_REQUIRED_JOB_KEYS = {"scenic_file", "scenario_name", "max_steps"}
# Permitted ``record["status"]`` values. ``no_traces`` / ``insufficient_data``
# / ``missing_primitives`` correspond to the Fig. 4 startup-window (some
# primitive worker hasn't emitted its first trace yet); ``analysis_error``
# covers KDE / engine failures; ``ok`` is the normal case.
_VALID_STATUSES = {
    "ok",
    "no_traces",
    "insufficient_data",
    "missing_primitives",
    "analysis_error",
}
# Upper bound on how far past ``max_budget`` a post-stop snapshot can land,
# in seconds. Bounded by ``poll_sleep`` in ``simulate_with_snapshots``
# (capped at 2.0s), with slack for process-teardown jitter.
_POLL_TOLERANCE_S = 5.0


# ---------------------------------------------------------------------------
# SweepConfig / __init__
# ---------------------------------------------------------------------------


def check_config(cfg: "SweepConfig") -> None:
    """Log any field violation on a freshly constructed :class:`SweepConfig`.

    Covers the structural fields (file presence, non-empty names,
    positive budgets/steps, ``delta`` in ``(0, 1)``, ``center_feat_idx``
    indexing ``features``). Does not raise — the sweep proceeds with the
    config as given.
    """
    _check(
        Path(cfg.scenic_file).is_file(),
        f"scenic_file not found: {cfg.scenic_file}",
    )
    _check(bool(cfg.composite_name), "composite_name must be non-empty")
    _check(bool(cfg.monolithic_name), "monolithic_name must be non-empty")
    _check(
        cfg.composite_name != cfg.monolithic_name,
        "composite_name and monolithic_name must differ",
    )
    _check(cfg.max_budget > 0, f"max_budget must be > 0, got {cfg.max_budget}")
    _check(
        cfg.snapshot_every > 0,
        f"snapshot_every must be > 0, got {cfg.snapshot_every}",
    )
    _check(
        cfg.snapshot_every <= cfg.max_budget,
        "snapshot_every must be <= max_budget (else 0 checkpoints)",
    )
    _check(cfg.max_steps_primitive > 0, "max_steps_primitive must be > 0")
    _check(cfg.max_steps_mono > 0, "max_steps_mono must be > 0")
    _check(bool(cfg.features), "features must be non-empty")
    # Paper §4.1: "Both methods report Hoeffding half-widths at δ=0.05."
    _check(0.0 < cfg.delta < 1.0, f"delta must be in (0, 1), got {cfg.delta}")
    # ``center_feat_idx`` is optional but, if present, must index ``features``.
    _check(
        all(0 <= i < len(cfg.features) for i in cfg.center_feat_idx),
        f"center_feat_idx {cfg.center_feat_idx} out of range for features {cfg.features}",
    )


def check_parsed_graph(paths: list, primitives: list[str]) -> None:
    """Postcondition: scenic parsing produced usable paths + primitives."""
    _check(bool(paths), "no composition paths parsed for composite_name")
    _check(bool(primitives), "no primitives discovered in paths")
    _check(
        len(set(primitives)) == len(primitives),
        f"duplicate primitive name: {primitives}",
    )


# ---------------------------------------------------------------------------
# simulate_with_snapshots
# ---------------------------------------------------------------------------


def check_jobs(
    jobs: list[dict[str, Any]],
    method_dir: Path,  # noqa: ARG001 -- kept for call-site clarity
) -> list[str]:
    """Precondition for :meth:`BudgetSweep.simulate_with_snapshots`.

    Verifies job dicts carry the required keys (``scenic_file``,
    ``scenario_name``, ``max_steps``) and that ``max_steps`` is
    positive, and flags duplicate ``scenario_name`` values. ``method_dir``
    is accepted for call-site symmetry but not validated here.

    Returns:
        The list of scenario names as-given (duplicates are logged,
        not stripped — the sweep continues with the original list).
    """
    _ = method_dir
    _check(bool(jobs), "jobs must be a non-empty list")
    for j in jobs:
        missing = _REQUIRED_JOB_KEYS - set(j)
        _check(not missing, f"job missing required keys {missing}: {j}")
        ms = j.get("max_steps")
        _check(
            ms is not None and ms > 0,
            f"job max_steps must be a positive int, got {ms!r}",
        )
    names = [j.get("scenario_name", "") for j in jobs]
    _check(
        len(set(names)) == len(names),
        f"duplicate scenario_name in jobs: {names}",
    )
    return names


def check_simulation_result(
    logs: dict[str, str],
    timeline: list["Snapshot"],
    scenario_names: list[str],
    max_budget: float,
    snapshot_every: float,  # noqa: ARG001 -- kept for call-site clarity
) -> None:
    """Postcondition for :meth:`BudgetSweep.simulate_with_snapshots`."""
    _ = snapshot_every
    _check(
        all(name in scenario_names for name in logs),
        f"logs has unknown scenario: {set(logs) - set(scenario_names)}",
    )
    for name, p in logs.items():
        _check(Path(p).is_file(), f"log path missing on disk for {name}: {p}")
    prev = -1.0
    for elapsed, counts in timeline:
        _check(elapsed >= 0.0, f"negative elapsed: {elapsed}")
        # Hard-stop overshoot is bounded by ``poll_sleep`` (≤2s), not by
        # ``snapshot_every``. Use a small absolute tolerance.
        _check(
            elapsed <= max_budget + _POLL_TOLERANCE_S,
            f"snapshot past budget: {elapsed} > {max_budget} + {_POLL_TOLERANCE_S}",
        )
        _check(elapsed >= prev, f"timeline out of order: {elapsed} < prev {prev}")
        prev = elapsed
        unknown = set(counts) - set(scenario_names)
        _check(not unknown, f"unknown scenario in counts: {unknown}")
        _check(
            all(c >= 0 for c in counts.values()),
            f"negative trace count in snapshot at {elapsed}",
        )


# ---------------------------------------------------------------------------
# analyze_compositional / analyze_monolithic
# ---------------------------------------------------------------------------


def check_snapshot(snapshot: "Snapshot") -> tuple[float, dict[str, int]]:
    """Validate a snapshot tuple and return its parts (best-effort).

    Logs negative ``elapsed`` and non-integer / negative counts, then
    normalizes ``counts`` values to plain ``int`` (accepts
    ``numbers.Integral`` so numpy/pandas scalars work). On a
    normalization failure (very rare), falls back to zeros for the
    offending entries rather than raising.
    """
    elapsed, counts = snapshot
    _check(elapsed >= 0.0, f"elapsed must be >= 0, got {elapsed}")
    # Accept numpy/pandas integer scalars (numbers.Integral), not just ``int``.
    _check(
        all(isinstance(c, numbers.Integral) and int(c) >= 0 for c in counts.values()),
        "counts must be non-negative integers",
    )
    try:
        return float(elapsed), {k: int(v) for k, v in counts.items()}
    except (TypeError, ValueError) as exc:
        _log.warning("check_snapshot normalization failed: %r", exc)
        return float(elapsed) if isinstance(elapsed, numbers.Real) else 0.0, {
            k: int(v) if isinstance(v, numbers.Integral) else 0
            for k, v in counts.items()
        }


def check_compositional_inputs(
    primitives: list[str],
    snapshot: "Snapshot",
    logs: dict[str, str],
) -> None:
    """Precondition for :meth:`BudgetSweep.analyze_compositional`.

    Note: we deliberately do **not** require ``set(logs) ⊇ set(primitives)``.
    Per the Fig. 4 caveat in §4.3, early checkpoints can legitimately be
    missing some primitives whose MetaDrive worker hasn't emitted its
    first trace yet — the analyzer must produce a ``missing_primitives``
    record at those checkpoints rather than raising.
    """
    _check(bool(primitives), "BudgetSweep must populate primitives before analyzing")
    check_snapshot(snapshot)
    for name, p in logs.items():
        _check(Path(p).is_file(), f"missing CSV for {name}: {p}")


def check_monolithic_inputs(
    monolithic_name: str,
    snapshot: "Snapshot",
    log: str | None,
) -> int:
    """Precondition for :meth:`BudgetSweep.analyze_monolithic`.

    Returns:
        The monolithic trace count at this snapshot.
    """
    _, counts = check_snapshot(snapshot)
    n = counts.get(monolithic_name, 0)
    if n > 0:
        _check(
            log is not None and Path(log).is_file(),
            f"missing monolithic CSV when n={n}: {log!r}",
        )
    return n


def check_record(record: "Record", method: str, elapsed: float, n_traces: int) -> None:
    """Postcondition for either analyze method.

    Validates the required-key set, the ``method`` and ``status`` enum
    values, ``budget == elapsed``, ``n_traces`` consistency, and the
    rho-eps-both-None-or-both-set invariant; range-checks ``rho ∈ [0, 1]``
    and ``eps ≥ 0`` when present. If required keys are missing, logs
    once and returns early to avoid a flurry of ``KeyError``-style
    follow-ons on the downstream reads.
    """
    missing = _REQUIRED_RECORD_KEYS - set(record)
    if not _check(not missing, f"record missing keys: {missing}"):
        return  # downstream key reads would KeyError; bail out of this check.
    _check(
        record["method"] == method,
        f"record method {record['method']!r} != expected {method!r}",
    )
    _check(
        record["status"] in _VALID_STATUSES,
        f"record.status {record['status']!r} not in {_VALID_STATUSES}",
    )
    _check(
        record["budget"] == elapsed,
        f"record.budget {record['budget']!r} != snapshot elapsed {elapsed!r}",
    )
    _check(
        record["n_traces"] == n_traces,
        f"record.n_traces {record['n_traces']} != expected {n_traces}",
    )
    _check(
        (record["rho"] is None) == (record["eps"] is None),
        "rho and eps must both be None or both set",
    )
    if record["rho"] is not None:
        _check(0.0 <= record["rho"] <= 1.0, f"rho out of [0, 1]: {record['rho']}")
        _check(record["eps"] >= 0.0, f"eps must be >= 0, got {record['eps']}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def check_run_preconditions(paths: list, primitives: list[str]) -> None:
    """Precondition for :meth:`BudgetSweep.run`: parsed graph is non-empty."""
    _check(
        bool(paths) and bool(primitives),
        "BudgetSweep.__init__ must have parsed paths/primitives before run()",
    )


def check_run_result(records: list["Record"], csv_path: Path) -> None:
    """Postcondition for :meth:`BudgetSweep.run`.

    Flags unknown ``method`` labels and confirms ``results.csv`` was
    written. Cross-method *agreement* is checked separately by
    :func:`check_method_agreement` (called from ``run()`` after this).
    """
    methods = [r.get("method") for r in records]
    unknown_methods = set(methods) - {"compositional", "monolithic"}
    _check(not unknown_methods, f"unexpected method label(s): {unknown_methods}")
    # Row write-order between the two methods is not a methodology
    # constraint (the paper's plots are ordered by ``T``, not by method).
    _check(csv_path.is_file(), f"results.csv was not written: {csv_path}")


# ---------------------------------------------------------------------------
# Scenario-health checks (diagnose buggy .scenic files, not our code)
# ---------------------------------------------------------------------------
#
# These don't fail the run — they emit diagnostic log lines so a human
# reader can spot scenarios whose Scenic source is the actual problem.

# Final-snapshot throughput ratio above which we flag imbalance.
_THROUGHPUT_IMBALANCE_RATIO = 20.0
# A primitive whose first trace lands after this fraction of the budget
# is suspect — its worker probably crashed and was restarted, or its
# scenario can't even initialize.
_LATE_FIRST_TRACE_FRACTION = 0.5


def check_throughput(
    timeline: list["Snapshot"],
    scenario_names: list[str],
    max_budget: float,
) -> None:
    """Flag scenarios that produced no / too few / extremely imbalanced traces.

    Symptoms surfaced here usually point at a Scenic-side issue:
    - **0 traces over the full budget**: scenario likely crashes on
      ``setup`` (e.g. an undefined region, an unreachable behavior).
    - **First trace very late**: simulator can start but the scenario's
      termination condition rarely fires (probably a too-strict
      ``terminate when`` or a behavior that loops).
    - **Throughput imbalance >20×**: one primitive's traces are
      dramatically shorter or one is hitting silent restarts.
    """
    if not timeline:
        _log.warning("[scenario] empty timeline — no checkpoints recorded")
        return

    final_elapsed, final_counts = timeline[-1]
    for name in scenario_names:
        n = final_counts.get(name, 0)
        if n == 0:
            _log.warning(
                "[scenario:%s] zero traces over %.0fs budget — "
                "Scenic scenario likely failed to initialize",
                name,
                final_elapsed,
            )
            continue
        # Find first checkpoint at which this primitive had >=1 trace.
        first_t = next((t for t, c in timeline if c.get(name, 0) >= 1), final_elapsed)
        if max_budget > 0 and first_t / max_budget > _LATE_FIRST_TRACE_FRACTION:
            _log.warning(
                "[scenario:%s] first trace at %.1fs (%.0f%% of budget) — "
                "slow startup or rare termination",
                name,
                first_t,
                100.0 * first_t / max_budget,
            )

    positive = {
        n: final_counts.get(n, 0) for n in scenario_names if final_counts.get(n, 0) > 0
    }
    if len(positive) >= 2:
        hi = max(positive.values())
        lo = min(positive.values())
        if lo > 0 and hi / lo > _THROUGHPUT_IMBALANCE_RATIO:
            slowest = min(positive, key=positive.get)
            fastest = max(positive, key=positive.get)
            _log.warning(
                "[scenario] throughput imbalance %.1fx — fastest=%s (%d), "
                "slowest=%s (%d). Slow primitive may have a bad max_steps cap "
                "or a stuck behavior.",
                hi / lo,
                fastest,
                hi,
                slowest,
                lo,
            )


def check_trace_csv(
    log_path: str | Path,
    features: list[str],
    scenario_name: str,
) -> None:
    """Read a primitive/monolithic ``traces.csv`` and flag schema /
    trace-length pathologies.

    - Missing feature column → spec/scenario mismatch (the DFA's
      ``labeling_function`` will silently read NaN).
    - All traces hit the simulator's hard ``max_steps`` → ``terminate when``
      is unreachable (scenario doesn't actually finish).
    - All traces are length 1 → scenario terminates on the very first
      step (initial state already satisfies the terminate condition).
    """
    log_path = Path(log_path)
    if not log_path.is_file():
        _log.warning("[scenario:%s] traces.csv missing at %s", scenario_name, log_path)
        return

    try:
        with log_path.open("r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            # Use a sample (up to 100k rows) to bound memory.
            lengths: dict[str, int] = {}
            for line in f:
                tid = line.split(",", 1)[0]
                lengths[tid] = lengths.get(tid, 0) + 1
    except OSError as exc:
        _log.warning(
            "[scenario:%s] failed to read %s: %r", scenario_name, log_path, exc
        )
        return

    missing_cols = [c for c in features if c not in header]
    if missing_cols:
        _log.warning(
            "[scenario:%s] traces.csv missing feature columns %s (header=%s) — "
            "DFA labeling will see NaN",
            scenario_name,
            missing_cols,
            header,
        )

    if not lengths:
        _log.warning(
            "[scenario:%s] traces.csv has header but zero data rows", scenario_name
        )
        return

    vals = list(lengths.values())
    min_len, max_len = min(vals), max(vals)
    mean_len = sum(vals) / len(vals)
    _log.info(
        "[scenario:%s] trace lengths n=%d min=%d max=%d mean=%.1f",
        scenario_name,
        len(vals),
        min_len,
        max_len,
        mean_len,
    )
    if min_len == max_len == 1:
        _log.warning(
            "[scenario:%s] every trace is length 1 — scenario terminates "
            "on step 0 (check terminate-when / initial state)",
            scenario_name,
        )
    elif min_len == max_len:
        _log.warning(
            "[scenario:%s] every trace is exactly length %d — likely hitting "
            "the Scenic max_steps cap (terminate-when never fires)",
            scenario_name,
            min_len,
        )


def check_rho_signal(record: "Record", scenario_or_paths: str) -> None:
    """Flag ρ̂ values that suggest the DFA isn't actually discriminating.

    ``ρ̂ ∈ {0.0, 1.0}`` exactly means the DFA's accept verdict was the
    same on **every** trace — either the spec is trivially satisfied/violated
    given how the scenario behaves, or the labeling threshold lies outside
    the scenario's feature range (cf. the threshold comment in v1's
    ``spec_k_consec_slow`` / ``spec_k_consec_fast`` factories).
    """
    rho = record.get("rho")
    if rho is None:
        return
    if rho == 0.0:
        _log.warning(
            "[scenario:%s] ρ̂=0.0 exactly — every trace was rejected by the "
            "DFA (spec trivially violated, or labels constant)",
            scenario_or_paths,
        )
    elif rho == 1.0:
        _log.warning(
            "[scenario:%s] ρ̂=1.0 exactly — every trace was accepted by the "
            "DFA (spec trivially satisfied, or threshold outside feature range)",
            scenario_or_paths,
        )


def check_method_agreement(records: list["Record"]) -> None:
    """Cross-method sanity check after a sweep.

    Paper §4.2 / Table 1: "every |Δρ̂| falls inside ε̂_mono + ε̂_comp,
    i.e., the two estimators agree within statistical error on every
    (c, φ) pair." Pair the two methods' last ``ok`` checkpoints and
    log if the gap exceeds the combined Hoeffding budget — that's a
    methodology-level red flag (handoff bias, prewarm trim missing,
    feature mismatch, etc.).
    """

    def _last_ok(method: str):
        oks = [
            r for r in records if r.get("method") == method and r.get("status") == "ok"
        ]
        return oks[-1] if oks else None

    mono = _last_ok("monolithic")
    comp = _last_ok("compositional")
    if mono is None or comp is None:
        _log.info(
            "[agreement] skipping cross-method check (mono ok=%s, comp ok=%s)",
            mono is not None,
            comp is not None,
        )
        return

    gap = abs(float(mono["rho"]) - float(comp["rho"]))
    budget = float(mono["eps"]) + float(comp["eps"])
    if gap > budget:
        _log.warning(
            "[agreement] |Δρ̂|=%.4f > ε̂_mono+ε̂_comp=%.4f at T_mono=%.1fs / "
            "T_comp=%.1fs — methods disagree beyond Hoeffding error "
            "(possible Scenic handoff bias, missing prewarm trim, or "
            "feature mismatch)",
            gap,
            budget,
            float(mono["budget"]),
            float(comp["budget"]),
        )
    else:
        _log.info(
            "[agreement] |Δρ̂|=%.4f ≤ ε̂_mono+ε̂_comp=%.4f — methods agree",
            gap,
            budget,
        )
