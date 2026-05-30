"""Data layer: :class:`SweepConfig` and shared type aliases.

Lives apart from :mod:`budget_sweep.sweep` so the checks module can
``TYPE_CHECKING``-import these names without dragging the engine in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verifai.monitor import automaton_specification

# (elapsed_seconds, {primitive_name: completed_trace_count})
Snapshot = tuple[float, dict[str, int]]
# One analyzed checkpoint row written to results.csv.
Record = dict[str, Any]


@dataclass
class SweepConfig:
    """Inputs for a single budget sweep.

    Attributes:
        scenic_file: Path to the ``.scenic`` file containing both the
            compositional scenarios and the monolithic counterpart.
            :meth:`BudgetSweep.__init__` calls ``.resolve()`` so Scenic's
            relative-import resolution still works when worker
            subprocesses inherit a different ``cwd``.
        composite_name: Name of the composite scenario whose path set is
            parsed and analyzed compositionally.
        monolithic_name: Name of the monolithic scenario used as the
            baseline.
        spec: DFA specification used to relabel traces and compute ``ρ``.
            Built via ``verifai.monitor.automaton_specification``.
            §4.1 of the paper: both methods report ρ as the empirical
            DFA accept rate.
        max_budget: Wall-clock seconds **per method**. Compositional and
            monolithic each run for ``max_budget`` seconds, sequentially
            (see :meth:`BudgetSweep.run`).
        snapshot_every: Seconds between trace-count checkpoints during a
            simulation.
        max_steps_primitive: Default per-primitive Scenic ``max_steps``
            cap; overridable per primitive via ``max_steps_overrides``.
        max_steps_mono: Monolithic scenario ``max_steps`` cap.
        features: Trace columns the engine uses for KDE matching.
            **Required**, no default — different specs in the paper read
            different signals (e.g. ``["speed"]`` for ``safe_under_max``,
            ``["x","y","heading","speed"]`` for the engine's full handoff).
        center_feat_idx: Indices into ``features`` used as the KDE
            centering basis (Alg. 2 ``ϕ_src``/``ϕ_tgt``). Empty list
            disables centering.
        delta: Hoeffding confidence parameter; §4.1 sets ``δ=0.05``.
            Threaded into ``check_with_dfa_scenic`` (compositional) and
            the monolithic CI formula ``sqrt(log(2/δ)/(2N))``.
        max_steps_overrides: Per-primitive overrides for
            ``max_steps_primitive`` (e.g. Set C's ``Subscenario2{L,R,S}``
            need more ticks than the other primitives).
        prewarm_trim: Per-primitive count of leading rows to drop from
            each trace post-simulation (paper §3.2: prewarming so
            successive supports overlap). Empty dict disables trimming.
        save_dir: Output directory. Layout:
            ``<save_dir>/results.csv``                  — analyzed records
            ``<save_dir>/compositional/<primitive>/traces.csv``
            ``<save_dir>/monolithic/<monolithic_name>/traces.csv``
    """

    scenic_file: Path
    composite_name: str
    monolithic_name: str
    spec: automaton_specification
    max_budget: float
    snapshot_every: float
    max_steps_primitive: int
    max_steps_mono: int
    features: list[str]
    center_feat_idx: list[int] = field(default_factory=list)
    delta: float = 0.05
    max_steps_overrides: dict[str, int] = field(default_factory=dict)
    prewarm_trim: dict[str, int] = field(default_factory=dict)
    save_dir: Path = Path("storage/budget_sweep_v2")
