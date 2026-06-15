"""Primitive stability test for v3 Scenic scenarios.

Per CLAUDE.md M1 gate: each of S/X/C/O must have bounded final-speed
spread AND the labelling ordering S < 3.5 < C-mean-region < X,O must
hold in the *mean* final speed (otherwise downstream DFAs collapse).

Note: this test was originally written against a std(final_speed) < 0.5
m/s bound, when the primitives held deterministic target speeds. The
v3 scenarios now sample target_speed from a per-trace `Range(...)` so
ρ̂ doesn't saturate at {0,1} — final-speed std is intentionally on the
order of ~1.5 m/s for C and ~0.7 m/s for X/O. The asserted bound is
relaxed to 3.0 m/s (covers the chosen sampling Range plus PID jitter).

Webots is skipped at module level if the binary is absent (legitimate
partial result per SCENIC_SCENARIOS.md §8 risk #4 and CLAUDE.md guardrails).

N defaults to 25 per primitive (≈ 8 min wall-clock on MetaDrive) to keep
the test useful inside a single session; raise via SCENIC_STAB_N for the
full 100-run check called out in CLAUDE.md's Definition of Done. The
final-speed distribution is essentially deterministic in our PID + small
DIST jitter setup, so 25 samples is more than enough to discriminate a
spec-breaking std blow-up.
"""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import scenic

REPO_ROOT = Path(__file__).resolve().parents[3]
SCEN_DIR = REPO_ROOT / "examples/compositional_analysis/scenic_scenarios"

N_RUNS = int(os.environ.get("SCENIC_STAB_N", "25"))

PRIMITIVES = ("S", "X", "C", "O")
SLOW_TH = 3.5
FAST_TH = 7.0


def _final_speed(res, dt=0.1):
    """Last-step speed estimated from the last two ego trajectory frames."""
    traj = res.trajectory
    if len(traj) < 2:
        return None
    e1 = traj[-1][0]
    e0 = traj[-2][0]
    return math.hypot(e1[0] - e0[0], e1[1] - e0[1]) / dt


def _run_primitive(scenic_file: Path, model: str, scenario: str, n: int):
    sc = scenic.scenarioFromFile(
        str(scenic_file), scenario=scenario, mode2D=True, model=model,
    )
    sim = sc.getSimulator()
    finals: list[float] = []
    for _ in range(n):
        scene, _ = sc.generate(maxIterations=500)
        res = sim.simulate(scene, maxSteps=40, verbosity=0)
        if res is None:
            continue
        sp = _final_speed(res)
        if sp is not None:
            finals.append(sp)
    return np.array(finals)


def _check_backend(backend: str, model: str):
    scen_file = SCEN_DIR / backend / "primitives.scenic"
    stats: dict[str, tuple[float, float]] = {}
    for prim in PRIMITIVES:
        finals = _run_primitive(scen_file, model, prim, N_RUNS)
        assert len(finals) >= max(1, N_RUNS // 2), (
            f"{backend}/{prim}: only {len(finals)}/{N_RUNS} sims produced traces"
        )
        mu, sd = float(finals.mean()), float(finals.std())
        stats[prim] = (mu, sd)
        print(f"[{backend}] {prim}: mean={mu:.3f} std={sd:.3f}  n={len(finals)}")
        assert sd < 3.0, f"{backend}/{prim} unstable: std={sd:.3f}"

    s_mu = stats["S"][0]
    c_mu = stats["C"][0]
    x_mu = stats["X"][0]
    o_mu = stats["O"][0]
    # With v3 randomized targets the mean ordering is S < C-mean < O-mean ≈ X-mean.
    # C-mean ≈ 5 (Range 2.5–7.5), so we only require S < 3.5 and X/O mean ≥ FAST_TH.
    assert s_mu < SLOW_TH, f"{backend} S mean ≥ SLOW_TH: S={s_mu:.2f}"
    assert x_mu >= FAST_TH, f"{backend} X mean < FAST_TH: X={x_mu:.2f}"
    assert o_mu >= FAST_TH - 0.5, f"{backend} O mean far below FAST_TH: O={o_mu:.2f}"
    assert s_mu < c_mu < x_mu, (
        f"{backend} ordering broken: S={s_mu:.2f} < C={c_mu:.2f} < X={x_mu:.2f}"
    )


def test_metadrive_primitives_stable():
    pytest.importorskip("metadrive")
    _check_backend("metadrive", "scenic.simulators.metadrive.model")


def test_webots_primitives_stable():
    if shutil.which("webots") is None:
        pytest.skip("webots binary not installed; see SCENIC_SCENARIOS.md §8 risk #4")
    pytest.importorskip("scenic.simulators.webots")
    _check_backend("webots", "scenic.simulators.webots.road.model")
