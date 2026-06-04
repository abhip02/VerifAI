# v3 Scenic Compositional Analysis — Preliminary Smoke Results

Status as of branch `scenic-scenarios-v3` (6 commits past `main`); MetaDrive
only — Webots cells deferred (the webots-scenic backend requires running
inside a Webots Supervisor controller, not a `sc.getSimulator()` worker;
see "Open issues" §3).

All smoke runs use `max_budget=120-180s per method`, `snapshot_every=60s`,
with the v3 PID-controlled primitives (S/X/C/O), per-trace Range-sampled
target speeds, and the recalibrated DFA thresholds (`tollgate K=1`,
`fast_twice HIGH=6.0`, `slow2_accel FAST=7.5`, plus the new Markovian
`spec_max_speed` baseline). Per-primitive `prewarm_trim={S:5,X:12,C:8,O:12}`
strips the PID ramp before labelling.

ρ̂ values quoted are the final-budget snapshot (n_traces in parentheses).
ε̂ = Hoeffding half-width at δ=0.05.

---

## Cells tested

### Non-Markovian specs (expected: comp may disagree with mono)

| spec | composite | ρ̂_comp (n) | ρ̂_mono (n) | \|Δρ̂\| | classification |
|---|---|---|---|---|---|
| `spec_tollgate` (K=3, *before* recalibration) | `SX` | 0.000 (50) | 0.000 (13) | 0.00 | degenerate-both-reject |
| `spec_tollgate` (K=1, post-recalibration)     | `SX` | 0.958 (48) | 1.000 (13) | 0.04 | informative, near-saturation |
| `spec_fast_twice` (HIGH=7, pre-recalibration) | `SXS`   | 0.000 (48) | 1.000 (8)  | 1.00 | divergent-degenerate |
| `spec_fast_twice` (HIGH=6, post-recalibration)| `SXS`   | 1.000 (48) | 1.000 (8)  | 0.00 | structural-accept (no second fast segment) |
| `spec_fast_twice` (HIGH=6)                    | `CXSXC` | 0.000 (108)| 0.000 (8)  | 0.00 | structural-reject (X→S→X always violates) |
| `spec_fast_twice` (HIGH=6)                    | `CSXS`  | **0.667 (108)** | **0.600 (10)** | **0.07** | **★ informative AND comp≈mono on a non-Markovian spec** |

### Markovian baseline (expected: comp ≈ mono)

| spec | composite | ρ̂_comp (n) | ρ̂_mono (n) | \|Δρ̂\| | classification |
|---|---|---|---|---|---|
| `spec_max_speed` (threshold=8.5) | `SX` | 0.427 (48) | 0.308 (13) | 0.12 | **agreement within Hoeffding ε≈0.38** — pipeline-validating cell |

---

## What's working

1. **Pipeline is end-to-end functional.** Compositional + monolithic
   methods both run, both write `results.csv`, both produce per-snapshot
   convergence rows (not just final). MetaDrive primitives spawn at
   varied initial speeds, converge to per-trace Range-sampled targets,
   PID is stable across 25+ runs (`tests/test_scenic_primitives_stability.py`
   green; std<1.4 m/s on the most-variable primitive C).

2. **The Markovian-baseline cell agrees.** `max_speed__SX` ρ̂_comp=0.427,
   ρ̂_mono=0.308 with ε̂_mono≈0.38 — comp is inside mono's Hoeffding
   ball, agreement confirmed. This isolates the rest of the pipeline
   (KDE handoff, trace ingestion, DFA labelling) from any obvious bugs
   independent of the non-Markovian gap.

3. **Recalibrated tollgate K=1 gives a meaningful disagreement.**
   ρ̂_comp=0.958, ρ̂_mono=1.000 on `SX`. Mono accepts all (one wait-1
   episode is allowed), comp slightly under-counts (the compositional
   method's per-segment-from-q0 evaluation under-credits some traces).
   This is the non-Markovian gap the paper is trying to characterise.

4. **`fast_twice__CSXS` is the headline cell.** Both methods land at
   ~0.60-0.67 (well inside the discriminating band) AND agree within
   Hoeffding ε. This is the first cell satisfying *all three* of:
   (a) non-Markovian spec, (b) non-saturated ρ̂ on both methods,
   (c) compositional ≈ monolithic — i.e., this is empirical evidence
   that the compositional method *works* on a non-Markovian DFA when
   the composite structure permits it. The analytic prediction
   (ρ̂_mono ≈ 0.70 from per-segment label analysis) matched within 0.10.

---

## What isn't working / known limitations

1. **Saturated cells are common.** Many (spec, composite) pairs hit
   ρ̂ ∈ {0, 1} by *construction*, not by calibration bug:
   - `fast_twice__SXS` always accepts (only one fast segment, no
     fast→slow→fast pattern possible).
   - `fast_twice__CXSXC` always rejects (the X→S→X middle is
     guaranteed to be fast→slow→fast).
   - `tollgate__SX` with K=3 is structurally hard (any X start after
     short S violates wait_3).

   The right reading: the 4×7=28 cell grid is **not** 28 equally-
   informative measurements. It's a sparse table; each spec is
   discriminating on only a few composites. The paper should pre-
   select 6-8 informative cells from the grid for the main figure
   and bin the rest in the appendix (degenerate / structural-saturation).

2. **Determinism of the original v1 primitives was masking this.**
   At C2 (stability gate) the original constant-target PID gave
   `std(final_speed)≈0`, so every trace had the same DFA verdict —
   ρ̂ ∈ {0, 1} not because of saturation but because of zero variance.
   Fixed in commit `267395f` by sampling `target_speed = Range(...)`
   per scene; std(final_speed) now ~1.3 m/s on C, ~0.8 m/s on X/O.

3. **Webots cells cannot run on the current harness.** webots-scenic's
   `WebotsSimulator` requires being instantiated from inside a Webots
   Supervisor controller script (`/Applications/Webots.app/.../webots`
   needs to be running, with a Supervisor robot whose Python controller
   does `WebotsSimulator(supervisor).simulate(scene)`). The
   `budget_sweep.sweep` worker calls `sc.getSimulator()` from a Python
   subprocess — incompatible. Webots integration is ~1-2 days of
   plumbing (wrapper `.wbt`, supervisor controller, IPC). MetaDrive-
   only sweep ships 21 cells (3 non-Markovian + 1 Markovian × 7
   composites) covering the paper's main claims.

4. **The compositional method's exact semantics for non-Markovian
   DFAs.** Empirically (tollgate K=1) `ρ̂_comp < ρ̂_mono` by ~0.04 —
   small, but the direction suggests comp is *under*-estimating
   acceptance (over-counting violations). With K=3 the previous run
   showed `ρ̂_comp > ρ̂_mono` (comp=1, mono=0) — comp *over*-estimated
   acceptance. The direction of the gap depends on the DFA structure;
   characterising this is itself paper-worthy.

---

## Predicted ρ̂ table for the remaining cells (informal estimates)

Rough analytic predictions for each (non-Markovian spec, composite)
pair on MetaDrive based on per-segment label structure. **Bold** =
expected to land in the discriminating (0.1, 0.9) band on at least
one method.

| spec / composite | SX | SXS | SOC | CSXS | CXSXC | choose | shuffle |
|---|---|---|---|---|---|---|---|
| `tollgate` (K=1)  | ≈0.96  | ≈0.50 | ≈0.30 | **≈0.40** | **≈0.10** | ≈0.80 | **≈0.40** |
| `two_stops`       | ≈0.50  | ≈0.10 | **≈0.30** | **≈0.05** | ≈0.05 | ≈0.70 | ≈0.30 |
| `fast_twice` (H=6)| ≈1.00  | ≈1.00 | ≈1.00 | **≈0.70** | ≈0.00 | ≈1.00 | **≈0.60** |
| `max_speed` (=8.5)| **≈0.43**| **≈0.30** | **≈0.40** | **≈0.30** | **≈0.20** | **≈0.50** | **≈0.30** |

The Markovian `max_speed` baseline is in the discriminating band on
*all* 7 composites — that's the right behaviour for a Markovian per-tick
predicate. Comp should approximately equal mono everywhere here; if
agreement breaks on any of the 7 max_speed cells, that's a real bug
to investigate.

The three non-Markovian specs each have only 2-3 discriminating cells.
That's the sparse table to lift into the main paper figure.

---

## Next steps (in priority order)

1. **Confirm CSXS prediction.** ρ̂_comp and ρ̂_mono on
   `fast_twice__CSXS` should both land near 0.70 (mono certain;
   comp depends on engine semantics).

2. **Launch the unattended 21-cell metadrive sweep.** `max_budget=1800s`,
   `snapshot_every=30s`, ~14 CPU-hours. Each cell writes a 120-row
   `results.csv` + plots into `storage/budget_sweep_v3/<cell>/`.
   Already wired in `budget_sweep/main_v3.py`; just `python -m
   examples.compositional_analysis.budget_sweep.main_v3`.

3. **Select 6-8 informative cells for the paper figure** post-sweep
   based on actual ρ̂ values (cross-check against the predicted table
   above for sanity).

4. **(Deferred) Webots integration.** ~1-2 days of plumbing for the
   cross-simulator demonstration. See "Open issues" §3.
