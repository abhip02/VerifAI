# SCENIC_SCENARIOS.md

> **Backend assignment**: 3 specs on **MetaDrive + Scenic** (using the same
> Town07 map already used by `wander_scenarios.scenic`), 1 spec on
> **Webots + Scenic** (uses a Webots `.wbt` world — Webots Scenic does not
> read XODR, so Town07 is MetaDrive‑only; see §2.1).

Design proposal for the Scenic primitives, composite scenarios, and budget‑sweep
configurations that populate Arya's **fixed‑time‑budget** experiments (and, by
construction, Abhi's **fixed‑trace‑budget** table) for the compositional
analysis paper.

Source survey of upstream Scenic: `/tmp/scenic_survey.md` (originally
recommended Newtonian as the easiest backend; this document overrides that
choice in favour of **MetaDrive (3 specs) + Webots (1 spec)** per Arya's
direction, while keeping the same `scenic.domains.driving` +
`FollowLaneBehavior` + PID stack the survey identified as the stable
closed‑loop core).

---

## 1. Why a new scenario family

`wander_scenarios.scenic` is unstable for three reasons (open‑loop random
throttle → uncontrolled speed variance; per‑primitive ego respawn so KDE
handoff has only `speed` to bridge on; random prewarm that pollutes the
"steady state"). The PG‑block primitives in the tollgate/cosafety tests are
stable but live in MetaDrive's native config, which Scenic's adapter cannot
drive (it only accepts SUMO/XODR). We therefore need a new family that is
**Scenic‑native end‑to‑end** and **closed‑loop** by construction.

The design here uses the upstream `FollowLaneBehavior` (PID longitudinal +
PID lateral, auto‑tuned via each backend's `getLaneFollowingControllers`
binding in `scenic.domains.driving`) running on **Town07** (MetaDrive) and
**`simple.wbt`** (Webots). Each primitive is parameterised by a target speed
(and, where relevant, a target lane); the random variable is the *parameter*,
not raw actuation. KDE handoffs work because the ego is shared across
primitives and three continuous features (`x`, `y`, `speed`) are all
meaningful at boundaries.

---

## 2. Backends + maps

Two simulator backends in this paper, assigned per spec:

| Backend | Specs | Driving model | Map |
|---------|-------|---------------|-----|
| **MetaDrive + Scenic** | tollgate, two_stops, fast_twice (3 specs) | `model scenic.simulators.metadrive.model` (+ `scenic.domains.driving.model` via the metadrive driving_model variant used in `wander_scenarios.scenic`) | `tests/scenic/scenic_tests/cases_realistic/CARLA/Town07.xodr` — the same Town07 the wander example already uses |
| **Webots + Scenic** | slow2_accel (1 spec) | `model scenic.simulators.webots.road.model` | a `.wbt` world (see §2.1) — Webots Scenic does **not** read XODR |

The split is 3 MetaDrive cells per primitive (`tollgate`, `two_stops`,
`fast_twice` already have working MetaDrive harnesses in
`dfa_tests/test_check_with_dfa_*`) and the remaining `slow2_accel` cell on
Webots as the cross‑simulator demonstration. If you prefer a different spec on
Webots, swap the row — nothing downstream depends on which one.

- **Controllers**: `getLaneFollowingControllers` from the active driving
  model — both backends bind it through `scenic.domains.driving`, so the
  primitive behaviors are written once and run on either backend without
  change.
- **Timestep**: 0.1 s (matches `wander_scenarios.scenic`, `train.py`, and the
  budget‑sweep pipeline).

### 2.1 The Webots map caveat

Webots Scenic uses `.wbt` Webots world files plus an OpenDRIVE companion
parsed by `scenic.simulators.webots.road.world`. The bundled
`simulators/webots/road/*.wbt` examples (`simple.wbt`, `mcity.wbt`,
`berkeley.wbt`, `richmond.wbt`, `southside2.wbt`) come with their own
geometry. **We cannot use `Town07.xodr` directly in Webots.** Two options:

1. **(Preferred, low effort)** Use `simple.wbt` (a straight road) for the
   Webots cell. Acknowledge in §"Implementation (VerifAI)" that the Webots
   experiment validates the *backend portability* of the primitives, not a
   like‑for‑like map comparison. This is the honest answer the paper can make.
2. **(High effort)** Author a `town07_like.wbt` that mimics Town07's 4‑way
   intersection geometry. Requires hand‑editing a Webots world file + a
   matching `.xodr` for the road domain; not worth it unless the paper
   explicitly claims same‑map cross‑simulator results.

Default: option 1. The README from here on assumes `simple.wbt`.

---

## 3. The four primitives — S, X, C, O

All primitives are full `scenario` blocks (not bare behaviors) so that
`analyze_scenic_composition` / `parse_scenic_spec` treats them as nodes in the
composition graph, matching the pattern in `composed_scenarios.scenic`. Each
spawns its own ego at a constrained `following roadDirection from spawnPoint
for Range(...)` distance so per‑primitive position variance is bounded and
KDE handoff to the next primitive's spawn region overlaps.

| Primitive | Meaning | Target speed | Lane | Termination |
|-----------|---------|--------------|------|-------------|
| **S** | **Slow** — decelerate, then hold near‑stop | 1.5 m/s | inner | until `ego.speed < 0.4` then brake; total 40 ticks |
| **X** | **Fast cross** — accelerate to highway cruise | 9.0 m/s | inner | 40 ticks |
| **C** | **Cruise** — steady moderate speed | 5.0 m/s | inner | 40 ticks |
| **O** | **Overtake** — lane‑change to outer lane and accelerate | 7.0 m/s | outer | until lane‑centred on outer, then hold; 40 ticks |

Speeds chosen so each primitive's *final speed distribution* sits in a regime
that cleanly maps to the DFA alphabet used by every spec (see §5):
S < `STOP_THRESHOLD_MS`=3.5 < C < `FAST_THRESHOLD_MS`=7.0 ≤ X, O. This keeps
the DFA labelling unambiguous at handoff boundaries, which was the root cause
of the wander spec collapse.

### File layout

```
examples/compositional_analysis/scenic_scenarios/
├── metadrive/                            # Town07 cells (3 specs)
│   ├── primitives.scenic                 # 4 scenarios: S, X, C, O on Town07
│   ├── composites/
│   │   ├── seq_SX.scenic
│   │   ├── seq_SXS.scenic
│   │   ├── seq_SOC.scenic
│   │   ├── seq_CSXS.scenic
│   │   ├── seq_CXSXC.scenic
│   │   ├── native_choose.scenic          # S then choose{C,X,O}
│   │   └── native_shuffle.scenic         # S then shuffle{C,X,O}
│   └── monolithic/
│       ├── mono_SX.scenic
│       ├── mono_SXS.scenic
│       ├── mono_SOC.scenic
│       ├── mono_CSXS.scenic
│       ├── mono_CXSXC.scenic
│       ├── mono_choose.scenic
│       └── mono_shuffle.scenic
└── webots/                               # simple.wbt cell (1 spec)
    ├── world/
    │   └── simple.wbt                    # symlink to Scenic's bundled file
    ├── primitives.scenic                 # same 4 scenario names; webots driving model
    ├── composites/                       # same 7 file names as metadrive/
    └── monolithic/                       # same 7 file names as metadrive/
```

The `metadrive/` and `webots/` subtrees share **scenario names and primitive
APIs** but use different driving models and spawn‑lane discovery code (Town07
has 4‑way intersections; `simple.wbt` is a straight road). Sharing names
keeps the `time_budget` `EXPERIMENTS` table uniform.

Each composite file imports its sibling `primitives.scenic` and exposes
`scenario Main()` plus `scenario Mono<Name>()`. The `Mono*` scenario runs
the same behaviour chain in one ego/episode (no respawn) and is the
ground‑truth reference for both budget studies.

### Skeleton — `metadrive/primitives.scenic`

Spawn‑lane discovery copied verbatim from `wander_scenarios.scenic` so the ego
always starts on the same incoming right lane of Town07's first 4‑way
intersection.

```scenic
param map       = localPath('../../../dfa_tests/e2e_4way_example/4_way_intersection_scenic/'
                            '../../../../tests/scenic/scenic_tests/cases_realistic/CARLA/Town07.xodr')
param carla_map = localPath(...)   # same path; required by the metadrive driving_model
param timestep  = 0.1
param use2DMap  = True
model scenic.simulators.metadrive.model

# Same intersection / spawn-lane discovery as wander_scenarios.scenic
fourWayIntersection = filter(lambda i: i.is4Way, network.intersections)
intersec   = fourWayIntersection[0]
rightLanes = filter(lambda lane: all([s._laneToRight is None for s in lane.sections]),
                    intersec.incomingLanes)
startLane     = rightLanes[0]
uberSpawnPoint = startLane.centerline[-1]

DIST  = Range(-15, -5)              # bounded jitter for primitive independence
TICKS = 40

behavior SlowBehavior():
    while self.speed > 0.4:
        do FollowLaneBehavior(target_speed=1.5) for 1 steps
    take SetBrakeAction(1.0)
    while True:
        wait

behavior FastBehavior():    do FollowLaneBehavior(target_speed=9.0)
behavior CruiseBehavior():  do FollowLaneBehavior(target_speed=5.0)
behavior OvertakeBehavior():
    # Town07 incoming lanes have a same-direction neighbour; fall back to
    # FollowLaneBehavior on startLane with a higher target speed if not.
    do FollowLaneBehavior(target_speed=7.0)

scenario S():
    setup:
        ego = new Car following roadDirection from uberSpawnPoint for DIST,
              with behavior SlowBehavior()
        terminate after TICKS steps
    compose: while True: wait

scenario X(): ...   # FastBehavior, same template
scenario C(): ...   # CruiseBehavior, same template
scenario O(): ...   # OvertakeBehavior, same template
```

The closed‑loop `FollowLaneBehavior` is identical in spirit to what wander
*should* have used. The instability fix is the controller, not the map —
Town07 is fine as long as the ego is closing a speed loop instead of taking
random throttle.

### Skeleton — `webots/primitives.scenic`

```scenic
model scenic.simulators.webots.road.model

# Webots road model exposes a single road network parsed from the .wbt.
# Pick the first available lane as `startLane`; FollowLaneBehavior on the
# driving domain handles the rest.
startLane     = network.lanes[0]
uberSpawnPoint = startLane.centerline[5]

DIST  = Range(0, 4)
TICKS = 40

behavior SlowBehavior():        # identical body to metadrive
    while self.speed > 0.4:
        do FollowLaneBehavior(target_speed=1.5) for 1 steps
    take SetBrakeAction(1.0)
    while True:
        wait

behavior FastBehavior():   do FollowLaneBehavior(target_speed=9.0)
behavior CruiseBehavior(): do FollowLaneBehavior(target_speed=5.0)
behavior OvertakeBehavior(): do FollowLaneBehavior(target_speed=7.0)

# scenario S/X/C/O exactly as in metadrive/primitives.scenic
```

The point: the behaviour bodies are byte‑identical; only the model line and
the spawn discovery differ. This is what makes the cross‑simulator claim in
the paper meaningful — same Scenic semantics, different simulator backend.

---

## 4. The seven composites

All five sequential composites use exactly the same template as
`composed_scenarios.scenic:Main()` — a `compose:` block with a `do
Subscenario()` per step. The two native composites use `do choose` and
`do shuffle`, which are already supported by the parser (`scenic_parser.py:79,
111` for `_is_shuffle`; `scenic_composition_analysis.py:877` for `choose`).

| File | `composite_name` | Decomposition | `n_steps` | Approx ticks (monolithic) |
|------|------------------|---------------|-----------|---------------------------|
| `seq_SX.scenic`        | `SX`        | `[S, X]`            | 2 | 80 |
| `seq_SXS.scenic`       | `SXS`       | `[S, X, S]`         | 3 | 120 |
| `seq_SOC.scenic`       | `SOC`       | `[S, O, C]`         | 3 | 120 |
| `seq_CSXS.scenic`      | `CSXS`      | `[C, S, X, S]`      | 4 | 160 |
| `seq_CXSXC.scenic`     | `CXSXC`     | `[C, X, S, X, C]`   | 5 | 200 |
| `native_choose.scenic` | `SChooseCXO`| `S; choose{C,X,O}`  | 2 | 80 |
| `native_shuffle.scenic`| `SShuffleCXO`| `S; shuffle{C,X,O}`| 4 | 160 |

The `n_steps` column drives the analyzer's `min_traces` requirement (≥2 per
primitive when `n_steps > 1`).

### Sequential skeleton — `seq_SX.scenic`

```scenic
# metadrive/composites/seq_SX.scenic — webots/ variant is identical except
# for the `model …` and map params (matches its sibling primitives.scenic).
from primitives import S, X
param map       = localPath('../../../../../tests/scenic/scenic_tests/cases_realistic/CARLA/Town07.xodr')
param carla_map = localPath('../../../../../tests/scenic/scenic_tests/cases_realistic/CARLA/Town07.xodr')
model scenic.simulators.metadrive.model

scenario Main():
    compose:
        do S()
        do X()

scenario MonoSX():
    setup:
        ego = new Car following roadDirection from spawn for DIST,
              with behavior MonoSXBehavior()
        terminate after 80 steps
    compose: while True: wait

behavior MonoSXBehavior():
    # Same control law as S then X, but on a single ego — no respawn.
    while self.speed > 0.4:
        do FollowLaneBehavior(target_speed=1.5) for 1 steps
    take SetBrakeAction(1.0)
    for i in range(5):
        wait
    do FollowLaneBehavior(target_speed=9.0)
```

The longer composites (`CXSXC`, `CSXS`) follow exactly the same pattern with
more `do` calls and a longer `MonoBehavior`.

### Native — `native_choose.scenic`

```scenic
from primitives import S, C, X, O

scenario Main():
    compose:
        do S()
        do choose { C(): 1, X(): 1, O(): 1 }

scenario MonoSChooseCXO():
    # Sample one of C/X/O at scene construction; run end-to-end on one ego.
    chosen = Uniform("C", "X", "O")
    ...
```

### Native — `native_shuffle.scenic`

```scenic
scenario Main():
    compose:
        do S()
        do shuffle { C(): 1, X(): 1, O(): 1 }   # all three in some random order

scenario MonoSShuffleCXO():
    order = Uniform(("C","X","O"), ("C","O","X"), ("X","C","O"),
                    ("X","O","C"), ("O","C","X"), ("O","X","C"))
    # MonoBehavior dispatches on `order` to drive the three segments in sequence.
```

---

## 5. The four DFA specs (already implemented)

These are the exact DFAs from the four tests under
`examples/compositional_analysis/dfa_tests/`. The labelling function reads
`speed` only, so every primitive/composite above is observable.

### 5.1 Safety — Tollgate "mandatory wait" (`spec_tollgate`)

> Once `speed < 3.5 m/s`, must remain slow for ≥ 3 consecutive steps before
> speeding up again. Source:
> `test_check_with_dfa_metadrive_tollgate.py`.

### 5.2 Safety — "Two stops" (`spec_two_stops`)

> At most one near‑stop episode. A second `near_stop` is a violation. Source:
> `test_check_with_dfa_metadrive_two_stops.py`.

### 5.3 Co‑safety — "Fast twice" / V‑shape speed (`spec_fast_twice`)

> Eventually witness `fast → slow → fast` in order. Source:
> `test_check_with_dfa_cosafety_fast_twice.py`.

### 5.4 Co‑safety — "Slow‑2 then accel" (`spec_slow2_accel`)

> Eventually: two consecutive `slow` steps immediately followed by a `fast`
> step. Source: `test_check_with_dfa_cosafety_slow2_accel.py`.

The spec factory functions already live inside those test files; lift the
four `make_spec()` bodies into
`examples/compositional_analysis/scenic_scenarios/specs.py` so both Abhi's
fixed‑trace harness and Arya's `time_budget` configs import the same source
of truth.

---

## 6. Wiring into `time_budget`

For each (spec × composite) cell we need one `SweepConfig` entry in
`time_budget/main.py`. Reuse the shared `_SCENARIO_KW` pattern already in
that file:

```python
from examples.compositional_analysis.scenic_scenarios.specs import (
    spec_tollgate, spec_two_stops, spec_fast_twice, spec_slow2_accel,
)

SCEN_DIR = REPO_ROOT / "examples/compositional_analysis/scenic_scenarios"

def _kw(scenic_file: str, composite: str, monolithic: str,
        max_steps_primitive: int, max_steps_mono: int) -> dict:
    return dict(
        scenic_file=SCEN_DIR / scenic_file,
        composite_name=composite,
        monolithic_name=monolithic,
        max_budget=1800.0,
        snapshot_every=30.0,
        max_steps_primitive=max_steps_primitive,
        max_steps_mono=max_steps_mono,
        features=["x", "y", "speed"],
        center_feat_idx=[0, 1],
        delta=0.05,
        save_dir=Path("storage/budget_sweep_v3"),
    )

COMPOSITES = [
    ("composites/seq_SX.scenic",        "SX",          "MonoSX",          40,  80),
    ("composites/seq_SXS.scenic",       "SXS",         "MonoSXS",         40, 120),
    ("composites/seq_SOC.scenic",       "SOC",         "MonoSOC",         40, 120),
    ("composites/seq_CSXS.scenic",      "CSXS",        "MonoCSXS",        40, 160),
    ("composites/seq_CXSXC.scenic",     "CXSXC",       "MonoCXSXC",       40, 200),
    ("composites/native_choose.scenic", "SChooseCXO",  "MonoSChooseCXO",  40,  80),
    ("composites/native_shuffle.scenic","SShuffleCXO", "MonoSShuffleCXO", 40, 160),
]

# 3 specs on MetaDrive + 1 spec on Webots → 4×7 = 28 cells total.
ASSIGNMENTS = [
    ("metadrive", "tollgate",    spec_tollgate),
    ("metadrive", "two_stops",   spec_two_stops),
    ("metadrive", "fast_twice",  spec_fast_twice),
    ("webots",    "slow2_accel", spec_slow2_accel),
]

EXPERIMENTS = [
    (f"{backend}__{spec_name}__{comp_name}",
     SweepConfig(spec=spec_factory(),
                 **_kw(f"{backend}/{file}", comp_name, mono, prim, mono_steps)))
    for (backend, spec_name, spec_factory) in ASSIGNMENTS
    for (file, comp_name, mono, prim, mono_steps) in COMPOSITES
]
```

This yields the 28 cells the paper requires. Each cell produces a
`storage/budget_sweep_v3/<spec>__<composite>/results.csv` + the standard
plots from `plots.py`. The "best results in main body, rest in appendix"
selection in the paper outline is purely a write‑up decision; nothing in the
pipeline needs to change to support it.

No `prewarm_trim` is set: with PID control the primitives are at steady state
within ~5 ticks, well inside the 40‑tick budget. If a stability check (§7.1)
shows residual transient, add `prewarm_trim={p: 5 for p in (...)}`.

---

## 7. Implementation order

Three milestones, each independently shippable.

### M1 — Primitives only (1 day)

1. Create `scenic_scenarios/metadrive/primitives.scenic` reusing Town07
   spawn‑lane discovery from `wander_scenarios.scenic`.
2. Create `scenic_scenarios/webots/primitives.scenic`; symlink
   `simulators/webots/road/simple.wbt` into `scenic_scenarios/webots/world/`.
3. Validate each primitive by hand on its own backend:
   ```bash
   scenic metadrive/primitives.scenic --simulate --scenario S --time 40
   scenic webots/primitives.scenic    --simulate --scenario S --time 40
   ```
   Confirm trace shows speed converging to target within 5–10 ticks and the
   ego stays on its starting lane.
4. Write `tests/test_scenic_primitives_stability.py` that runs each primitive
   100× on each backend and asserts `std(final_speed) < 0.5 m/s`. This is the
   stability check wander never had.

### M2 — Composites + specs (1.5 days)

1. Lift the 4 `make_spec()` bodies into `scenic_scenarios/specs.py`.
2. Write the 5 sequential `seq_*.scenic` composites + their `Mono*` siblings,
   in both `metadrive/composites/` and `webots/composites/`.
3. Write `native_choose.scenic` and `native_shuffle.scenic` + `Mono*` in both
   backend subtrees.
4. Smoke‑test each in the `test_scenic_primitives_metadrive.py` style harness
   on its assigned backend: ρ̂_composite ≈ ρ̂_monolithic within ε. Webots
   needs Webots installed and the world file present.

### M3 — Budget sweep (½ day plus compute)

1. Add the `EXPERIMENTS` block to `time_budget/main.py` (or fork
   `main_v3.py` to keep wander as the v2 record).
2. Run `python -m examples.compositional_analysis.time_budget` —
   28 × `max_budget=1800s` ≈ 14 CPU‑hours for the 21 MetaDrive cells,
   plus 7 Webots cells which run slower (Webots is heavier than MetaDrive
   headless — budget ≈ 6–8 CPU‑h for the Webots row alone). Parallelise
   across machines; the Webots row can run on a dedicated box.
3. `plots.py` produces the per‑cell budget curves; assemble the 4‑spec ×
   7‑composite figure for the appendix and pick the best ρ̂‑agreement curves
   for the main body. Tag the Webots row in plots to make the cross‑simulator
   demonstration explicit.

---

## 8. What this design *doesn't* address (open risks)

1. **`do shuffle` at scenario scope**. The parser tests it at behavior scope
   (`polychrome.scenic`). If scenario‑scope shuffle has surprises in
   `analyze_scenic_composition`, the `native_shuffle.scenic` cell may need a
   manual graph spec passed to `parse_scenic_spec`. Mitigation: verify with a
   2‑primitive shuffle smoke test before authoring all 4.
2. **Town07 overtake feasibility**. Town07's incoming intersection lanes may
   not always have a same‑direction neighbour for the `O` primitive's
   lane‑change. The fallback in the skeleton degrades `O` into a "fast on the
   start lane" primitive, which still discriminates against `S/C/X` on
   `speed`. If the paper needs a real lane change, swap the spawn lane to an
   incoming lane that has `_laneToLeft` non‑null, or pick a different
   Town07 intersection.
3. **Webots map honesty**. `simple.wbt` is not Town07. Discuss in the paper
   that the Webots cell demonstrates **backend portability** — same Scenic
   primitives + composites run on a different simulator/map — not a
   map‑controlled cross‑simulator comparison. See §2.1.
4. **Webots installation cost**. Webots itself is a heavy install (≈ 1 GB,
   GUI optional but expected). The CI side of the paper artifact may need to
   skip the Webots row unless the runner has Webots provisioned. Mitigation:
   gate the Webots row behind an `if shutil.which('webots'):` check in
   `main.py`.
5. **`MonoSShuffleCXO` ground truth**. We treat the monolithic as a sample
   over the 6 permutations of (C, X, O). If the paper wants a
   per‑permutation table, expose `order` as a `param` and run the budget
   sweep once per permutation (6 extra cells — appendix only).

---

## 9. Hand‑off checklist for Arya (fixed time budget)

- [ ] M1 primitives committed and stability test green
- [ ] M2 composites + `specs.py` committed
- [ ] M3 `EXPERIMENTS` block committed in `time_budget/main.py`
- [ ] One full sweep on a 4‑core machine (≈ 14 h) → `storage/budget_sweep_v3`
- [ ] `plots.py` rendered for all 28 cells
- [ ] Best 1–2 cells per spec selected for §"Fixed Time Budget" figure
- [ ] Remaining 24 cells deposited in appendix

## 10. Hand‑off checklist for Abhi (fixed trace budget)

- [ ] Import the same `primitives.scenic` and `composites/*.scenic`
- [ ] Reuse the four `make_spec()` factories from `specs.py`
- [ ] For each (spec, composite): generate N traces of each primitive and the
      monolithic; tabulate ρ̂_comp vs. ρ̂_mono with their Hoeffding CIs
- [ ] 28‑row table; best 8 in main body, rest in appendix

Both budgets read the same Scenic source — the only difference is whether the
outer loop terminates on tick‑count (Abhi) or wall‑clock seconds (Arya).
