# CLAUDE.md — compositional_analysis

## The one task

**Implement the Scenic primitives + composites specified in
`SCENIC_SCENARIOS.md`, then plug them into `time_budget/main.py` so that
`python -m examples.compositional_analysis.time_budget` produces the
4‑specs × 7‑composites = 28‑cell fixed‑time‑budget results for the paper.**

**Do all of it in one session.** M1 → M2 → M3 end‑to‑end, no hand‑off
between sessions, no "I'll finish this tomorrow." The reason: every
file in the system has to agree (primitive names, composite names,
spec factory names, `SweepConfig` fields, file paths in `main_v3.py`),
and a partially‑done state is worse than no progress because it makes
the next session re‑derive context you already had. Plan the session
length accordingly — expect ~3–5 hours of continuous work plus the
final smoke run.

Don't expand scope. Don't refactor `time_budget` itself. Don't touch
wander code — leave it as the v2 record and add v3 alongside.

Read `SCENIC_SCENARIOS.md` first. It is the spec; this file is the
playbook for executing it.

---

## Definition of done

- [ ] `scenic_scenarios/metadrive/primitives.scenic` + `webots/primitives.scenic`
      exist and each of S/X/C/O passes the stability check (per‑primitive
      `std(final_speed) < 0.5 m/s` across 100 runs)
- [ ] 7 composite files + 7 `Mono*` siblings exist in each backend subtree
      (14 + 14 = 28 `.scenic` files total)
- [ ] `scenic_scenarios/specs.py` exposes `spec_tollgate`,
      `spec_two_stops`, `spec_fast_twice`, `spec_slow2_accel`
      (lifted verbatim from `dfa_tests/test_check_with_dfa_*.py`)
- [ ] `time_budget/main.py` (or a new `main_v3.py`) defines the
      `EXPERIMENTS` block from §6 of `SCENIC_SCENARIOS.md`; Webots row
      gated by `shutil.which('webots')`
- [ ] One end‑to‑end smoke run of a single short‑budget cell
      (`max_budget=120s`) completes and writes `results.csv` with both
      compositional and monolithic rows
- [ ] `tests/test_scenic_primitives_stability.py` green
- [ ] No edits to `wander_scenarios.scenic` or the existing wander
      `EXPERIMENTS` entries

If any item conflicts with the spec, **stop and surface it** — don't
silently improvise.

---

## Checkpoints (course‑correct in flight, don't barrel through)

Six explicit checkpoints during the session. Each is a stop‑think‑adjust
moment — the equivalent of a self‑graded reward signal. The pattern at
every checkpoint:

1. **State** what you just produced (files, test results, what worked,
   what surprised you).
2. **Verify** it against `SCENIC_SCENARIOS.md` (does naming match? does
   the file layout match? does the spec table match?).
3. **Call `advisor()`** at checkpoints flagged ★. They see the full
   conversation and catch silent drift you can't see from inside.
4. **Decide**: continue / re‑do the previous segment / escalate.

| # | After… | What to check | advisor? |
|---|--------|---------------|----------|
| C1 | `metadrive/primitives.scenic` is written, before authoring webots variant | Does Town07 spawn‑lane block match `wander_scenarios.scenic` exactly? Do `FollowLaneBehavior` calls have the speeds from §3 table? Did you accidentally use `SetThrottleAction(Range(…))`? | ★ |
| C2 | Stability test green on **both** backends | What's the actual `std(final_speed)` per primitive? Is the labelling‑alphabet ordering (S < 3.5 < C < 7 ≤ X, O) actually realised? If a primitive is borderline, will it confuse the DFAs in §5? | ★ |
| C3 | First composite + `Mono*` sibling written (`seq_SX`) on MetaDrive | Does the `MonoBehavior` reproduce the segment sequence faithfully (brake hold time matches `S`'s, etc.)? Run it once by hand; eyeball the trace. | — |
| C4 | All 14 composite files + 14 Mono* siblings written across both backends | Diff the metadrive and webots subtrees: same filenames? same scenario names? same behaviour bodies modulo `model` line? Is `specs.py` importable and do all 4 factories build automaton specs without error? | ★ |
| C5 | One 120‑second smoke run completes on a single cell (`tollgate__SX`) | Does `results.csv` have both `compositional` and `monolithic` rows with non‑null ρ values? Is `|ρ̂_comp − ρ̂_mono| < 0.3`? If yes → continue. If no → root‑cause the disagreement before scaling up. | ★ |
| C6 | `main_v3.py` lists all 28 `EXPERIMENTS`, Webots row gated correctly | Cross‑product math: 4 ASSIGNMENTS × 7 COMPOSITES = 28? Names match the smoke test? Webots cells skip cleanly when `shutil.which('webots')` is None? | — |

If a checkpoint fails: **don't paper over it and move on**. Either fix
the cause (re‑author the failing file), or escalate to the user and
stop. Silent drift between checkpoints is the failure mode this
playbook is designed to prevent.

The two `advisor()` calls that matter most: **C2** (the entire paper
depends on primitive stability) and **C5** (the entire 14‑hour sweep
depends on this cell working). Don't skip them.

---

## Order of operations

The README's M1 → M2 → M3 is the order. Don't skip M1; the entire point
of this rewrite is to get stable primitives before composing.

### M1: primitives only
1. `metadrive/primitives.scenic` — copy Town07 spawn‑lane discovery
   from `dfa_tests/e2e_4way_example/4_way_intersection_scenic/wander_scenarios.scenic`
   *verbatim*. Replace each leaf behavior with
   `do FollowLaneBehavior(target_speed=…)`. Use the speed table in §3.
2. `webots/primitives.scenic` — `model scenic.simulators.webots.road.model`,
   pick `startLane = network.lanes[0]`, behavior bodies byte‑identical
   to MetaDrive variant.
3. Symlink `simple.wbt` into `scenic_scenarios/webots/world/` from the
   upstream Scenic checkout at
   `/Users/aryaraeesi/Documents/UC Berkeley/eecs219c/project/Scenic/src/scenic/simulators/webots/road/simple.wbt`.
4. `tests/test_scenic_primitives_stability.py`: run each primitive 100×
   per backend, assert `std(final_speed) < 0.5`.

**Gate**: do not start M2 until the stability test passes. If a primitive
is too variable, retune `target_speed` (keep the S < 3.5 < C < 7 ≤ X, O
ordering — that's the labelling alphabet the four specs assume) or add a
short prewarm. Do not fall back to open‑loop throttle.

### M2: composites + specs
5. Lift the four `make_spec()` factories from
   `dfa_tests/test_check_with_dfa_{tollgate,two_stops,cosafety_fast_twice,cosafety_slow2_accel}.py`
   into `scenic_scenarios/specs.py`. No logic changes — same DFA,
   same `labeling_function`.
6. Author the 7 composites per backend (`seq_SX`, `seq_SXS`, `seq_SOC`,
   `seq_CSXS`, `seq_CXSXC`, `native_choose`, `native_shuffle`). Use the
   `composed_scenarios.scenic:Main()` pattern for `do choose`; the
   skeleton in §4 of the README for `do shuffle`.
7. Each composite also defines a `MonoXXX` scenario that runs the same
   behaviour chain on a single ego (no respawn). The `MonoBehavior`
   reproduces the speed targets in sequence, with a short `wait` between
   segments where the original primitive holds (e.g., after `S`'s brake).
8. Smoke‑test each composite once via a one‑shot script (pattern:
   `dfa_tests/test_check_with_dfa_metadrive_tollgate.py` — but generating
   traces via `verifai.generate_graph_traces._worker_generate_scenario`
   from the new Scenic files, not `utils.generate_traces`). Assert
   `|ρ̂_comp − ρ̂_mono| < 0.2` on a 50‑trace sample. Cheap sanity check;
   not the paper result.

### M3: wire into time_budget
9. Paste the `COMPOSITES` / `ASSIGNMENTS` / `EXPERIMENTS` block from §6
   of the README into a new `time_budget/main_v3.py` (don't edit
   `main.py` — keep wander as the v2 record).
10. Add `if not shutil.which('webots'): continue` around the Webots
    rows so MetaDrive‑only runs work.
11. Short smoke run: `max_budget=120, snapshot_every=30` on one cell
    (`tollgate__SX`). Confirm `results.csv` has both compositional and
    monolithic rows with non‑empty ρ values.
12. Stop. Do not launch the full 14 CPU‑h sweep from this session —
    that's an unattended job Arya kicks off on a dedicated machine.

---

## How to use available skills/agents

This task is large enough that delegating is worth it. Suggested mapping:

- **Explore agent** (read‑only, fast) — for any "how does X work in this
  repo" question that needs >2 greps. Examples: "how does
  `_worker_generate_scenario` resolve the scenic backend?", "what does
  `analyze_scenic_composition` return for a `do shuffle` block?"
  Don't open‑ended‑grep yourself; delegate.
- **Plan agent** — once before authoring composites, to lock in the
  `MonoBehavior` design for the 5 sequential + 2 native composites. The
  monolithic ground‑truth construction is the trickiest part; get a
  written plan before writing 14 files.
- **codex** (`/codex` skill, consult mode) — second opinion on the first
  `MonoBehavior` you write. The hand‑off between segments
  (`brake → wait → accelerate` for `MonoSX`) is the kind of thing a
  fresh reviewer catches.
- **investigate skill** — if a primitive's stability test fails. Don't
  patch‑and‑pray; root‑cause first (controller saturation? PID gains
  wrong for this map? lateral overshoot at lane boundary?).
- **review skill** — before declaring M2 done; runs a diff‑level review
  of all new `.scenic` files against the parser's expected structure.
- **TaskCreate / TaskUpdate** — yes, use them. M1/M2/M3 are 3 tasks; the
  primitive‑authoring sub‑steps in M1 are 4 more. Don't try to hold
  this in working memory.
- **advisor** — call at least once before M3 (after primitives + first
  composite are written and smoke‑tested) and once before declaring
  done. The composite‑monolithic agreement is exactly the kind of
  silently‑wrong result an advisor catches.

Skills to **avoid** for this task: `/ship`, `/qa`, `/land-and-deploy`,
`/canary`, design skills, web skills. They're not relevant.

---

## Pointers (read these before writing)

| Question | File |
|---|---|
| What does the spec look like for each backend? | `SCENIC_SCENARIOS.md` §2–§4 |
| Spawn‑lane discovery pattern (Town07) | `dfa_tests/e2e_4way_example/4_way_intersection_scenic/wander_scenarios.scenic:30–47` |
| Scenario‑level `do choose` pattern | `dfa_tests/e2e_4way_example/4_way_intersection_scenic/composed_scenarios.scenic:140–148` |
| Existing `make_spec()` bodies to lift | `dfa_tests/test_check_with_dfa_{tollgate,two_stops,cosafety_fast_twice,cosafety_slow2_accel}.py` |
| `SweepConfig` schema | `time_budget/config.py` |
| How the Scenic worker is invoked | `time_budget/sweep.py:163–177` + `verifai/generate_graph_traces.py` (`_worker_generate_scenario`) |
| How `analyze_scenic_composition` parses the graph | `verifai/scenic_composition_analysis.py` |
| What the parser expects for `choose`/`shuffle` | `verifai/scenic_parser.py:77–118` |
| Upstream Scenic source for `FollowLaneBehavior` | `/Users/aryaraeesi/Documents/UC Berkeley/eecs219c/project/Scenic/src/scenic/domains/driving/behaviors.scenic` |
| Upstream Webots road model | `/Users/aryaraeesi/Documents/UC Berkeley/eecs219c/project/Scenic/src/scenic/simulators/webots/road/model.scenic` |
| Per‑backend controller bindings | `/Users/aryaraeesi/Documents/UC Berkeley/eecs219c/project/Scenic/src/scenic/simulators/{metadrive,webots,newtonian}/` |

---

## Style + guardrails

- Two backend subtrees mirror each other: same file names, same scenario
  names, same behaviour bodies. Differences live only in `model …` and
  `param map …` lines.
- Behavior bodies use `FollowLaneBehavior(target_speed=…)` — **no raw
  `SetThrottleAction(Range(…))`**. That's what made wander unstable.
- Primitives spawn their own ego (matches `composed_scenarios.scenic`
  pattern, *not* `wander_scenarios.scenic`'s ego‑on‑Main pattern).
  Composites compose scenarios, not behaviours.
- Monolithic counterpart per composite shares one ego across all
  segments; no respawn. This is the ground truth.
- Don't add new dependencies. MetaDrive and Webots are the only sim
  installs allowed.
- Don't write docs other than this file + `SCENIC_SCENARIOS.md` updates.
  No per‑scenario READMEs.
- One branch, one commit per milestone (M1, M2, M3) inside the same
  session. Don't split across PRs — the whole point of doing this in
  one go is that the next reviewer sees the full picture at once.
  Sweep *results* (CSVs, plots) land in their own follow‑up commit
  after the unattended run finishes.

---

## When to stop and ask

- Stability test fails after one round of retuning → ask, don't keep
  patching. Could mean Town07's incoming lane needs a different
  starting offset, or `FollowLaneBehavior` doesn't bind cleanly in the
  metadrive driving model.
- `analyze_scenic_composition` returns an empty or unexpected
  `paths` list for a composite → ask. Parser support for
  `do shuffle` at scenario scope is the §8 risk in the README; if it
  bites, escalate.
- Webots install absent or fails at import → ask. Don't fake it; the
  MetaDrive‑only path (gated by `shutil.which`) is a legitimate
  partial result.
- ρ̂_comp and ρ̂_mono disagree by > 0.3 on the smoke test → ask. That's
  a methodological issue, not a tuning one.

Otherwise, just execute. The spec is in `SCENIC_SCENARIOS.md`. The
acceptance criteria are in §"Definition of done" above. Use the agents.
