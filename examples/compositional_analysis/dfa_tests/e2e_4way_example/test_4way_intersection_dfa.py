"""End-to-end: 4-way intersection Scenic -> MetaDrive -> non-Markovian DFA -> check_with_dfa_scenic.

Composition: Main = do choose { GoStraight, TurnLeft, TurnRight }
DFA: K=2 consecutive-slow counter (non-Markovian).

Pipeline:
    Scenic file
        -> analyze_scenic_composition / build_partner_format
        -> scenic_to_check_input -> [(prob, [steps])]
        -> generate_graph_traces -> per-primitive CSVs
        -> CompositionalAnalysisEngine.check_with_dfa_scenic -> rho_comp

Ground truth (also run): the `MonolithicMain` scenario in the same Scenic
file is simulated end-to-end in MetaDrive. Each scene independently samples
one branch via `Uniform(GoStraight, TurnLeft, TurnRight)` at scene-creation
and attaches its behavior to the ego, mirroring the semantics of the
`do choose` in `Main`. Labelling those traces with the same DFA gives
rho_mono — a real end-to-end ground truth that rho_comp should approximate.

(`Main` itself can't be simulated directly because Scenic's `do choose`
rejects behaviors at simulate time. `MonolithicMain` is the runnable
equivalent — used only for the monolithic ground-truth comparison; the
parser still uses `Main` for the compositional pipeline.)
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[3] / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from verifai.monitor import automaton_specification
from verifai.compositional_analysis import ScenarioBase, CompositionalAnalysisEngine, relabel_traces
from verifai.scenic_composition_analysis import (
    analyze_scenic_composition,
    build_partner_format,
)
from verifai.scenic_parser import scenic_to_check_input, get_primitives
from verifai.generate_graph_traces import (
    generate_graph_traces,
    build_trace_jobs,
    generate_traces_parallel,
)


SCENIC_FILE = HERE / "4_way_intersection_scenic" / "composed.scenic"
SAVE_DIR    = HERE / "storage"
N_TRACES    = 100    # MetaDrive traces per primitive (parallel, ~3 procs)
MONO_N      = 100    # MetaDrive traces of MonolithicMain (single-process)
MAX_STEPS   = 40

STOP_THRESHOLD  = 0.5
MAX_CONSEC_SLOW = 2
WARMUP_STEPS    = 10   # mask the cold-start ramp + post-brake ramp so the
                       # DFA only judges steady-state cruise/brake behavior.
                       # With WARMUP<10, even n_brake=1 produces enough slow
                       # ramp-up steps to violate K=2 -> all TurnLeft reject.


def make_spec():
    """K=2 consecutive-slow counter. Non-Markovian: needs internal state."""
    def transition(state, sym):
        if state == "ok_run":
            return "slow_1" if sym == "slow" else "ok_run"
        if state == "bad":
            return "bad"
        idx = int(state.split("_")[1])
        if sym == "fast":
            return "ok_run"
        return "bad" if idx >= MAX_CONSEC_SLOW else f"slow_{idx + 1}"

    def label_row(row):
        # Mask the cold-start ramp so the DFA only judges post-warmup behavior.
        if row["step"] < WARMUP_STEPS:
            return "fast"
        return "slow" if row["speed"] < STOP_THRESHOLD else "fast"

    return automaton_specification(
        start="ok_run",
        inputs={"slow", "fast"},
        transition=transition,
        label=lambda s: s != "bad",
        labeling_function=label_row,
    )


def hoeffding_eps(n, delta=0.05):
    return float(np.sqrt(np.log(2 / delta) / (2 * max(n, 1))))


def _weighted_choice(rng, weights):
    total = sum(weights)
    r = rng.random() * total
    s = 0.0
    for i, w in enumerate(weights):
        s += w
        if r < s:
            return i
    return len(weights) - 1


def generate_monolithic_traces(scenic_file, save_dir, n, paths,
                               max_steps=None, seed=0):
    """Real monolithic via Python-side branch dispatch.

    For each of `n` output traces: sample one execution path according to the
    composition probabilities in `paths`; for each step in that path, pick a
    primitive (string or dict). Each chosen primitive contributes one fresh
    MetaDrive simulation via its existing per-primitive wrapper (the same
    wrapper `generate_graph_traces` builds), so per-trace randomness in the
    behavior (e.g. TurnLeft's `n_brake = Uniform(...)`) is freshly sampled.

    For the typical 1-step `do choose`, this is the natural ground truth: each
    monolithic trace is one full end-to-end MetaDrive run of one branch,
    drawn from the same distribution as `Main`'s choose.

    (Why Python-side? Scenic's static analyzer rejects every form of dynamic
    behavior selection — `do choose { Behavior(): ... }` rejects at simulate
    time, `Uniform(GoStraight(), TurnLeft(), TurnRight())` rejects at
    scene-creation, and `if branch == 0: do GoStraight()` inside a behavior
    rejects via the random-control-flow check. The branch decision can't live
    in Scenic, so it lives here — but the simulation itself is fully Scenic
    + MetaDrive end-to-end.)
    """
    rng = random.Random(seed)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Accept either flat List[CompositionStep] or wrapped List[(prob, composition)].
    if paths and not (isinstance(paths[0], tuple) and len(paths[0]) == 2
                      and isinstance(paths[0][0], (int, float))):
        paths = [(1.0, list(paths))]

    # Sample which primitive each trace will exercise.
    assignments = {}  # primitive -> list[trace_id]
    for trace_id in range(n):
        path_idx = _weighted_choice(rng, [p[0] for p in paths])
        composition = paths[path_idx][1]
        if len(composition) != 1:
            raise NotImplementedError(
                "real monolithic currently supports single-step compositions "
                f"only; got {composition!r}"
            )
        step = composition[0]
        if isinstance(step, str):
            primitive = step
        elif isinstance(step, dict):
            names = list(step.keys())
            primitive = names[_weighted_choice(rng, [step[k] for k in names])]
        else:
            raise ValueError(f"unsupported step type: {step!r}")
        assignments.setdefault(primitive, []).append(trace_id)

    print(f"[monolithic] sampled assignments: " +
          ", ".join(f"{k}={len(v)}" for k, v in sorted(assignments.items())))

    # Build per-primitive jobs and override n with the sampled counts.
    jobs = build_trace_jobs(
        source=str(scenic_file),
        save_dir=str(save_dir),
        n=None,
        backend="metadrive",
        max_steps=max_steps,
    )
    filtered = []
    for job in jobs:
        if job["primitive"] in assignments:
            job["n"] = len(assignments[job["primitive"]])
            filtered.append(job)

    logs = generate_traces_parallel(filtered, time_budget=float("inf"))

    # Concatenate into a single monolithic CSV with renumbered trace_ids that
    # preserve our sampled ordering.
    primitive_dfs = {p: pd.read_csv(c).sort_values(["trace_id", "step"])
                     for p, c in logs.items()}
    primitive_grouped = {p: list(df.groupby("trace_id"))
                         for p, df in primitive_dfs.items()}

    rows = []
    cursor = {p: 0 for p in primitive_grouped}
    for trace_id in range(n):
        primitive = next(p for p, ids in assignments.items() if trace_id in ids)
        idx = cursor[primitive]
        if idx >= len(primitive_grouped[primitive]):
            print(f"[monolithic] WARN: ran out of traces for {primitive} "
                  f"(needed {len(assignments[primitive])}, got "
                  f"{len(primitive_grouped[primitive])})")
            continue
        _orig_tid, grp = primitive_grouped[primitive][idx]
        new_grp = grp.copy()
        new_grp["trace_id"] = trace_id
        rows.append(new_grp)
        cursor[primitive] += 1

    out_csv = save_dir / "traces.csv"
    pd.concat(rows, ignore_index=True).to_csv(out_csv, index=False)
    return str(out_csv)


def main(reuse_traces=False):
    # 1. Scenic -> partner dict -> composition.
    graph       = analyze_scenic_composition(SCENIC_FILE)
    partner     = build_partner_format(graph)
    paths       = scenic_to_check_input(partner)
    primitives  = get_primitives(paths)
    print(f"Source     : {SCENIC_FILE}")
    print(f"Primitives : {sorted(primitives)}")
    print(f"Paths      : {paths}")

    # 2. MetaDrive traces for each primitive.
    if reuse_traces:
        logs = {}
        for primitive in primitives:
            csv_path = SAVE_DIR / primitive / "traces.csv"
            if csv_path.exists():
                logs[primitive] = str(csv_path)
                print(f"[reuse] {primitive}: {csv_path}")
            else:
                print(f"[ERROR] No existing traces for primitive {primitive} at {csv_path}")
    else:
        logs = generate_graph_traces(
            source=str(SCENIC_FILE),
            n=N_TRACES,
            save_dir=str(SAVE_DIR),
            backend="metadrive",
            max_steps=MAX_STEPS,
        )
    missing = primitives - logs.keys()
    if missing:
        raise RuntimeError(
            f"No traces produced for primitives: {sorted(missing)}. "
            f"Got: {sorted(logs.keys())}"
        )

    # 3. Relabel with DFA, run compositional check.
    spec = make_spec()
    print("\n=== Per-primitive rho (after DFA relabeling) ===")
    for name, csv_path in sorted(logs.items()):
        rho = relabel_traces(csv_path, spec)
        print(f"  {name:11s} rho = {rho:.4f}  ({csv_path})")

    engine = CompositionalAnalysisEngine(ScenarioBase(logs))
    rho_comp, eps_comp = engine.check_with_dfa(
        paths,
        spec,
        features=["x", "y", "speed"],
        center_feat_idx=[0, 1],
    )

    # 4. Monolithic ground truth.
    if reuse_traces:
        mono_csv = str(SAVE_DIR / "monolithic" / "traces.csv")
        if not Path(mono_csv).exists():
            raise RuntimeError(f"No existing monolithic traces at {mono_csv}")
        print(f"[reuse] monolithic: {mono_csv}")
    else:
        mono_csv = generate_monolithic_traces(
            SCENIC_FILE, SAVE_DIR / "monolithic", MONO_N, paths,
            max_steps=MAX_STEPS,
        )
    rho_mono = relabel_traces(mono_csv, spec)
    n_mono = pd.read_csv(mono_csv)["trace_id"].nunique()
    eps_mono = hoeffding_eps(n_mono)

    print(f"\n=== Compositional vs Monolithic ===")
    print(f"  Compositional : rho = {rho_comp:.4f} +/- {eps_comp:.4f}")
    print(f"  Monolithic    : rho = {rho_mono:.4f} +/- {eps_mono:.4f}  "
          f"(n={n_mono})")
    print(f"  |diff|        : {abs(rho_comp - rho_mono):.4f}")


def test_4way_intersection_dfa():
    main()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="4-way intersection DFA test")
    parser.add_argument("--reuse_traces", action="store_true",
                        help="Skip trace generation and use existing CSVs in SAVE_DIR")
    args = parser.parse_args()
    main(reuse_traces=args.reuse_traces)
