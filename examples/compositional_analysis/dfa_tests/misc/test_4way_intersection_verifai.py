"""End-to-end: VerifAI-sampled per-primitive traces -> non-Markovian DFA -> check_with_dfa_scenic.

Parallel to test_4way_intersection_dfa.py but uses VerifAI's external sampler
to drive scene generation. This is what unlocks `VerifaiRange(...)` and
`VerifaiOptions(...)` in the Scenic source — the standard
`scenic.scenarioFromFile().generate()` path used by generate_graph_traces
doesn't attach an external sampler, so VerifaiRange asserts. Here, the
sampler is attached automatically by `verifai.samplers.ScenicSampler`.

Pipeline (per primitive):
    verifai_<P>.scenic
        -> verifai.samplers.ScenicSampler.fromScenario  (attaches sampler)
        -> verifai.falsifier.generic_falsifier (with ScenicServer)
        -> for each iteration: VerifAI samples VerifaiRange,
           ScenicServer runs MetaDrive, monitor captures ego trajectory
        -> traces.csv (same schema as generate_graph_traces output)

DFA + compositional analysis identical to test_4way_intersection_dfa.py.

Monolithic ground truth: same Python-side branch dispatch as the standard
test, but each chosen primitive's traces are produced by re-running VerifAI's
falsifier with n_iters set to the primitive's sampled assignment count.
"""
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotmap import DotMap

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[3] / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from verifai.samplers import ScenicSampler
from verifai.falsifier import generic_falsifier
from verifai.scenic_server import ScenicServer
from verifai.monitor import specification_monitor, automaton_specification
from verifai.compositional_analysis import ScenarioBase, CompositionalAnalysisEngine, relabel_traces


PRIMITIVES = ["GoStraight", "TurnLeft", "TurnRight"]
SCENIC_DIR = HERE / "4_way_intersection_scenic"
SAVE_DIR = HERE / "storage_4way_intersection_verifai"
N_TRACES = 2    # VerifAI falsifier iterations per primitive (compositional)
MONO_N   = 6    # total VerifAI traces for the monolithic ground truth
MAX_STEPS = 40  # cap per-simulation length (our behaviors are `while True`)

STOP_THRESHOLD  = 0.5
MAX_CONSEC_SLOW = 2
WARMUP_STEPS    = 10   # mask the cold-start ramp + post-brake ramp so the
                       # DFA only judges steady-state cruise/brake behavior.


def make_spec():
    """K=2 consecutive-slow counter (same as test_4way_intersection_dfa)."""
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


def collect_traces_via_verifai(scenic_file, n, save_dir):
    """Run VerifAI's falsifier on `scenic_file` for `n` iterations.

    A custom specification_monitor extracts `record`-ed ego state from each
    sim_result and accumulates trace rows; after all iterations finish, the
    rows are written to traces.csv (schema matches generate_graph_traces).
    The returned spec value is always 0 — we're collecting traces, not
    falsifying — so n_iters traces are produced regardless of values.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "traces.csv"

    rows_by_trace = []  # list of list-of-row-dicts

    class TraceMonitor(specification_monitor):
        def __init__(self):
            def spec(sim_result):
                # sim_result.records is a dict; each entry is a list of (t, value)
                # pairs (one per simulation step). Format matches what
                # `record ego.X as Y` produces in Scenic.
                speeds = sim_result.records.get("ego_speed", [])
                positions = sim_result.records.get("ego_position", [])
                headings = sim_result.records.get("ego_heading", [])
                trace_id = len(rows_by_trace)
                trace_rows = []
                n_steps = min(len(speeds), len(positions), len(headings))
                for step in range(n_steps):
                    _t, sp = speeds[step]
                    _t, pos = positions[step]
                    _t, hd = headings[step]
                    trace_rows.append({
                        "trace_id": trace_id,
                        "step": step,
                        "x": float(pos[0]),
                        "y": float(pos[1]),
                        "heading": float(hd),
                        "speed": float(sp),
                        "action": None,
                        "reward": 0.0,
                        "label": False,
                    })
                rows_by_trace.append(trace_rows)
                return 0  # always pass — we're collecting, not falsifying
            super().__init__(spec)

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ["PANDA3D_SHOWBASE_WINDOWTYPE"] = "offscreen"

    sampler = ScenicSampler.fromScenario(str(scenic_file))
    params = DotMap(n_iters=n, save_error_table=False, fal_thres=-1)
    falsifier = generic_falsifier(
        sampler=sampler,
        monitor=TraceMonitor(),
        falsifier_params=params,
        server_class=ScenicServer,
        server_options={"maxSteps": MAX_STEPS},
    )
    print(f"[verifai] falsifying {scenic_file.name} for {n} iterations...")
    falsifier.run_falsifier()

    all_rows = [row for trace in rows_by_trace for row in trace]
    if not all_rows:
        raise RuntimeError(
            f"no traces produced for {scenic_file.name} — "
            f"sim_result.records may be empty (check `record` directives)"
        )
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"[verifai] wrote {len(rows_by_trace)} traces -> {csv_path}")
    return str(csv_path)


def _weighted_choice(rng, weights):
    total = sum(weights)
    r = rng.random() * total
    s = 0.0
    for i, w in enumerate(weights):
        s += w
        if r < s:
            return i
    return len(weights) - 1


def generate_monolithic_traces_verifai(paths, save_dir, n, seed=0):
    """Real monolithic via Python-side branch dispatch + VerifAI sampler.

    For each of `n` output traces: sample one execution path per the
    composition probabilities, pick a primitive, and group by primitive.
    For each primitive group, run VerifAI's falsifier on its scenic file
    with n_iters set to the group's size. Concatenate per-primitive monolithic
    CSVs into a single traces.csv with renumbered trace_ids.

    Same semantics as `Main`'s `do choose` (each scene picks one branch and
    runs only that branch); each trace is a fresh VerifAI-driven MetaDrive
    simulation with VerifaiRange resampled per scene.
    """
    rng = random.Random(seed)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    assignments = {}
    for trace_id in range(n):
        path_idx = _weighted_choice(rng, [p[0] for p in paths])
        composition = paths[path_idx][1]
        if len(composition) != 1:
            raise NotImplementedError(
                "verifai monolithic supports single-step compositions only; "
                f"got {composition!r}"
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

    print(f"[mono-verifai] sampled assignments: " +
          ", ".join(f"{k}={len(v)}" for k, v in sorted(assignments.items())))

    per_primitive_csvs = {}
    for primitive, trace_ids in assignments.items():
        scenic_file = SCENIC_DIR / f"verifai_{primitive}.scenic"
        per_primitive_csvs[primitive] = collect_traces_via_verifai(
            scenic_file, len(trace_ids), save_dir / primitive,
        )

    # Concatenate into a single CSV with renumbered trace_ids preserving
    # the sampled ordering.
    primitive_grouped = {
        p: list(pd.read_csv(c).sort_values(["trace_id", "step"]).groupby("trace_id"))
        for p, c in per_primitive_csvs.items()
    }
    rows = []
    cursor = {p: 0 for p in primitive_grouped}
    for trace_id in range(n):
        primitive = next(p for p, ids in assignments.items() if trace_id in ids)
        idx = cursor[primitive]
        if idx >= len(primitive_grouped[primitive]):
            print(f"[mono-verifai] WARN: ran out of traces for {primitive}")
            continue
        _orig_tid, grp = primitive_grouped[primitive][idx]
        new_grp = grp.copy()
        new_grp["trace_id"] = trace_id
        rows.append(new_grp)
        cursor[primitive] += 1

    out_csv = save_dir / "traces.csv"
    pd.concat(rows, ignore_index=True).to_csv(out_csv, index=False)
    return str(out_csv)


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. VerifAI-driven trace generation per primitive.
    logs = {}
    for primitive in PRIMITIVES:
        scenic_file = SCENIC_DIR / f"verifai_{primitive}.scenic"
        if not scenic_file.exists():
            raise FileNotFoundError(f"missing {scenic_file}")
        logs[primitive] = collect_traces_via_verifai(
            scenic_file, N_TRACES, SAVE_DIR / primitive,
        )

    # 2. Apply DFA, compute per-primitive rho.
    spec = make_spec()
    print("\n=== Per-primitive rho (after DFA relabeling) ===")
    for name in PRIMITIVES:
        rho = relabel_traces(logs[name], spec)
        print(f"  {name:11s} rho = {rho:.4f}  ({logs[name]})")

    # 3. Compositional analysis (same composition as test_4way_intersection_dfa).
    paths = [(1.0, [{p: 1.0 / len(PRIMITIVES) for p in PRIMITIVES}])]
    engine = CompositionalAnalysisEngine(ScenarioBase(logs))
    rho_comp, eps_comp = engine.check_with_dfa_scenic(
        paths, spec,
        features=["x", "y", "speed"], center_feat_idx=[0, 1],
    )

    # 4. Monolithic ground truth: VerifAI-driven, branch dispatched in Python.
    mono_csv = generate_monolithic_traces_verifai(
        paths, SAVE_DIR / "monolithic", MONO_N,
    )
    rho_mono = relabel_traces(mono_csv, spec)
    n_mono = pd.read_csv(mono_csv)["trace_id"].nunique()
    eps_mono = hoeffding_eps(n_mono)

    print(f"\n=== Compositional vs Monolithic (VerifAI-sampled) ===")
    print(f"  Compositional : rho = {rho_comp:.4f} +/- {eps_comp:.4f}")
    print(f"  Monolithic    : rho = {rho_mono:.4f} +/- {eps_mono:.4f}  (n={n_mono})")
    print(f"  |diff|        : {abs(rho_comp - rho_mono):.4f}")
    print(f"  paths         : {paths}")


def test_4way_intersection_verifai():
    main()


if __name__ == "__main__":
    main()
