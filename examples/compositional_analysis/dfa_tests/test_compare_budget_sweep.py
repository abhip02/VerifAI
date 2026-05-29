"""Smoke tests for examples/compositional_analysis/compare_budget_sweep.py.

These exercise the script's pure helpers and its per-checkpoint analysis on
*synthetic* traces (no MetaDrive simulation), plus a static Scenic parse of
the bundled wander scenario. The goal is fast confidence that the sweep tool
behaves for both markovian and non-markovian DFA specs and that arbitrary
scenarios plug in via the parse → primitives → analyze path.

Run just these:
    pytest examples/compositional_analysis/dfa_tests/test_compare_budget_sweep.py
"""

import csv
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; must precede the script's `import pyplot`

import pytest


# ----------------------------------------------------------------------------
# Load the script as a module (it lives outside any package).
# ----------------------------------------------------------------------------

SCRIPT = Path(__file__).resolve().parent.parent / "compare_budget_sweep.py"


@pytest.fixture(scope="module")
def bs():
    spec = importlib.util.spec_from_file_location("compare_budget_sweep", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_trace(path, n_traces, steps, speed_fn):
    """Write a minimal raw trace CSV (trace_id, step, speed)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trace_id", "step", "speed"])
        for tid in range(n_traces):
            for s in range(steps):
                w.writerow([tid, s, speed_fn(tid, s)])


# ----------------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------------


def test_hoeffding_eps_monotone_and_value(bs):
    assert bs.hoeffding_eps(10) > bs.hoeffding_eps(100) > bs.hoeffding_eps(1000)
    # eps(n) = sqrt(ln(2/delta) / (2n)); spot-check n=1.
    import numpy as np

    expected = np.sqrt(np.log(2 / bs.HOEFFDING_DELTA) / 2.0)
    assert bs.hoeffding_eps(1) == pytest.approx(expected)
    # Guards against div-by-zero: n=0 clamps to n=1.
    assert bs.hoeffding_eps(0) == bs.hoeffding_eps(1)


def test_count_trace_ids(bs, tmp_path):
    p = tmp_path / "t.csv"
    write_trace(p, n_traces=3, steps=5, speed_fn=lambda t, s: 1.0)
    assert bs._count_trace_ids(str(p)) == 3
    assert bs._count_trace_ids(str(tmp_path / "missing.csv")) == 0


def test_trim_partial_drops_high_ids(bs, tmp_path):
    p = tmp_path / "t.csv"
    write_trace(p, n_traces=4, steps=3, speed_fn=lambda t, s: 1.0)
    bs._trim_partial(str(p), keep=2)  # keep trace_id < 2
    assert bs._count_trace_ids(str(p)) == 2


def test_filter_csv_first_n_traces(bs, tmp_path):
    src = tmp_path / "src.csv"
    dst = tmp_path / "dst.csv"
    write_trace(src, n_traces=5, steps=4, speed_fn=lambda t, s: 1.0)
    bs._filter_csv_first_n_traces(str(src), str(dst), n=3)
    assert bs._count_trace_ids(str(dst)) == 3
    # src untouched
    assert bs._count_trace_ids(str(src)) == 5


def test_trim_prewarm_renumbers_steps(bs, tmp_path):
    import pandas as pd

    p = tmp_path / "t.csv"
    write_trace(p, n_traces=2, steps=10, speed_fn=lambda t, s: float(s))
    bs.trim_prewarm(str(p), n=3)
    df = pd.read_csv(p)
    # 10 - 3 = 7 rows per trace, step renumbered 0..6
    for _tid, grp in df.groupby("trace_id"):
        assert list(grp.sort_values("step")["step"]) == list(range(7))
        # first kept speed is the old step 3
        assert grp.sort_values("step")["speed"].iloc[0] == pytest.approx(3.0)


def test_load_records_roundtrip(bs, tmp_path):
    p = tmp_path / "results.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bs.FIELDS)
        w.writeheader()
        w.writerow({
            "scenic_file": "x", "method": "monolithic", "budget": "30",
            "elapsed": "30", "rho": "0.5", "eps": "0.1", "n_traces": "4",
            "n_traces_breakdown": "m=4", "graph_build_s": "0.1",
            "status": "ok", "note": "",
        })
        w.writerow({
            "scenic_file": "x", "method": "compositional", "budget": "30",
            "elapsed": "30", "rho": "None", "eps": "", "n_traces": "0",
            "n_traces_breakdown": "c=0", "graph_build_s": "0.1",
            "status": "no_traces", "note": "",
        })
    recs = bs.load_records(str(p))
    assert len(recs) == 2
    assert recs[0]["rho"] == 0.5 and recs[0]["budget"] == 30.0
    # "None"/"" coerce to None; n_traces always an int
    assert recs[1]["rho"] is None and recs[1]["eps"] is None
    assert recs[1]["n_traces"] == 0


def test_load_env_file_no_overwrite(bs, tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        '# comment\nFOO_BS_TEST="abc"\nBAR_BS_TEST=already\n\nMALFORMED\n'
    )
    monkeypatch.delenv("FOO_BS_TEST", raising=False)
    monkeypatch.setenv("BAR_BS_TEST", "preexisting")
    bs.load_env_file(str(env))
    import os

    assert os.environ["FOO_BS_TEST"] == "abc"          # quotes stripped
    assert os.environ["BAR_BS_TEST"] == "preexisting"  # not overwritten
    bs.load_env_file(str(tmp_path / "nope.env"))        # missing file: no-op


# ----------------------------------------------------------------------------
# Config / specs
# ----------------------------------------------------------------------------


def test_experiments_well_formed(bs):
    from verifai.monitor import automaton_specification

    assert bs.EXPERIMENTS, "no experiments defined"
    names = [name for name, _ in bs.EXPERIMENTS]
    assert len(names) == len(set(names)), "duplicate experiment names"
    for name, cfg in bs.EXPERIMENTS:
        assert Path(cfg["scenic_file"]).exists(), f"{name}: missing scenic_file"
        assert callable(cfg["spec"]), f"{name}: spec must be a factory callable"
        # the factory must actually build a usable spec
        assert isinstance(cfg["spec"](), automaton_specification), \
            f"{name}: spec() must return an automaton_specification"
        assert isinstance(cfg["prewarm_trim"], dict)
        assert isinstance(cfg["max_steps_overrides"], dict)
        for key in ("composite_name", "monolithic_name",
                    "max_steps_primitive", "max_steps_mono"):
            assert key in cfg, f"{name}: missing {key}"


def test_default_spec_evaluates(bs):
    from verifai.monitor import automaton_specification

    spec = bs.default_spec(max_speed=4.0, warmup_steps=0)
    assert isinstance(spec, automaton_specification)
    fast = [{"step": i, "speed": 9.0} for i in range(5)]
    slow = [{"step": i, "speed": 1.0} for i in range(5)]
    assert spec.evaluate(slow) > 0   # always under limit -> accepting
    assert spec.evaluate(fast) < 0   # exceeds limit -> rejecting


def test_load_spec_nonmarkovian(bs):
    from verifai.monitor import automaton_specification

    spec = bs.load_spec(str(bs.SPEC_AT_MOST_ONE_BRAKE))
    assert isinstance(spec, automaton_specification)


def test_load_spec_missing_file_raises(bs, tmp_path):
    with pytest.raises(FileNotFoundError):
        bs.load_spec(str(tmp_path / "no_such_spec.py"))


# ----------------------------------------------------------------------------
# Per-checkpoint analysis on synthetic traces
# ----------------------------------------------------------------------------


@pytest.fixture
def two_primitive_logs(tmp_path):
    a = tmp_path / "A.csv"
    b = tmp_path / "B.csv"
    write_trace(a, n_traces=4, steps=10, speed_fn=lambda t, s: 3.0 + 0.5 * t)
    write_trace(b, n_traces=4, steps=10, speed_fn=lambda t, s: 2.0 + 0.5 * t)
    return {"A": str(a), "B": str(b)}


def test_analyze_monolithic_ok(bs, tmp_path, two_primitive_logs):
    spec = bs.default_spec(max_speed=10.0)  # all traces accept
    r = bs.analyze_monolithic_at(
        60.0, 4, two_primitive_logs["A"], "A", spec, str(tmp_path / "tmp")
    )
    assert r["status"] == "ok"
    assert 0.0 <= r["rho"] <= 1.0
    assert r["eps"] == pytest.approx(bs.hoeffding_eps(4))
    assert r["n_traces"] == 4


def test_analyze_monolithic_no_traces(bs, tmp_path):
    spec = bs.default_spec()
    r0 = bs.analyze_monolithic_at(0.0, 0, None, "A", spec, str(tmp_path / "t"))
    assert r0["status"] == "no_traces" and r0["rho"] is None
    r1 = bs.analyze_monolithic_at(5.0, 3, None, "A", spec, str(tmp_path / "t"))
    assert r1["status"] == "no_traces"  # count>0 but no CSV path


def test_analyze_compositional_ok(bs, tmp_path, two_primitive_logs):
    spec = bs.default_spec(max_speed=10.0)
    r = bs.analyze_compositional_at(
        60.0, {"A": 4, "B": 4}, two_primitive_logs, ["A", "B"], ["A", "B"],
        spec, ["speed"], [], str(tmp_path / "tmp"),
    )
    assert r["status"] == "ok"
    assert 0.0 <= r["rho"] <= 1.0
    assert r["eps"] >= 0.0 and r["eps"] == r["eps"]  # finite, not NaN
    assert "A=4" in r["n_traces_breakdown"]


def test_analyze_compositional_insufficient(bs, tmp_path, two_primitive_logs):
    spec = bs.default_spec()
    r = bs.analyze_compositional_at(
        30.0, {"A": 1, "B": 0}, two_primitive_logs, ["A", "B"], ["A", "B"],
        spec, ["speed"], [], str(tmp_path / "tmp"),
    )
    assert r["status"] == "insufficient_data" and r["rho"] is None


def test_analyze_compositional_missing_primitive(bs, tmp_path, two_primitive_logs):
    spec = bs.default_spec()
    logs = {"A": two_primitive_logs["A"]}  # B absent from final_logs
    r = bs.analyze_compositional_at(
        60.0, {"A": 4, "B": 4}, logs, ["A", "B"], ["A", "B"],
        spec, ["speed"], [], str(tmp_path / "tmp"),
    )
    assert r["status"] == "missing_primitives"


def test_composition_length(bs):
    # flat composition
    assert bs.composition_length(["A", "B", "C"]) == 3
    assert bs.composition_length(["A"]) == 1
    assert bs.composition_length([]) == 1
    # wrapped [(prob, composition), ...] — returns the max step count
    assert bs.composition_length([(1.0, ["A", "B"])]) == 2
    assert bs.composition_length([(0.5, ["A"]), (0.5, ["A", "B", "C"])]) == 3
    # a single random-choice step is still one step
    assert bs.composition_length([{"A": 0.6, "B": 0.4}]) == 1


def test_single_step_composition_needs_one_trace(bs, tmp_path):
    """Fix #4: a single-step composition uses no KDE, so 1 trace is enough —
    it should NOT report insufficient_data (which would blank out comp while
    monolithic already has a rho at the same checkpoint)."""
    a = tmp_path / "A.csv"
    write_trace(a, n_traces=1, steps=10, speed_fn=lambda t, s: 3.0)
    spec = bs.default_spec(max_speed=10.0)
    r = bs.analyze_compositional_at(
        60.0, {"A": 1}, {"A": str(a)}, ["A"], ["A"],
        spec, ["speed"], [], str(tmp_path / "tmp"),
    )
    assert r["status"] == "ok"
    assert 0.0 <= r["rho"] <= 1.0
    # A multi-step composition with a 1-trace primitive still needs >=2.
    r2 = bs.analyze_compositional_at(
        60.0, {"A": 1, "B": 1}, {"A": str(a)}, ["A", "B"], ["A", "B"],
        spec, ["speed"], [], str(tmp_path / "tmp2"),
    )
    assert r2["status"] == "insufficient_data"


@pytest.mark.parametrize("spec_kind", ["markovian", "nonmarkovian"])
def test_both_spec_kinds_run_through_analysis(
    bs, tmp_path, two_primitive_logs, spec_kind
):
    """Core consistency check: the same analysis path produces a valid
    (rho, eps) for a markovian (speed-threshold) and a non-markovian
    (brake-episode-counting) DFA spec."""
    if spec_kind == "markovian":
        spec = bs.default_spec(max_speed=10.0)
    else:
        spec = bs.load_spec(str(bs.SPEC_AT_MOST_ONE_BRAKE))

    rc = bs.analyze_compositional_at(
        60.0, {"A": 4, "B": 4}, two_primitive_logs, ["A", "B"], ["A", "B"],
        spec, ["speed"], [], str(tmp_path / "c"),
    )
    rm = bs.analyze_monolithic_at(
        60.0, 4, two_primitive_logs["A"], "A", spec, str(tmp_path / "m"),
    )
    for r in (rc, rm):
        assert r["status"] == "ok"
        assert r["rho"] is not None and 0.0 <= r["rho"] <= 1.0
        assert r["eps"] is not None and r["eps"] >= 0.0


# ----------------------------------------------------------------------------
# Plotting / records
# ----------------------------------------------------------------------------


def make_records():
    recs = []
    for method in ("monolithic", "compositional"):
        for t, rho, eps, n in [
            (30, None, None, 0), (60, 0.7, 0.4, 4),
            (120, 0.75, 0.25, 12), (300, 0.8, 0.12, 40),
        ]:
            recs.append({
                "scenic_file": "x", "method": method, "budget": float(t),
                "elapsed": float(t), "rho": rho, "eps": eps, "n_traces": n,
                "n_full_traces": float(n) if method == "monolithic" else n / 5.0,
                "n_traces_breakdown": f"{method}={n}", "graph_build_s": 0.1,
                "status": "ok" if n else "no_traces", "note": "",
            })
    return recs


def test_render_plots_creates_files(bs, tmp_path):
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    bs.render_plots(make_records(), plots_dir)
    for _key, fname in bs.PLOT_FILES:
        assert (plots_dir / fname).exists(), f"missing {fname}"


def test_render_plots_handles_empty(bs, tmp_path):
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    bs.render_plots([], plots_dir)  # no points: should skip, not crash


# ----------------------------------------------------------------------------
# Static Scenic parse (no simulation) — plug-and-play smoke test
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Weights & Biases logging
# ----------------------------------------------------------------------------


class _FakeArtifact:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.files = []

    def add_file(self, path):
        self.files.append(path)


class _FakeWandb:
    """Records calls so the logging path can be asserted without a network."""

    def __init__(self):
        self.Image = lambda path: ("image", path)
        self.logged = {}
        self.init_kwargs = None
        self.artifacts = []
        self.finished = False
        self.login_key = None

    def login(self, key=None):
        self.login_key = key

    def init(self, **kwargs):
        self.init_kwargs = kwargs

    def log(self, payload):
        self.logged.update(payload)

    def Artifact(self, name, type):
        art = _FakeArtifact(name, type)
        self.artifacts.append(art)
        return art

    def log_artifact(self, art):
        pass

    def finish(self):
        self.finished = True


def test_log_to_wandb_mocked(bs, tmp_path, monkeypatch):
    """log_to_wandb opens a run, pushes every rendered plot + the CSV, and
    closes the run — verified against a fake wandb (no network)."""
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    bs.render_plots(make_records(), plots_dir)
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("trace_id,step,label\n0,0,True\n")

    fake = _FakeWandb()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake)
    monkeypatch.setenv("WANDB_API_KEY", "dummy-key")

    bs.log_to_wandb(
        "test-project", "smoke-run", {"foo": "bar"}, plots_dir, csv_path
    )

    assert fake.login_key == "dummy-key"
    assert fake.init_kwargs["project"] == "test-project"
    assert fake.init_kwargs["name"] == "smoke-run"
    assert fake.init_kwargs["config"] == {"foo": "bar"}
    # all five figures logged as images
    logged_keys = set(fake.logged)
    assert {k for k, _ in bs.PLOT_FILES} <= logged_keys
    # CSV pushed as an artifact, run closed
    assert fake.artifacts and fake.artifacts[0].files == [str(csv_path)]
    assert fake.finished


def test_log_to_wandb_real(bs, tmp_path):
    """Opt-in: actually push plots to W&B so they can be eyeballed.

    Skipped unless VERIFAI_WANDB_TEST=1. Reads the key from the repo .env
    (or the ambient environment). Run with:

        VERIFAI_WANDB_TEST=1 pytest \\
          examples/compositional_analysis/dfa_tests/test_compare_budget_sweep.py \\
          -k log_to_wandb_real -s
    """
    import os

    if os.environ.get("VERIFAI_WANDB_TEST") != "1":
        pytest.skip("set VERIFAI_WANDB_TEST=1 to run the real W&B logging test")

    bs.load_env_file(bs.REPO_ROOT / ".env")
    if not os.environ.get("WANDB_API_KEY"):
        pytest.skip("no WANDB_API_KEY available (checked env and repo .env)")

    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    records = make_records()
    bs.render_plots(records, plots_dir)
    csv_path = tmp_path / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bs.FIELDS)
        w.writeheader()
        w.writerows(records)

    import time

    bs.log_to_wandb(
        project="verifai-compositional-analysis",
        name=f"smoketest_budget_sweep_{int(time.time())}",
        config={"smoke_test": True, "source": "test_compare_budget_sweep.py"},
        plots_dir=plots_dir,
        csv_path=csv_path,
    )


def test_wander_scenic_parses_to_primitives(bs):
    scenic_file = dict(bs.EXPERIMENTS)["wander_at_most_one_brake"]["scenic_file"]
    graph = bs.analyze_scenic_composition(scenic_file)
    partner = bs.build_partner_format(graph)
    paths = bs.parse_scenic_spec(partner)["Main"]
    primitives = sorted(bs.get_primitives(paths))
    assert primitives == [
        "BrakeScenario", "GoStraightScenario",
        "TurnLeftScenario", "TurnRightScenario",
    ]
    assert len(paths) == 5  # five sequential `do choose` steps


# ----------------------------------------------------------------------------
# Opt-in end-to-end smoke test (~5 min): real MetaDrive simulation
# ----------------------------------------------------------------------------


def test_end_to_end_short_sweep(bs, tmp_path):
    """Run the real sweep on the wander preset at a small budget so the
    plots reflect genuine monolithic-vs-compositional behavior (the synthetic
    fixtures make both curves identical, which hides the comparison).

    Skipped unless VERIFAI_SLOW_TEST=1. Tunables (seconds):
        VERIFAI_SLOW_TEST_BUDGET   per-method wall budget (default 120)
        VERIFAI_SLOW_TEST_SNAPSHOT checkpoint cadence    (default 20)
    Set VERIFAI_WANDB_TEST=1 as well to also push the plots to W&B.

        VERIFAI_SLOW_TEST=1 pytest \\
          examples/compositional_analysis/dfa_tests/test_compare_budget_sweep.py \\
          -k end_to_end -s
    """
    import os

    if os.environ.get("VERIFAI_SLOW_TEST") != "1":
        pytest.skip("set VERIFAI_SLOW_TEST=1 to run the ~5 min end-to-end sweep")
    if importlib.util.find_spec("metadrive") is None:
        pytest.skip("metadrive not installed")

    budget = float(os.environ.get("VERIFAI_SLOW_TEST_BUDGET", "120"))
    snapshot = float(os.environ.get("VERIFAI_SLOW_TEST_SNAPSHOT", "20"))

    cfg = dict(bs.EXPERIMENTS)["wander_at_most_one_brake"]
    scenic_file = str(Path(cfg["scenic_file"]).resolve())
    spec = cfg["spec"]()

    source_text = Path(scenic_file).read_text(encoding="utf-8")
    backend_name, scenic_model = bs.resolve_backend(None, None, source_text)
    mode2d = bs.default_mode2d_for_backend(backend_name)

    graph = bs.analyze_scenic_composition(scenic_file)
    partner = bs.build_partner_format(graph)
    paths = bs.parse_scenic_spec(partner)[cfg["composite_name"]]
    primitives = sorted(bs.get_primitives(paths))

    save_dir = tmp_path / "sweep"
    save_dir.mkdir()
    csv_path = save_dir / "results.csv"

    records = bs.sweep_snapshot(
        scenic_file,
        cfg["monolithic_name"],
        paths,
        primitives,
        spec,
        budget,
        snapshot,
        str(save_dir),
        cfg["features"],
        cfg["center_feat_idx"],
        cfg["max_steps_mono"],
        cfg["max_steps_primitive"],
        cfg["max_steps_overrides"],
        str(csv_path),
        0.0,
        scenic_model,
        mode2d,
        cfg["prewarm_trim"],
    )

    # Structural assertions: both methods recorded, CSV written, plots render.
    assert csv_path.exists()
    methods = {r["method"] for r in records}
    assert methods == {bs.METHOD_MONO, bs.METHOD_COMP}

    plots_dir = save_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    bs.render_plots(records, plots_dir)
    for _key, fname in bs.PLOT_FILES:
        assert (plots_dir / fname).exists(), f"missing {fname}"

    # Visibility: report how many checkpoints actually produced (rho, eps).
    ok = {m: sum(1 for r in records
                 if r["method"] == m and r["status"] == "ok")
          for m in methods}
    print(f"\n[end_to_end] ok checkpoints: {ok}")
    print(f"[end_to_end] plots: {plots_dir}/")

    # Optionally push to W&B for visual inspection.
    if os.environ.get("VERIFAI_WANDB_TEST") == "1":
        bs.load_env_file(bs.REPO_ROOT / ".env")
        if os.environ.get("WANDB_API_KEY"):
            import time

            bs.log_to_wandb(
                project="verifai-compositional-analysis",
                name=f"e2e_smoketest_{int(time.time())}",
                config={
                    "smoke_test": "end_to_end",
                    "max_budget_s": budget,
                    "snapshot_every_s": snapshot,
                    "primitives": primitives,
                },
                plots_dir=plots_dir,
                csv_path=csv_path,
            )
