from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from verifai.monitor import automaton_specification

# A composition step is either a scenario name (sequential) or a
# weighted dict of scenario names (random choice).
# Examples:
#   "S"                    → sequential step
#   {"X": 0.6, "O": 0.4}  → random choice: X with prob 0.6, O with prob 0.4
CompositionStep = Union[str, Dict[str, float]]


def relabel_traces(csv_path, spec: automaton_specification) -> float:
    """Relabel a per-primitive trace CSV with the given DFA spec's verdicts."""
    df = pd.read_csv(csv_path).sort_values("step")
    labels = {
        tid: spec.evaluate(grp.to_dict("records")) > 0
        for tid, grp in df.groupby("trace_id")
    }
    df["label"] = df["trace_id"].map(labels)
    df.to_csv(csv_path, index=False)
    return df.groupby("trace_id")["label"].last().astype(float).mean()


@dataclass
class ScenarioStats:
    rho: float
    uncertainty: float


class ScenarioBase:
    """
    Handles loading and basic statistics of scenario trace data.
    """

    REQUIRED_COLUMNS = {"trace_id", "step", "label"}

    def __init__(self, logbase: Dict[str, str], delta: float = 0.05):
        self.logbase = logbase
        self.delta = delta
        self.data: Dict[str, pd.DataFrame] = {}

        for name, path in logbase.items():
            path_obj = Path(path)
            if not path_obj.exists():
                raise FileNotFoundError(
                    f"CSV file for scenario '{name}' not found: {path}"
                )
            df = pd.read_csv(path)
            missing = self.REQUIRED_COLUMNS - set(df.columns)
            if missing:
                raise ValueError(
                    f"CSV for scenario '{name}' missing columns: {missing}"
                )
            df["trace_id"] = df["trace_id"].astype(str)
            self.data[name] = df

        self.success_stats: Dict[str, ScenarioStats] = {}
        self._compute_success_stats()

    def _compute_success_stats(self):
        for name, df in self.data.items():
            last_steps = df.sort_values("step").groupby("trace_id").tail(1)
            labels = last_steps["label"].astype(float).to_numpy()
            rho = labels.mean() if len(labels) > 0 else 0.0
            epsilon = (
                np.sqrt(np.log(2 / self.delta) / (2 * len(labels)))
                if len(labels) > 0
                else 0.0
            )
            self.success_stats[name] = ScenarioStats(rho=rho, uncertainty=epsilon)

    def get_success_prob(self, scenario: str) -> float:
        return self.success_stats[scenario].rho

    def get_success_prob_uncertainty(self, scenario: str) -> float:
        return self.success_stats[scenario].uncertainty


class CompositionalAnalysisEngine:
    """
    Compositional verification with Gaussian KDE importance sampling and
    DFA-based non-Markovian specifications, with per-sub-DFA decomposition.
    """

    def __init__(self, scenario_base: ScenarioBase):
        self.scenario_base = scenario_base

    @staticmethod
    def _normalize_features(features: np.ndarray) -> np.ndarray:
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1.0
        return (features - mean) / std

    def check_with_dfa(
        self,
        scenario: List[CompositionStep],
        spec: automaton_specification,
        features: Optional[List[str]] = None,
        center_feat_idx: Optional[List[int]] = None,
        bw_method: Union[str, float] = 10,
    ) -> Tuple[float, float]:
        if len(scenario) == 0:
            raise ValueError("Scenario list must contain at least one step.")

        if any(isinstance(s, dict) and "__shuffle__" in s for s in scenario):
            return self._check_with_dfa_shuffle(
                scenario, spec, features, center_feat_idx, bw_method
            )

        steps = [s if isinstance(s, dict) else {s: 1.0} for s in scenario]

        is_cosafety = not spec._dfa._label(spec._dfa.start)
        if is_cosafety:
            spec = ~spec

        prev_q_dist = None
        q_dist = {q: 0.0 for q in spec._dfa.states()}
        q_dist[spec._dfa.start] = 1.0

        rho = 1.0
        eps_rho_ratios = []

        prev_step = None

        q_dists = []
        n = len(steps)  # FIXME: added for eps calculation
        per_step_delta = self.scenario_base.delta / n  # FIXME: added
        eps_rho_ratios: list[float] = []  # FIXME: added

        for step in steps:
            next_q_dist, n_eff = self.forward(  # FIXME: changed, unpack n_eff
                prev_step,
                step,
                spec,
                prev_q_dist,
                q_dist,
                features=features,
                center_feat_idx=center_feat_idx,
                bw_method=bw_method,
            )
            rho_step = sum(
                v for q, v in next_q_dist.items() if spec._dfa._label(q)
            )  # FIXME: added
            if prev_step is None:  # FIXME: added, like check()
                branch_name = next(iter(step))  # FIXME: added
                eps_rho_ratios.append(
                    self.scenario_base.success_stats[branch_name].uncertainty / rho_step
                )  # FIXME: added
            else:  # FIXME: added
                epsilon_i = np.sqrt(
                    np.log(2.0 / per_step_delta) / (2.0 * n_eff)
                )  # FIXME: added
                eps_rho_ratios.append(epsilon_i / rho_step)  # FIXME: added
            q_dists.append(next_q_dist)
            prev_q_dist = q_dist
            q_dist = next_q_dist
            prev_step = step

        rho = sum(q_dist[q] for q in q_dist if spec._dfa._label(q))
        if is_cosafety:
            rho = 1 - rho

        # return rho, 0.0  # FIXME: deleted
        uncertainty = rho * float(
            np.sqrt(sum(r**2 for r in eps_rho_ratios))
        )  # FIXME: added
        return rho, uncertainty  # FIXME: added

    def _check_with_dfa_shuffle(
        self,
        scenario: List[CompositionStep],
        spec: automaton_specification,
        features: Optional[List[str]],
        center_feat_idx: Optional[List[int]],
        bw_method: Union[str, float],
    ) -> Tuple[float, float]:
        """Average check_with_dfa over all permutations of each shuffle step."""
        from itertools import permutations as _perms

        def expand(steps):
            for i, s in enumerate(steps):
                if isinstance(s, dict) and "__shuffle__" in s:
                    branches = s["__shuffle__"]
                    result = []
                    for perm in _perms(branches):
                        expanded = steps[:i] + list(perm) + steps[i + 1 :]
                        result.extend(expand(expanded))
                    return result
            return [steps]

        all_scenarios = expand(list(scenario))
        weight = 1.0 / len(all_scenarios)
        total_rho = 0.0
        variance_sum = 0.0
        for s in all_scenarios:
            rho, eps = self.check_with_dfa(
                s, spec, features, center_feat_idx, bw_method
            )
            total_rho += weight * rho
            variance_sum += (weight * eps) ** 2
        return total_rho, float(np.sqrt(variance_sum))

    def check_with_dfa_scenic(
        self,
        paths: Union[List[Tuple[float, List[CompositionStep]]], List[CompositionStep]],
        spec: automaton_specification,
        features: Optional[List[str]] = None,
        center_feat_idx: Optional[List[int]] = None,
        bw_method: Union[str, float] = 10,
    ) -> Tuple[float, float]:
        """Compositional verification for a Scenic spec."""
        # Auto-wrap a flat composition into [(1.0, composition)]
        if paths and not (
            isinstance(paths[0], tuple)
            and len(paths[0]) == 2
            and isinstance(paths[0][0], (int, float))
        ):
            paths = [(1.0, list(paths))]
        rho = 0.0
        variance_sum = 0.0
        for path_prob, composition in paths:
            path_rho, path_eps = self.check_with_dfa(
                composition,
                spec,
                features=features,
                center_feat_idx=center_feat_idx,
                bw_method=bw_method,
            )
            rho += path_prob * path_rho
            variance_sum += (path_prob * path_eps) ** 2
        return rho, np.sqrt(variance_sum)

    def forward(
        self,
        prev_step: Dict[str, float] | None,
        step: Dict[str, float],
        spec: automaton_specification,
        prev_q_dist: Dict[object, float],
        q_dist: Dict[object, float],
        features: Optional[List[str]],
        center_feat_idx: Optional[List[int]],
        bw_method: Union[str, float],
    ) -> Tuple[float, float]:

        # Compositional step: per-q IS
        if not features:
            raise ValueError("Feature list must be provided for KDE.")

        next_q_dist = {q: 0.0 for q in spec._dfa.states()}
        n_eff = 0.0  # FIXME: added, eps calculation
        for q_init, p in q_dist.items():
            if p <= 0:
                continue
            next_q_dist_from_q_init = {q: 0.0 for q in spec._dfa.states()}
            # First step: no importance sampling
            if prev_step is None:
                for branch_name, branch_weight in step.items():
                    df = self.scenario_base.data[branch_name]
                    states = self._dfa_labels(df, spec, q_init, binary_labels=False)
                    states = list(states.values())
                    if len(states) == 0:
                        continue
                    for q in next_q_dist_from_q_init:
                        next_q_dist_from_q_init[q] += branch_weight * (
                            states.count(q) / len(states)
                        )
                # Z = sum(next_q_dist_from_q_init.values())
                # if Z > 0:
                #     next_q_dist_from_q_init = {q: next_q_dist_from_q_init[q] / Z for q in next_q_dist_from_q_init}
            else:
                s_last_features, s_last_weights = self._get_states_ending_at_q(
                    prev_step,
                    spec,
                    q_init,
                    prev_q_dist,
                    features,
                    center_feat_idx,
                )

                if s_last_features.shape[0] < 2:
                    continue

                for branch_name, branch_weight in step.items():
                    df_t = self.scenario_base.data[branch_name]
                    states = self._dfa_labels(df_t, spec, q_init, binary_labels=False)
                    trace_ids = list(states.keys())
                    states = list(states.values())
                    t_first = df_t.sort_values("step").groupby("trace_id").head(1)
                    # Order t_first by trace_ids
                    t_first = t_first.set_index("trace_id").loc[trace_ids].reset_index()

                    t_first_features = t_first[features].to_numpy()
                    if center_feat_idx:
                        for j in center_feat_idx:
                            t_first_features[:, j] -= np.mean(t_first_features[:, j])

                    n_t = t_first_features.shape[0]
                    n_s = s_last_features.shape[0]
                    n_dims = (
                        t_first_features.shape[1] if t_first_features.ndim > 1 else 1
                    )
                    if n_t < 2 or n_s <= n_dims:
                        continue

                    try:
                        kde_s = gaussian_kde(
                            s_last_features.T,
                            bw_method=bw_method,
                            weights=s_last_weights,
                        )
                        kde_t = gaussian_kde(t_first_features.T, bw_method=bw_method)
                        p_vals = kde_s(t_first_features.T)
                        q_vals = kde_t(t_first_features.T)
                        weights = np.nan_to_num(
                            p_vals / q_vals, nan=0.0, posinf=0.0, neginf=0.0
                        )
                    except (
                        np.linalg.LinAlgError,
                        ValueError,
                    ):  # FIXME: ValueError, used when sample size < 2
                        weights = np.ones(n_t, dtype=float)

                    total_w = float(np.sum(weights))
                    den = np.sum(weights**2)
                    if total_w == 0 or den == 0:
                        continue

                    if total_w > 0:
                        n_eff = total_w**2 / float(
                            np.sum(weights**2)
                        )  # FIXME: added, N_eff

                    if total_w == 0:
                        continue

                    if len(states) == 0:
                        continue
                    for q in next_q_dist_from_q_init:
                        temp = np.array([s == q for s in states])
                        next_q_dist_from_q_init[q] += (
                            branch_weight * np.sum(weights * temp) / total_w
                        )
                # Z = sum(next_q_dist_from_q_init.values())
                # if Z > 0:
                #     next_q_dist_from_q_init = {q: next_q_dist_from_q_init[q] / Z for q in next_q_dist_from_q_init}

            for q in next_q_dist:
                next_q_dist[q] += (
                    next_q_dist_from_q_init[q] * p
                )  # here p is prob of starting from q_init

        # Z = sum(next_q_dist.values())
        # if Z > 0:
        #     next_q_dist = {q: next_q_dist[q] / Z for q in next_q_dist}

        # return next_q_dist  # FIXME:
        return next_q_dist, n_eff  # FIXME: added

    def _get_states_ending_at_q(
        self,
        step: Dict[str, float],
        spec: automaton_specification,
        q,
        init_q_dist: Dict[object, float],
        features: List[str],
        center_feat_idx: Optional[List[int]],
    ) -> Dict[object, Tuple[np.ndarray, np.ndarray]]:

        states_ending_at_q: Dict[object, Dict[str, list]] = {
            "states": [],
            "weights": [],
        }

        for branch_name, branch_weight in step.items():
            df = self.scenario_base.data[branch_name]
            grouped = {
                tid: group.sort_values("step").to_dict("records")
                for tid, group in df.groupby("trace_id")
            }
            s_last_all = df.sort_values("step").groupby("trace_id").tail(1).copy()
            s_last_all["trace_id"] = s_last_all["trace_id"].astype(str)
            s_last_indexed = s_last_all.set_index("trace_id")

            for q_init, p in init_q_dist.items():
                for tid, traj in grouped.items():
                    q_final = spec.advance_on_trace(traj, start=q_init)
                    if q_final != q:
                        continue

                    last_state = s_last_indexed.loc[tid, features].to_numpy()
                    states_ending_at_q["states"].append(last_state)
                    states_ending_at_q["weights"].append(p * branch_weight)

        result: Dict[object, Tuple[np.ndarray, np.ndarray]] = {}
        states = np.array(states_ending_at_q["states"], dtype=float)
        weights = np.array(states_ending_at_q["weights"], dtype=float)
        if center_feat_idx and len(states) > 0:
            for j in center_feat_idx:
                states[:, j] -= np.average(states[:, j], weights=weights)
        weights = weights / weights.sum()

        return states, weights

    @staticmethod
    def _dfa_labels(
        df: pd.DataFrame,
        spec: automaton_specification,
        q_init=None,
        binary_labels: bool = True,
    ) -> Tuple[np.ndarray, Dict[object, float]]:
        if q_init is None:
            q_init = spec._dfa.start
        grouped = {
            tid: group.sort_values("step").to_dict("records")
            for tid, group in df.groupby("trace_id")
        }

        labels = {}

        for tid in grouped.keys():
            traj = grouped[tid]
            q_final = spec.advance_on_trace(traj, start=q_init)
            if binary_labels:
                labels[tid] = spec._dfa._label(q_final)
            else:
                labels[tid] = q_final

        return labels
