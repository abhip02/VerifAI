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

@dataclass
class ScenarioStats:
    rho: float
    uncertainty: float


class ScenarioBase:
    """
    Handles loading and basic statistics of scenario trace data.
    Computes empirical success probabilities and uncertainty.
    """

    REQUIRED_COLUMNS = {"trace_id", "step", "label"}

    def __init__(self, logbase: Dict[str, str], delta: float = 0.05):
        self.logbase = logbase
        self.delta = delta
        self.data: Dict[str, pd.DataFrame] = {}

        for name, path in logbase.items():
            path_obj = Path(path)
            if not path_obj.exists():
                raise FileNotFoundError(f"CSV file for scenario '{name}' not found: {path}")
            df = pd.read_csv(path)
            missing = self.REQUIRED_COLUMNS - set(df.columns)
            if missing:
                raise ValueError(f"CSV for scenario '{name}' missing columns: {missing}")
            df["trace_id"] = df["trace_id"].astype(str)
            self.data[name] = df

        self.success_stats: Dict[str, ScenarioStats] = {}
        self._compute_success_stats()

    def _compute_success_stats(self):
        for name, df in self.data.items():
            last_steps = df.sort_values("step").groupby("trace_id").tail(1)
            labels = last_steps["label"].astype(float).to_numpy()
            rho = labels.mean() if len(labels) > 0 else 0.0
            epsilon = np.sqrt(np.log(2 / self.delta) / (2 * len(labels))) if len(labels) > 0 else 0.0
            self.success_stats[name] = ScenarioStats(rho=rho, uncertainty=epsilon)

    def get_success_prob(self, scenario: str) -> float:
        return self.success_stats[scenario].rho

    def get_success_prob_uncertainty(self, scenario: str) -> float:
        return self.success_stats[scenario].uncertainty


class CompositionalAnalysisEngine:
    """
    Computes importance-sampled success probabilities across sequential
    scenarios using Gaussian KDE and uncertainty propagation.
    """

    def __init__(self, scenario_base: ScenarioBase):
        self.scenario_base = scenario_base

    @staticmethod
    def _normalize_features(features: np.ndarray) -> np.ndarray:
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1.0
        return (features - mean) / std

    def check(
        self,
        scenario: List[str],
        features: Optional[List[str]] = None,
        center_feat_idx: Optional[List[int]] = None,
        bw_method: Union[str, int] = 10,
    ) -> Tuple[float, float]:
        if len(scenario) == 0:
            raise ValueError("Scenario list must contain at least one scenario.")

        n = len(scenario)
        if n == 1:
            result = self.scenario_base.success_stats[scenario]
            return result.rho, result.uncertainty

        first_scenario_result = self.scenario_base.success_stats[scenario[0]]

        rho = first_scenario_result.rho
        eps_rho_ratios = [first_scenario_result.uncertainty/rho]

        delta = self.scenario_base.delta
        per_step_delta = delta / n

        for i in range(len(scenario) - 1):
            s_name, t_name = scenario[i], scenario[i+1]
            df_s, df_t = self.scenario_base.data[s_name], self.scenario_base.data[t_name]

            s_last = df_s.sort_values("step").groupby("trace_id").tail(1)
            s_last = s_last[s_last["label"] == True]
            t_first = df_t.sort_values("step").groupby("trace_id").head(1)
            t_last = df_t.sort_values("step").groupby("trace_id").tail(1)

            if features:
                s_last_features = s_last[features].to_numpy()
                t_first_features = t_first[features].to_numpy()
                if s_last_features.shape[0] < 2 or t_first_features.shape[0] < 2:
                    return 0.0, 0.0
                if center_feat_idx:
                    for j in center_feat_idx:
                        s_last_features[:, j] = s_last_features[:, j] - np.mean(s_last_features[:, j])
                        t_first_features[:, j] = t_first_features[:, j] - np.mean(t_first_features[:, j])
            else:
                raise ValueError("Feature list must be provided for KDE.")

            s_last_features, t_first_features = s_last_features.T, t_first_features.T

            kde_s_last = gaussian_kde(s_last_features, bw_method=bw_method)
            kde_t_first = gaussian_kde(t_first_features, bw_method=bw_method)

            p_vals = kde_s_last(t_first_features)
            q_vals = kde_t_first(t_first_features)

            weights = np.nan_to_num(p_vals / q_vals, nan=0.0, posinf=0.0, neginf=0.0)

            labels_t_last = t_last["label"].astype(float).to_numpy()

            rho_step = np.sum(weights * labels_t_last) / np.sum(weights)
            rho *= rho_step

            N_eff = np.sum(weights)**2 / np.sum(weights**2)
            epsilon_i = np.sqrt(np.log(2 / per_step_delta) / (2 * N_eff))
            eps_rho_ratios.append(epsilon_i / rho_step)

        uncertainty = rho * np.sqrt(np.sum([eps_rho_ratios**2 for eps_rho_ratios in eps_rho_ratios]))

        return rho, uncertainty

    def check_with_dfa(
        self,
        scenario: List[CompositionStep],
        spec: automaton_specification,
        features: Optional[List[str]] = None,
        center_feat_idx: Optional[List[int]] = None,
        bw_method: Union[str, float] = 10,
    ) -> Tuple[float, float]:
        """
        Compositional verification with DFA spec, supporting both sequential
        and random composition.

        Each element of `scenario` is either:
            str              -> single scenario (sequential step)
            dict[str, float] -> weighted random choice, e.g. {"X": 0.6, "O": 0.4}

        Sequential:  rho *= rho_step_i
        Random:      rho *= sum(w_j * rho_step_j)

        DFA state distribution is tracked through each step, conditioned on
        acceptance, so rejection probability is never double-counted.
        """
        if len(scenario) == 0:
            raise ValueError("Scenario list must contain at least one step.")

        if any(isinstance(s, dict) and "__shuffle__" in s for s in scenario):
            return self._check_with_dfa_shuffle(scenario, spec, features, center_feat_idx, bw_method)

        # Normalize: wrap bare strings so every step is a dict
        steps = [
            s if isinstance(s, dict) else {s: 1.0}
            for s in scenario
        ]

        n = len(steps)
        delta = self.scenario_base.delta
        per_step_delta = delta / n

        # Forward pass: compute q_init_dist at the start of each step,
        # conditioned on acceptance through all prior steps.
        q_init_dists = []
        q_dist: Dict[object, float] = {spec._dfa.start: 1.0}
        for step in steps:
            q_init_dists.append(q_dist)
            q_dist = self._advance_q_dist_through_step(step, spec, q_dist)

        # First step
        first_rho, first_eps_ratio = self._evaluate_step(
            steps[0], spec, q_init_dists[0], per_step_delta,
            prev_step=None,
            features=features, center_feat_idx=center_feat_idx,
            bw_method=bw_method,
        )

        if first_rho == 0.0:
            return 0.0, 0.0

        rho = first_rho
        eps_rho_ratios = [first_eps_ratio]

        if n == 1:
            return rho, rho * eps_rho_ratios[0]

        # Subsequent steps
        for i in range(1, n):
            step_rho, step_eps_ratio = self._evaluate_step(
                steps[i], spec, q_init_dists[i], per_step_delta,
                prev_step=steps[i - 1],
                features=features, center_feat_idx=center_feat_idx,
                bw_method=bw_method,
            )

            rho *= step_rho
            eps_rho_ratios.append(step_eps_ratio)

        uncertainty = rho * np.sqrt(sum(e ** 2 for e in eps_rho_ratios))
        return rho, uncertainty

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
                        expanded = steps[:i] + list(perm) + steps[i + 1:]
                        result.extend(expand(expanded))
                    return result
            return [steps]

        all_scenarios = expand(list(scenario))
        weight = 1.0 / len(all_scenarios)
        total_rho = 0.0
        variance_sum = 0.0
        for s in all_scenarios:
            rho, eps = self.check_with_dfa(s, spec, features, center_feat_idx, bw_method)
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
        """
        Compositional verification for a Scenic spec.

        Accepts either:
          - List[(prob, composition)] — the legacy wrapped format
          - List[CompositionStep]      — the flat composition output from
            `scenic_to_check_input` (treated as a single path with prob=1.0)

        Calls check_with_dfa on each path and returns the weighted sum:
            rho = sum(p_i * rho_i)
            eps = sqrt(sum((p_i * eps_i)^2))   [conservative, assumes independence]
        """
        # Auto-wrap a flat composition into [(1.0, composition)] so callers
        # don't need to wrap manually.
        if paths and not (isinstance(paths[0], tuple) and len(paths[0]) == 2
                          and isinstance(paths[0][0], (int, float))):
            paths = [(1.0, list(paths))]
        rho = 0.0
        variance_sum = 0.0
        for path_prob, composition in paths:
            path_rho, path_eps = self.check_with_dfa(
                composition, spec,
                features=features, center_feat_idx=center_feat_idx,
                bw_method=bw_method,
            )
            rho += path_prob * path_rho
            variance_sum += (path_prob * path_eps) ** 2
        return rho, np.sqrt(variance_sum)

    def _evaluate_step(
        self,
        step: Dict[str, float],
        spec: automaton_specification,
        q_init_dist: Dict[object, float],
        per_step_delta: float,
        prev_step: Optional[Dict[str, float]],
        features: Optional[List[str]],
        center_feat_idx: Optional[List[int]],
        bw_method: Union[str, float],
    ) -> Tuple[float, float]:
        """
        Evaluate one composition step (sequential or random).

        For a sequential step {"S": 1.0}, this behaves exactly as the
        original check_with_dfa loop body.

        For a random step {"X": 0.6, "O": 0.4}, it computes the
        importance-weighted rho for each branch and returns the weighted
        average.

        Returns (rho_step, eps_ratio).
        """
        if prev_step is None:
            # First step: no importance sampling, just DFA labels
            rho_step = 0.0
            weighted_eps_sq = 0.0

            for branch_name, branch_weight in step.items():
                df = self.scenario_base.data[branch_name]
                labels, _ = self._dfa_labels(df, spec, q_init_dist)
                branch_rho = float(np.mean(labels)) if len(labels) > 0 else 0.0
                N = len(labels)
                branch_eps = np.sqrt(np.log(2 / per_step_delta) / (2 * N)) if N > 0 else 0.0

                rho_step += branch_weight * branch_rho
                weighted_eps_sq += (branch_weight * branch_eps) ** 2

            if rho_step == 0.0:
                return 0.0, 0.0

            return rho_step, np.sqrt(weighted_eps_sq) / rho_step

        # Compositional step with importance sampling
        if not features:
            raise ValueError("Feature list must be provided for KDE.")

        # Get accepting features and per-sample weights from the previous step
        s_last_features, s_last_weights = self._get_prev_step_accepting_features(
            prev_step, spec, q_init_dist, features, center_feat_idx,
        )

        rho_step = 0.0
        weighted_eps_sq = 0.0

        for branch_name, branch_weight in step.items():
            df_t = self.scenario_base.data[branch_name]
            t_first = df_t.sort_values("step").groupby("trace_id").head(1)
            labels_t, _ = self._dfa_labels(df_t, spec, q_init_dist)

            t_first_features = t_first[features].to_numpy()
            if center_feat_idx:
                for j in center_feat_idx:
                    t_first_features[:, j] -= np.mean(t_first_features[:, j])

            if s_last_features.shape[0] < 2 or t_first_features.shape[0] < 2:
                continue

            kde_s = gaussian_kde(s_last_features.T, bw_method=bw_method,
                                 weights=s_last_weights)
            kde_t = gaussian_kde(t_first_features.T, bw_method=bw_method)

            p_vals = kde_s(t_first_features.T)
            q_vals = kde_t(t_first_features.T)
            weights = np.nan_to_num(p_vals / q_vals, nan=0.0, posinf=0.0, neginf=0.0)

            branch_rho = np.sum(weights * labels_t) / np.sum(weights)
            N_eff = np.sum(weights) ** 2 / np.sum(weights ** 2)
            branch_eps = np.sqrt(np.log(2 / per_step_delta) / (2 * N_eff))

            rho_step += branch_weight * branch_rho
            weighted_eps_sq += (branch_weight * branch_eps) ** 2

        if rho_step == 0.0:
            return 0.0, 0.0

        return rho_step, np.sqrt(weighted_eps_sq) / rho_step

    def _get_prev_step_accepting_features(
        self,
        prev_step: Dict[str, float],
        spec: automaton_specification,
        q_init_dist: Dict[object, float],
        features: List[str],
        center_feat_idx: Optional[List[int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect last-step features from accepting traces across all branches
        of the previous step, with per-sample weights reflecting branch
        probabilities.

        If prev_step is {"S1": 0.6, "S2": 0.4} and S1 has n1 accepting
        traces while S2 has n2, each S1 sample gets weight 0.6/n1 and each
        S2 sample gets weight 0.4/n2 (then normalized to sum to 1).

        Returns (features_array, weights_array).
        """
        feat_parts = []
        weight_parts = []

        for branch_name, branch_weight in prev_step.items():
            df = self.scenario_base.data[branch_name]
            labels, _ = self._dfa_labels(df, spec, q_init_dist)

            s_last_all = df.sort_values("step").groupby("trace_id").tail(1).copy()
            s_last_all["trace_id"] = s_last_all["trace_id"].astype(str)
            s_tids = (df.sort_values("step")
                      .groupby("trace_id").tail(1)["trace_id"].astype(str).tolist())
            accepting_tids = {tid for tid, acc in zip(s_tids, labels) if acc > 0.0}
            s_last = s_last_all[s_last_all["trace_id"].isin(accepting_tids)]

            n_acc = len(s_last)
            if n_acc > 0:
                feat_parts.append(s_last[features].to_numpy())
                # Each sample from this branch gets weight branch_weight / n_acc
                weight_parts.append(np.full(n_acc, branch_weight / n_acc))

        if not feat_parts:
            return np.empty((0, len(features))), np.empty(0)

        feat_array = np.concatenate(feat_parts, axis=0)
        raw_weights = np.concatenate(weight_parts)

        # Normalize so weights sum to 1
        raw_weights /= raw_weights.sum()

        if center_feat_idx:
            for j in center_feat_idx:
                feat_array[:, j] -= np.average(feat_array[:, j], weights=raw_weights)

        return feat_array, raw_weights

    def _advance_q_dist_through_step(
        self,
        step: Dict[str, float],
        spec: automaton_specification,
        q_init_dist: Dict[object, float],
    ) -> Dict[object, float]:
        """
        Advance the DFA state distribution through one composition step.
        For a random step, the result is the weighted mixture of each
        branch's q_final_dist.
        """
        mixed_dist: Dict[object, float] = {}

        for branch_name, branch_weight in step.items():
            df = self.scenario_base.data[branch_name]
            _, branch_q_final = self._dfa_labels(df, spec, q_init_dist)

            for q, w in branch_q_final.items():
                mixed_dist[q] = mixed_dist.get(q, 0.0) + branch_weight * w

        total = sum(mixed_dist.values())
        if total > 0:
            return {q: w / total for q, w in mixed_dist.items()}
        return {spec._dfa.start: 1.0}

    @staticmethod
    def _dfa_labels(
        df: pd.DataFrame,
        spec: automaton_specification,
        q_init_dist: Dict[object, float],
    ) -> Tuple[np.ndarray, Dict[object, float]]:
        """
        For each trace in df, compute acceptance probability by marginalising
        over q_init_dist:

            label(τ) = Σ_q  q_init_dist[q] * is_accepting(advance_on_trace(τ, q))

        Also returns the updated q_init_dist for the next scenario:
        the distribution of q_final values from **accepting evaluations
        only** (conditioned on acceptance).  This ensures that the
        multiplicative rho product in check_with_dfa does not double-count
        rejection probability.

        Returns
        -------
        labels       : float array, one value per trace
        q_final_dist : {q -> weight} conditioned on acceptance
        """
        grouped = {
            str(tid): group.sort_values("step").to_dict("records")
            for tid, group in df.groupby("trace_id")
        }
        trace_ids = list(grouped.keys())

        labels = np.zeros(len(trace_ids))
        q_final_accepting: Dict[object, float] = {}

        for q_init, w in q_init_dist.items():
            if w == 0:
                continue
            for idx, tid in enumerate(trace_ids):
                traj = grouped[tid]
                q_final = spec.advance_on_trace(traj, start=q_init)
                is_acc = spec._dfa._label(q_final)
                labels[idx] += w * (1.0 if is_acc else 0.0)

                if is_acc:
                    q_final_accepting[q_final] = (
                        q_final_accepting.get(q_final, 0.0) + w
                    )

        total_acc = sum(q_final_accepting.values())
        if total_acc > 0:
            q_final_dist = {q: c / total_acc
                            for q, c in q_final_accepting.items()}
        else:
            q_final_dist = {spec._dfa.start: 1.0}

        return labels, q_final_dist

    def falsify(
        self,
        scenario: Union[str, Sequence[str]],
        features: Optional[List[str]] = None,
        center_feat_idx: Optional[List[int]] = None,
        align_feat_idx: Optional[List[int]] = None,
        bw_method: Union[str, int] = 10,
    ) -> Tuple[Optional[pd.DataFrame], float]:
        if len(scenario) == 0:
            raise ValueError("Scenario list must contain at least one scenario.")

        cex = None
        n = len(scenario)

        if n == 1:
            t_name = scenario[0]
            df_t = self.scenario_base.data[t_name]

            t_traces = df_t.sort_values("step").groupby("trace_id")
            t_first = t_traces.head(1).sort_values("trace_id")
            t_last = t_traces.tail(1).sort_values("trace_id")

            fail_idx = (t_last["label"] == False).to_numpy()
            t_first = t_first[fail_idx].sort_values("trace_id")
            t_last = t_last[fail_idx].sort_values("trace_id")

            if t_first.empty or t_last.empty:
                return None

            t_last_features = t_last[features].to_numpy()
            if t_last_features.shape[0] < 1:
                return None
            elif t_last_features.shape[0] <= t_last_features.shape[1]:
                random_idx = np.random.randint(t_last_features.shape[0])
                t_trace_id = t_last.iloc[random_idx]["trace_id"]
                t_trace = t_traces.get_group(t_trace_id)
                return t_trace

            kde_t_last = gaussian_kde(t_last_features.T, bw_method=bw_method)
            t_last_prob = kde_t_last(t_last_features.T)
            t_idx = np.argmax(t_last_prob)
            t_trace_id = t_first.iloc[t_idx]["trace_id"]
            t_trace = t_traces.get_group(t_trace_id)
            return t_trace

        for i in reversed(range(n - 1)):
            s_name, t_name = scenario[i], scenario[i+1]
            df_s, df_t = self.scenario_base.data[s_name], self.scenario_base.data[t_name]

            s_traces = df_s.sort_values("step").groupby("trace_id")
            s_last = s_traces.tail(1).sort_values("trace_id")
            s_last = s_last[s_last["label"] == True].sort_values("trace_id")

            if cex is None:
                t_traces = df_t.sort_values("step").groupby("trace_id")
                t_first = t_traces.head(1).sort_values("trace_id")
                t_last = t_traces.tail(1).sort_values("trace_id")

                fail_idx = (t_last["label"] == False).to_numpy()
                t_first = t_first[fail_idx].sort_values("trace_id")
                t_last = t_last[fail_idx].sort_values("trace_id")

                if t_first.empty or t_last.empty:
                    continue

            if features:
                s_last_features = s_last[features].to_numpy()
                if s_last_features.shape[0] < 2:
                    continue
                if cex is None:
                    t_first_features = t_first[features].to_numpy()
                    if t_first_features.shape[0] < 2:
                        continue
                if center_feat_idx:
                    for j in center_feat_idx:
                        s_last_features[:, j] = s_last_features[:, j] - np.mean(s_last_features[:, j])
                        if cex is None:
                            t_first_features[:, j] = t_first_features[:, j] - np.mean(t_first_features[:, j])
            else:
                raise ValueError("Feature list must be provided for KDE.")

            if cex is None:
                if t_first_features.shape[0] <= t_first_features.shape[1]:
                    random_idx = np.random.randint(t_first_features.shape[0])
                    t_trace_id = t_first.iloc[random_idx]["trace_id"]
                    t_trace = t_traces.get_group(t_trace_id)

                    diffs = s_last_features - t_first_features[random_idx].reshape(1, -1)
                    dists = np.linalg.norm(diffs, axis=1)

                    s_idx = int(np.argmin(dists))
                    s_trace_id = s_last.iloc[s_idx]["trace_id"]
                    s_trace = s_traces.get_group(s_trace_id)

                else:
                    kde_s_last = gaussian_kde(s_last_features.T, bw_method=bw_method)
                    kde_t_first = gaussian_kde(t_first_features.T, bw_method=bw_method)

                    s_last_prob = kde_t_first(s_last_features.T)
                    t_first_prob = kde_s_last(t_first_features.T)

                    s_idx = np.argmax(s_last_prob)
                    t_idx = np.argmax(t_first_prob)

                    s_trace_id = s_last.iloc[s_idx]["trace_id"]
                    t_trace_id = t_first.iloc[t_idx]["trace_id"]

                    s_trace = s_traces.get_group(s_trace_id)
                    t_trace = t_traces.get_group(t_trace_id)

                if align_feat_idx:
                    for idx in align_feat_idx:
                        s_feat = s_trace[features[idx]]
                        t_feat = t_trace[features[idx]]
                        offset = s_feat.iloc[-1] - t_feat.iloc[0]
                        t_trace.loc[:, features[idx]] = t_feat + offset

                cex = t_trace

            else:
                if align_feat_idx:
                    compare_idx = align_feat_idx
                else:
                    compare_idx = list(range(len(features)))

                s_feat_mat = s_last_features[:, compare_idx]
                cex_first = cex[features].iloc[0].to_numpy()[compare_idx]

                diffs = s_feat_mat - cex_first.reshape(1, -1)
                dists = np.linalg.norm(diffs, axis=1)

                s_idx = int(np.argmin(dists))
                s_trace_id = s_last.iloc[s_idx]["trace_id"]
                s_trace = s_traces.get_group(s_trace_id)

                if align_feat_idx:
                    for idx in align_feat_idx:
                        s_feat = s_trace[features[idx]]
                        cex_feat = cex[features[idx]]
                        offset = s_feat.iloc[-1] - cex_feat.iloc[0]
                        cex.loc[:, features[idx]] = cex_feat + offset

            cex = pd.concat([s_trace, cex])

        if cex is None:
            return None

        final_features = [feat for feat in features] + ["label"]
        return cex[final_features].reset_index(drop=True)

    # TODO: check this, first implementation; not tested/verified
    def falsify_with_dfa(
        self,
        scenario: Union[str, Sequence[str]],
        spec: automaton_specification,
        features: Optional[List[str]] = None,
        center_feat_idx: Optional[List[int]] = None,
        align_feat_idx: Optional[List[int]] = None,
        bw_method: Union[str, int] = 10,
    ) -> Optional[pd.DataFrame]:
        """
        Identical structure to falsify(), with two substitutions:

        1. s_last filter  (was: label == True)
           → keep traces whose DFA final state is accepting, under q_init_dist

        2. failing t filter  (was: label == False)
           → keep traces whose DFA final state is NOT accepting, under q_init_dist

        Because falsify() iterates in reverse, we first do a forward pass to
        compute q_init_dist at the start of each scenario, then use those
        distributions in the reverse loop.
        """
        if len(scenario) == 0:
            raise ValueError("Scenario list must contain at least one scenario.")

        n = len(scenario)

        # --- Forward pass: compute q_init_dist at the start of each scenario ---
        # q_init_dists[i] is the distribution to use when evaluating scenario i
        q_init_dists = []
        q_init_dist: Dict[object, float] = {spec._dfa.start: 1.0}
        for s_name in scenario:
            q_init_dists.append(q_init_dist)
            df_s = self.scenario_base.data[s_name]
            _, q_init_dist = self._dfa_labels(df_s, spec, q_init_dist)

        # --- Single scenario case ---
        if n == 1:
            t_name = scenario[0]
            df_t = self.scenario_base.data[t_name]
            t_traces = df_t.sort_values("step").groupby("trace_id")
            t_last = t_traces.tail(1).sort_values("trace_id")
            t_first = t_traces.head(1).sort_values("trace_id")

            # find failing traces via DFA  (replaces label == False)
            t_labels, _ = self._dfa_labels(df_t, spec, q_init_dists[0])
            t_tids = t_last["trace_id"].astype(str).tolist()
            failing_tids = {tid for tid, lab in zip(t_tids, t_labels) if lab == 0.0}
            t_last  = t_last[t_last["trace_id"].astype(str).isin(failing_tids)]
            t_first = t_first[t_first["trace_id"].astype(str).isin(failing_tids)]

            if t_first.empty or t_last.empty:
                return None

            t_last_features = t_last[features].to_numpy()
            if t_last_features.shape[0] < 1:
                return None
            elif t_last_features.shape[0] <= t_last_features.shape[1]:
                random_idx = np.random.randint(t_last_features.shape[0])
                return t_traces.get_group(t_last.iloc[random_idx]["trace_id"])

            kde_t_last = gaussian_kde(t_last_features.T, bw_method=bw_method)
            t_idx = np.argmax(kde_t_last(t_last_features.T))
            return t_traces.get_group(t_first.iloc[t_idx]["trace_id"])

        # --- Compositional case: reverse loop ---
        cex = None

        for i in reversed(range(n - 1)):
            s_name, t_name = scenario[i], scenario[i + 1]
            df_s, df_t = self.scenario_base.data[s_name], self.scenario_base.data[t_name]

            s_traces = df_s.sort_values("step").groupby("trace_id")
            s_last_all = s_traces.tail(1).sort_values("trace_id").copy()
            s_last_all["trace_id"] = s_last_all["trace_id"].astype(str)

            # s_last: accepting traces under q_init_dist for scenario i
            # (replaces label == True)
            s_labels, _ = self._dfa_labels(df_s, spec, q_init_dists[i])
            s_tids = s_last_all["trace_id"].tolist()
            accepting_tids = {tid for tid, acc in zip(s_tids, s_labels) if acc > 0.0}
            s_last = s_last_all[s_last_all["trace_id"].isin(accepting_tids)]

            if cex is None:
                t_traces = df_t.sort_values("step").groupby("trace_id")
                t_last  = t_traces.tail(1).sort_values("trace_id").copy()
                t_first = t_traces.head(1).sort_values("trace_id").copy()
                t_last["trace_id"]  = t_last["trace_id"].astype(str)
                t_first["trace_id"] = t_first["trace_id"].astype(str)

                # failing t traces under q_init_dist for scenario i+1
                # (replaces label == False)
                t_labels, _ = self._dfa_labels(df_t, spec, q_init_dists[i + 1])
                t_tids = t_last["trace_id"].tolist()
                failing_tids = {tid for tid, lab in zip(t_tids, t_labels) if lab == 0.0}
                t_last  = t_last[t_last["trace_id"].isin(failing_tids)]
                t_first = t_first[t_first["trace_id"].isin(failing_tids)]

                if t_first.empty or t_last.empty:
                    continue

            # KDE and counterexample selection — identical to falsify()
            if features:
                s_last_features = s_last[features].to_numpy()
                if s_last_features.shape[0] < 2:
                    continue
                if cex is None:
                    t_first_features = t_first[features].to_numpy()
                    if t_first_features.shape[0] < 2:
                        continue
                if center_feat_idx:
                    for j in center_feat_idx:
                        s_last_features[:, j] -= np.mean(s_last_features[:, j])
                        if cex is None:
                            t_first_features[:, j] -= np.mean(t_first_features[:, j])
            else:
                raise ValueError("Feature list must be provided for KDE.")

            if cex is None:
                if t_first_features.shape[0] <= t_first_features.shape[1]:
                    random_idx = np.random.randint(t_first_features.shape[0])
                    t_trace_id = t_first.iloc[random_idx]["trace_id"]
                    t_trace = t_traces.get_group(t_trace_id)

                    diffs = s_last_features - t_first_features[random_idx].reshape(1, -1)
                    s_idx = int(np.argmin(np.linalg.norm(diffs, axis=1)))
                    s_trace = s_traces.get_group(s_last.iloc[s_idx]["trace_id"])
                else:
                    kde_s_last  = gaussian_kde(s_last_features.T, bw_method=bw_method)
                    kde_t_first = gaussian_kde(t_first_features.T, bw_method=bw_method)

                    s_idx = np.argmax(kde_t_first(s_last_features.T))
                    t_idx = np.argmax(kde_s_last(t_first_features.T))

                    s_trace = s_traces.get_group(s_last.iloc[s_idx]["trace_id"])
                    t_trace = t_traces.get_group(t_first.iloc[t_idx]["trace_id"])

                if align_feat_idx:
                    for idx in align_feat_idx:
                        offset = s_trace[features[idx]].iloc[-1] - t_trace[features[idx]].iloc[0]
                        t_trace = t_trace.copy()
                        t_trace.loc[:, features[idx]] = t_trace[features[idx]] + offset

                cex = t_trace

            else:
                compare_idx = align_feat_idx if align_feat_idx else list(range(len(features)))
                s_feat_mat = s_last_features[:, compare_idx]
                cex_first  = cex[features].iloc[0].to_numpy()[compare_idx]

                diffs = s_feat_mat - cex_first.reshape(1, -1)
                s_idx = int(np.argmin(np.linalg.norm(diffs, axis=1)))
                s_trace = s_traces.get_group(s_last.iloc[s_idx]["trace_id"])

                if align_feat_idx:
                    for idx in align_feat_idx:
                        offset = s_trace[features[idx]].iloc[-1] - cex[features[idx]].iloc[0]
                        cex = cex.copy()
                        cex.loc[:, features[idx]] = cex[features[idx]] + offset

            cex = pd.concat([s_trace, cex])

        if cex is None:
            return None

        final_features = [feat for feat in features] + ["label"]
        return cex[final_features].reset_index(drop=True)