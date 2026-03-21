from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from verifai.monitor import automaton_specification

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

    ## New "check" function that works on "automaton_specification" specs
    def check_with_dfa(
        self,
        scenario: List[str],
        spec: automaton_specification,
        features: Optional[List[str]] = None,
        center_feat_idx: Optional[List[int]] = None,
        bw_method: Union[str, int] = 10,
    ) -> Tuple[float, float]:
        """
        Identical structure to check(), with two substitutions:

        1. s_last filter  (was: label == True)
           → keep traces whose DFA final state is accepting, under q_init_dist

        2. labels_t_last  (was: t_last["label"])
           → per-trace DFA acceptance float, evaluated under q_init_dist

        q_init_dist is a {state -> weight} distribution over DFA states.
        It starts at q0 with full probability and is updated after each
        scenario to the empirical q_final distribution from that scenario's
        traces, so each subsequent scenario is evaluated from wherever the
        automaton actually left off.
        """
        if len(scenario) == 0:
            raise ValueError("Scenario list must contain at least one scenario.")

        n = len(scenario)
        delta = self.scenario_base.delta
        per_step_delta = delta / n

        # All traces start the DFA at q0
        q_init_dist: Dict[object, float] = {spec._dfa.start: 1.0}

        # --- first scenario ---
        df_first = self.scenario_base.data[scenario[0]]
        labels_first, q_init_dist = self._dfa_labels(df_first, spec, q_init_dist)

        rho = float(np.mean(labels_first)) if len(labels_first) > 0 else 0.0
        if rho == 0.0:
            return 0.0, 0.0
        N_first = len(labels_first)
        eps_first = np.sqrt(np.log(2 / per_step_delta) / (2 * N_first))
        eps_rho_ratios = [eps_first / rho]

        if n == 1:
            return rho, rho * eps_rho_ratios[0]

        # --- compositional steps ---
        for i in range(n - 1):
            s_name, t_name = scenario[i], scenario[i + 1]
            df_s, df_t = self.scenario_base.data[s_name], self.scenario_base.data[t_name]

            # Run DFA on s traces to find which are accepting (replaces label == True)
            s_labels, _ = self._dfa_labels(df_s, spec, q_init_dist)
            s_last_all = df_s.sort_values("step").groupby("trace_id").tail(1).copy()
            s_last_all["trace_id"] = s_last_all["trace_id"].astype(str)
            s_tids = (df_s.sort_values("step")
                      .groupby("trace_id").tail(1)["trace_id"].astype(str).tolist())
            accepting_tids = {tid for tid, acc in zip(s_tids, s_labels) if acc > 0.0}
            s_last = s_last_all[s_last_all["trace_id"].isin(accepting_tids)]

            t_first = df_t.sort_values("step").groupby("trace_id").head(1)

            # Run DFA on t traces to get labels_t_last (replaces t_last["label"])
            labels_t_last, q_init_dist = self._dfa_labels(df_t, spec, q_init_dist)

            # KDE and IS weights — identical to check()
            if features:
                s_last_features = s_last[features].to_numpy()
                t_first_features = t_first[features].to_numpy()
                if s_last_features.shape[0] < 2 or t_first_features.shape[0] < 2:
                    return 0.0, 0.0
                if center_feat_idx:
                    for j in center_feat_idx:
                        s_last_features[:, j] -= np.mean(s_last_features[:, j])
                        t_first_features[:, j] -= np.mean(t_first_features[:, j])
            else:
                raise ValueError("Feature list must be provided for KDE.")

            s_last_features, t_first_features = s_last_features.T, t_first_features.T

            kde_s_last = gaussian_kde(s_last_features, bw_method=bw_method)
            kde_t_first = gaussian_kde(t_first_features, bw_method=bw_method)

            p_vals = kde_s_last(t_first_features)
            q_vals = kde_t_first(t_first_features)
            weights = np.nan_to_num(p_vals / q_vals, nan=0.0, posinf=0.0, neginf=0.0)

            rho_step = np.sum(weights * labels_t_last) / np.sum(weights)
            rho *= rho_step

            N_eff = np.sum(weights) ** 2 / np.sum(weights ** 2)
            epsilon_i = np.sqrt(np.log(2 / per_step_delta) / (2 * N_eff))
            eps_rho_ratios.append(epsilon_i / rho_step)

        uncertainty = rho * np.sqrt(np.sum([e ** 2 for e in eps_rho_ratios]))
        return rho, uncertainty

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

        Also returns the updated q_init_dist for the next scenario: the
        empirical distribution of q_final values observed across all traces.

        Returns
        -------
        labels       : float array, one value per trace
        q_final_dist : {q -> weight} for the next scenario
        """
        grouped = {
            str(tid): group.sort_values("step").to_dict("records")
            for tid, group in df.groupby("trace_id")
        }
        trace_ids = list(grouped.keys())

        labels = np.zeros(len(trace_ids))
        q_final_counts: Dict[object, float] = {}

        for q_init, w in q_init_dist.items():
            if w == 0:
                continue
            for idx, tid in enumerate(trace_ids):
                traj = grouped[tid]
                q_final = spec.advance_on_trace(traj, start=q_init)
                labels[idx] += w * (1.0 if spec._dfa._label(q_final) else 0.0)
                q_final_counts[q_final] = q_final_counts.get(q_final, 0.0) + w

        total = sum(q_final_counts.values())
        q_final_dist = (
            {q: c / total for q, c in q_final_counts.items()}
            if total > 0
            else {spec._dfa.start: 1.0}
        )

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