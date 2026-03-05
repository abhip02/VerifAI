from abc import ABC
import networkx as nx
import mtl
from typing import Callable, Set
from dfa import DFA

class specification_monitor(ABC):
    def __init__(self, specification):
        self.specification = specification

    def evaluate(self, traj):
        return self.specification(traj)


class mtl_specification(specification_monitor):
    def __init__(self, specification):
        mtl_specs = [mtl.parse(spec) for spec in specification]
        mtl_spec = mtl_specs[0]
        if len(mtl_specs) > 1:
            for spec in mtl_specs[1:]:
                mtl_spec = (mtl_spec & spec)
        super().__init__(mtl_spec)

    def evaluate(self, traj):
        return self.specification(traj)

class multi_objective_monitor(specification_monitor):
    def __init__(self, specification, priority_graph=None, linearize=False):
        super().__init__(specification)
        self.linearize = linearize
        if priority_graph is None:
            self.graph = nx.DiGraph()
            self.graph.add_nodes_from(range(self.num_objectives))
        else:
            self.graph = priority_graph
        if linearize: # "linearize" the prioritize graph by topologically sorting and connecting nodes
            self._linearize()
    
    def _linearize(self):
        new_graph = nx.DiGraph()
        S = set([node for node, degree in self.graph.in_degree() if degree == 0])
        nodes = []
        while len(S) > 0:
            n = random.choice(S)
            S.remove(n)
            nodes.append(n)
            neighbors = self.graph.neighbors(n)
            random.shuffle(neighbors)
            for m in neighbors:
                self.graph.remove_edge(n, m)
                if self.graph.in_degree(m) == 0:
                    S.add(m)
        new_graph.add_nodes_from(nodes)
        for i in range(1, len(nodes)):
            new_graph.add_edge(nodes[i - 1], nodes[i])
        self.graph = new_graph

## new class for Non-Markovian states

class automaton_specification(specification_monitor):
    """
    Automaton-based specification monitor (based on mvcisback/dfa)

    The key idea:
      - `labeling_function` maps a raw MDP state (e.g. metadrive obs dict) → automaton input symbol
      - We feed the labeled trace into a `DFA` object and check acceptance
      - `advance_on_trace` applies the labeling function to every step in the
        trace and returns the advanced DFA (i.e. the DFA whose start state is
        wherever the trace left us)

    DFA is constructed with:
        start     : initial automaton state (any hashable)
        inputs    : set of valid automaton input symbols (the "alphabet")
        label     : Q → bool  (True iff state is accepting)
        transition: (Q, symbol) → Q
    """

    def __init__(
        self,
        start,
        inputs: Set,
        transition: Callable,
        label: Callable,
        labeling_function: Callable,
    ):
        """
        Args:
            start              : Initial DFA state (any hashable).
            inputs             : Set of alphabet symbols the DFA accepts.
            transition         : (state, symbol) → next_state
            label              : state → bool
            labeling_function  : mdp_state → alphabet symbol  (L: S_MDP → Σ)
        """
        self._dfa = DFA(
            start=start,
            inputs=inputs,
            label=label,
            transition=transition,
        )
        self.L = labeling_function
        super().__init__(self._evaluate)

    ## external methods
    
    def advance_on_trace(self, traj) -> DFA:
        """
        Apply the labeling function to each MDP state in `traj`, then
        advance the DFA along the resulting symbol sequence.

        Returns the advanced DFA (start state = state after consuming traj).
        This is the main building block for incremental / online monitoring.
        """
        word = [self.L(mdp_state) for mdp_state in traj]
        return self._dfa.advance(word)

    def evaluate(self, traj) -> float:
        return self._evaluate(traj)


    ## internal methods
    
    def _evaluate(self, traj) -> float:
        """
        Evaluate a full trajectory.

        Returns:
            +1.0  if the labeled trace is accepted by the DFA
            -1.0  otherwise
        """
        word = [self.L(mdp_state) for mdp_state in traj]
        accepted = self._dfa.label(word)
        return 1.0 if accepted else -1.0