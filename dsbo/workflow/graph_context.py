from __future__ import annotations

import networkx as nx
from typing import Generator
from dataclasses import dataclass
from contextlib import contextmanager

from dsbo.surrogates.graph_transformations import GraphMode, SurrogateTrainingPlan


_TRAINING_MODES = {
    GraphMode.WORKFLOW_SURROGATE_TRAIN,
    GraphMode.MODULE_SURROGATE_TRAIN,
}


@dataclass(frozen=True)
class WorkflowGraphExecutionStateState:
    mode: GraphMode
    graph: nx.DiGraph
    training_plan: SurrogateTrainingPlan | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.graph, nx.DiGraph):
            raise TypeError("Workflow graph selection requires a networkx.DiGraph.")
        if self.mode in _TRAINING_MODES and self.training_plan is None:
            raise ValueError(f"{self.mode.name} selection requires a surrogate training plan.")
        if self.mode not in _TRAINING_MODES and self.training_plan is not None:
            raise ValueError("A surrogate training plan is only valid in a training selection.")


class WorkflowExecutionGraphContext:
    def __init__(self, graph: nx.DiGraph) -> None:
        if not isinstance(graph, nx.DiGraph):
            raise TypeError("WorkflowGraphContext requires a networkx.DiGraph object.")
        
        self._state = WorkflowGraphExecutionStateState(GraphMode.DEFAULT, graph)

    @property
    def current(self) -> WorkflowGraphExecutionStateState:
        return self._state

    @property
    def mode(self) -> GraphMode:
        return self._state.mode

    @property
    def graph(self) -> nx.DiGraph:
        return self._state.graph

    @property
    def training_plan(self) -> SurrogateTrainingPlan | None:
        return self._state.training_plan

    @contextmanager
    def use(self, selection: WorkflowGraphExecutionStateState) -> Generator[WorkflowGraphExecutionStateState]:
        if not isinstance(selection, WorkflowGraphExecutionStateState):
            raise TypeError("WorkflowGraphContext.use() requires a valid WorkflowGraphExecutionStateState.")
        
        previous = self._state
        self._state = selection
        try:
            yield selection
        finally:
            self._state = previous
