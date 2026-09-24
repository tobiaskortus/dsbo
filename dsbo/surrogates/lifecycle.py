from __future__ import annotations

import os
import networkx as nx
from contextlib import contextmanager
from typing import Iterable, Iterator, TYPE_CHECKING, Sequence

from dsbo.modules.utils import ResultRecorder
from dsbo.workflow.runner import candidate_subdirs
from dsbo.utils.io import Workdir, get_subdir, get_workdir
from dsbo.workflow.graph_context import WorkflowExecutionGraphContext, WorkflowGraphExecutionStateState
from dsbo.surrogates.graph_transformations import (
    GraphMode,
    SurrogateTrainingPlan,
    build_module_surrogate_eval_graph,
    build_module_surrogate_training_plan,
    build_workflow_surrogate_training_plan,
)

if TYPE_CHECKING:
    from dsbo.modules import Module
    from dsbo.parameters import Candidate
    from dsbo.surrogates import SurrogateBase
    from dsbo.surrogates.workflow_surrogate import WorkflowSurrogate


def _reset_stale_records(plan: SurrogateTrainingPlan) -> None:
    """Remove stale datasets from reused directories (e.g. during development/debugging) 
    such that they cannot leak into new surrogate training."""
    record = plan.record_path
    if record is None:
        return

    filenames = {record.filename} | {
        record.filename_for((str(node.i),))
        for node in plan.graph.nodes()
        if isinstance(node, ResultRecorder) and node.i is not None
    }

    
    base = os.path.join(get_workdir().root, get_subdir(), record.pathname)

    for batch in candidate_subdirs(len(plan.candidates)):
        record_dir = os.path.join(base, batch) if batch else base
        for filename in filenames:
            path = os.path.join(record_dir, filename)
            if os.path.exists(path):
                os.remove(path)


class SurrogateTrainingContext:
    def __init__(
        self,
        graph_context: WorkflowExecutionGraphContext,
        default_graph: nx.DiGraph,
        dependency_groups: Iterable,
        surrogate: SurrogateBase,
        candidate: Candidate,
        training_task: Module,
        workflow_surrogate: bool,
    ) -> None:
        
        self._graph_context = graph_context
        self._default_graph = default_graph
        self._dependency_groups = tuple(dependency_groups)
        self._surrogate = surrogate
        self._candidate = candidate
        self._training_task = training_task
        self._workflow_surrogate = workflow_surrogate

    @contextmanager
    def activate(self):
        candidates = self._surrogate.generate_training_candidates(
            self._candidate,
            self._training_task,
        )

        training_plans = {GraphMode.WORKFLOW_SURROGATE_TRAIN: build_workflow_surrogate_training_plan,
                          GraphMode.MODULE_SURROGATE_TRAIN: build_module_surrogate_training_plan}

        with Workdir.create_subdir("surrogate_training"):
            mode = GraphMode.WORKFLOW_SURROGATE_TRAIN if self._workflow_surrogate else \
                   GraphMode.MODULE_SURROGATE_TRAIN
                            
            plan = training_plans[mode](self._default_graph, self._dependency_groups, self._surrogate,
                                        self._training_task, candidates)

            if plan.record_path is not None:
                _reset_stale_records(plan)

            state = WorkflowGraphExecutionStateState(mode, plan.graph, plan)
            
            with self._graph_context.use(state):
                yield plan.candidates

            self._surrogate.publish_trained_model(
                plan.model_file,
                stage=self._surrogate.training_stage(self._training_task),
            )


class SurrogateLifecycle:
    def __init__(
        self,
        graph_context: WorkflowExecutionGraphContext,
        dependency_groups: Iterable
        ) -> None:
        
        self._graph_context = graph_context
        self._default_graph = graph_context.graph
        self._dependency_groups = tuple(dependency_groups)
        self._workflow_surrogate: WorkflowSurrogate | None = None

    @property
    def dependency_groups(self):
        return self._dependency_groups

    @property
    def workflow_surrogate(self) -> WorkflowSurrogate | None:
        return self._workflow_surrogate

    def set_workflow_surrogate(self, surrogate: WorkflowSurrogate) -> None:
        self._workflow_surrogate = surrogate

    @property
    def module_surrogates(self) -> dict[Module, SurrogateBase]:
        return {
            module: module.surrogate
            for module in self._default_graph.nodes
            if hasattr(module, "surrogate") and module.surrogate is not None
        }

    @property
    def uses_module_surrogates(self) -> bool:
        return bool(self.module_surrogates)

    @property
    def uses_workflow_surrogates(self) -> bool:
        return self._workflow_surrogate != None    

    def pending_training_sessions(self, candidate: Candidate) -> Iterator[SurrogateTrainingContext]:
        if self._workflow_surrogate is not None:
            targets = ((self._workflow_surrogate, True),)
        else:
            targets = ((surrogate, False) for surrogate in self.module_surrogates.values())

        for surrogate, is_workflow_surrogate in targets:
            training_task = surrogate.select_training_task(candidate)
            if training_task is not None:
                yield SurrogateTrainingContext(
                    self._graph_context,
                    self._default_graph,
                    self._dependency_groups,
                    surrogate,
                    candidate,
                    training_task,
                    is_workflow_surrogate,
                )

    @contextmanager
    def use_surrogates_if_available(self, candidates: Sequence[Candidate], use_surrogates: bool = True):
        if not use_surrogates or self._graph_context.mode != GraphMode.DEFAULT:
            #TODO: Should nested contexts raise an exception?
            yield
            return

        if self.uses_workflow_surrogates:
            state = WorkflowGraphExecutionStateState(
                GraphMode.WORKFLOW_SURROGATE_EVAL,
                nx.DiGraph({self._workflow_surrogate: []}),
            )
        elif self.uses_module_surrogates:
            if len(candidates) > 1:
                raise NotImplementedError("Batch execution for module surrogates is not supported yet.")
            
            state = WorkflowGraphExecutionStateState(
                GraphMode.MODULE_SURROGATE_EVAL,
                build_module_surrogate_eval_graph(
                    self._default_graph,
                    self._dependency_groups,
                ),
            )
        else:
            yield
            return

        with self._graph_context.use(state):
            yield
