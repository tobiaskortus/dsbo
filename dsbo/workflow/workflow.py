from __future__ import annotations

import torch
import networkx as nx
import torch.autograd.forward_ad as fw
from typing import Any, Iterable, Sequence, TYPE_CHECKING, overload

from dsbo.config import config
from dsbo.modules import Module
from dsbo.utils.io import get_workdir
from dsbo.parameters import CandidateBatch
from dsbo.modules.dsl import DependencyDict
from dsbo.modules.fused import is_batch_fused
from dsbo.utils.cache import WorkflowExecutionCache
from dsbo.container import warm_container_dependencies
from dsbo.surrogates.lifecycle import SurrogateLifecycle
from dsbo.workflow.runner import run_batch, RunBatchAutogradFn
from dsbo.workflow.graph_context import WorkflowExecutionGraphContext

from dsbo.workflow.graph_utils import (
    task_levels,
    assert_no_self_loops,
    assert_no_duplicate_nodes,
)

if TYPE_CHECKING:
    from dsbo.worker.spec import WorkerSpec
    from dsbo.parameters import Candidate, ParameterSpace
    from dsbo.surrogates.workflow_surrogate import WorkflowSurrogate


class ResultsBatch(list):
    @overload
    def __init__(self, items: list[dict[str, Any]]) -> None: ...

    @overload
    def __init__(self, *items: dict[str, Any]) -> None: ...

    def __init__(self, *items):
        if len(items) == 0:
            items = []

        if len(items) == 1 and isinstance(items[0], list):
            super().__init__(items[0])
        else:
            super().__init__(items)

    @classmethod
    def from_tensor(
        cls,
        values: torch.Tensor,
        keys: Sequence[str],
        splits: Sequence[int],
        is_scalar: dict[str, bool],
    ) -> ResultsBatch:
        return cls([
            {
                key: chunk.squeeze(0) if is_scalar[key] else chunk
                for key, chunk in zip(keys, torch.split(value, splits))
            }
            for value in values
        ])

    @overload
    def __getitem__(self, key: int) -> torch.Tensor: ...

    @overload
    def __getitem__(self, key: str) -> list[dict]: ...

    def __getitem__(self, key: int | str):
        if isinstance(key, int):
            return super().__getitem__(key)

        values: list[Any] = [item[key] for item in self]

        if values and isinstance(values[0], torch.Tensor):
            if all(isinstance(v, torch.Tensor) for v in values):
                return torch.concat(values)

        return values

    @property
    def keys(self):
        #TODO: handle failed tasks
        if not self: return []
        return self[0].keys()

    @property
    def tensor(self):
        #Handle failed tasks
        return torch.stack([torch.tensor(list(item.values())) for item in self])


class Workflow:
    def __init__(self, dependencies: dict[Module, Iterable[Module]] | Module) -> None:
        #Convert single module into DependencyDict format
        if isinstance(dependencies, Module):
            dependencies = DependencyDict({dependencies: []})

        self.graph_context = WorkflowExecutionGraphContext(nx.DiGraph(dependencies))
        self.surrogate_execution = SurrogateLifecycle(self.graph_context, dependencies.groups)

        self._container_warmed_up: bool = False

        assert_no_self_loops(self.task_graph)
        assert_no_duplicate_nodes(self.task_graph)

    def patch(self, surrogate: WorkflowSurrogate):
        from dsbo.utils.cache import compute_workflow_hash

        surrogate.bind_artifact_name(
            type(self).__name__,
            compute_workflow_hash(self),
        )
        self.surrogate_execution.set_workflow_surrogate(surrogate)

    def run(self, design: Candidate, catch: tuple[type[Exception], ...] = (), use_surrogates: bool = True) -> dict[str, float]:
        if use_surrogates:
            for context in self.surrogate_execution.pending_training_sessions(design):
                with context.activate() as candidates:
                    # Automatically runs the relevant surrogate training and writes
                    # the trained model to the artifact storage.
                    self.run_batch(candidates, use_surrogates=use_surrogates)

        return self.run_batch([design], catch, use_surrogates)[0]

    def run_batch(self, candidates: Sequence[Candidate], catch: tuple[type[Exception], ...] = (), use_surrogates: bool = True):
        if not candidates:
            return ResultsBatch([])

        # Make sure that the provided candidates are a CandidateBatch object.‚
        candidates = CandidateBatch(candidates)

        with self.surrogate_execution.use_surrogates_if_available(candidates, use_surrogates):
            
            self._warm_container_dependencies(verbose=True)
            
            #TODO: FIX caching for batch_fused‚
            cache = self._get_cache()
            use_cache = (
                cache.config.should_cache(self.graph_context.mode)
                if cache and not any(is_batch_fused(task) for task in self.task_graph)
                else False
            )

            if fw._current_level > -1 or not torch.is_grad_enabled():
                with torch.no_grad():
                    if use_cache:
                        return cache.load_or_compute(
                            candidates,
                            lambda misses: run_batch(self, misses, catch),
                        )
    
                    results = run_batch(self, candidates, catch)
                    if use_cache:
                        cache.save_all(candidates, results)
    
                    return results

            # NOTE: RunBatchAutogradFn is excluded from the caching as it implements ‚its own caching logic.
            vals, keys, splits, is_scalar = RunBatchAutogradFn.apply(self, candidates.tensor, candidates.space, catch)
            if not keys: return ResultsBatch([{} for _ in candidates])
    
            return ResultsBatch.from_tensor(vals, keys, splits, is_scalar)


    def get_unique_workers(self) -> set[WorkerSpec]:
        surrogates = self.surrogate_execution.module_surrogates.values()
        if (ws := self.surrogate_execution.workflow_surrogate) is not None:
            surrogates = (*surrogates, ws)
        return {
            node.resources
            for node in (*self.task_graph.nodes, *surrogates)
            if node.resources is not None
        }

    def visualize(self, filename=None, format='png'):
        from dsbo.utils.visualize import visualize_workflow
        return visualize_workflow(self, filename=filename, format=format)

    def _get_cache(self) -> WorkflowExecutionCache | None:
        cache_config = config.get("cache_config")
        if cache_config is None:
            return None
        
        workdir = get_workdir()

        return WorkflowExecutionCache(
            workflow=self,
            config=cache_config,
            workdir_root=workdir.root,
        )

    def _warm_container_dependencies(self, verbose: bool = True) -> dict:
        if not self._container_warmed_up:
            warm_container_dependencies(self, verbose=verbose)
            self._container_warmed_up = True

    @property
    def task_graph(self):
        return self.graph_context.graph
    
    @property
    def levels(self):
        return task_levels(self.task_graph)

    @property
    def max_depth(self):
        return max(self.levels.values()) + 1 if self.levels else 0
    
    @property
    def num_tasks(self):
        return self.task_graph.number_of_nodes()
