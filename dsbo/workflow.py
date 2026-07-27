from __future__ import annotations

import os
import torch
import warnings
import logging
import networkx as nx
from tqdm import tqdm
from itertools import chain, groupby
from contextlib import nullcontext
from contextlib import contextmanager
import torch.autograd.forward_ad as fw
from typing import Any, Iterable, List, Sequence, TYPE_CHECKING, overload

from dsbo.config import config
from dsbo.modules import Module
from dsbo.modules.dsl import DependencyDict
from dsbo.storage import get_current_storage
from dsbo.utils.cache import WorkflowExecutionCache
from dsbo.parameters import Candidate, ParameterSpace
from dsbo.utils.io import Workdir, get_subdir, get_workdir
from dsbo.utils.random import get_random_engine
from dsbo.modules.fused import file_overlays, is_batch_fused, create_candidate_symlinks
from dsbo.surrogates.graph_transformations import (GraphMode,
                                                      build_module_surrogate_eval_graph,
                                                      build_module_surrogate_train_graph)
from dsbo.container import warm_container_dependencies

if TYPE_CHECKING:
    from dsbo.worker.spec import WorkerSpec
    from dsbo.parameters import ParameterSpace
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


def assert_no_self_loops(G: nx.DiGraph) -> None:
    assert nx.number_of_selfloops(G) == 0, "The specified task graph cannot be executed due to self-loops."

def assert_no_disconnected_nodes(G: nx.DiGraph):
    # No need to check for disconnected nodes if there is only one node
    if G.number_of_nodes() == 1:
        return
    
    disconnected_nodes = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) == 0]
    assert len(disconnected_nodes) == 0, f"The specified task graph has disconnected nodes: {[n.name for n in disconnected_nodes]}"

def assert_no_duplicate_nodes(G: nx.DiGraph):
    groups = [list(group) for _, group in groupby(sorted(G.nodes(), key=id), key=id)]
    duplicates = [nodes for nodes in groups if len(nodes) > 1]
    assert len(duplicates) == 0, f"The specified task graph has duplicate nodes: {[[n.name for n in nodes] for nodes in duplicates]}"

def task_levels(G: nx.DiGraph):
    levels = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        levels[node] = 0 if not preds else 1 + max(levels[p] for p in preds)
    return levels

def is_output_task(G: nx.DiGraph, task: Module) -> bool:
    return G.out_degree(task) == 0

def get_dependent_task_groups(G: nx.DiGraph):
    levels_dict = task_levels(G)
    return [
        [node for node, _ in group]
        for _, group in groupby(sorted(levels_dict.items(), key=lambda kv: kv[1]), key=lambda kv: kv[1])
    ]


def candidate_subdirs(count: int):
    return tuple(f"batch_{index}" if count > 1 else None for index in range(count))


def num_tasks_for_batch(workflow: Workflow, num_candidates: int) -> int:
    count = 0
    for group in get_dependent_task_groups(workflow.task_graph):
        for task in group:
            count += 1 if is_batch_fused(task) else num_candidates
    return count


def register_task_stream(worker, graph, exec_ids, batch_path, task_map, name_map, name):
    storage = get_current_storage()
    if storage is None:
        return
    workflow_record = storage.workflow.add_workflow(graph)
    storage.workflow.add_runs(workflow_record, exec_ids, batch_path)
    plugin = storage.workflow.create_task_stream_plugin(task_map, name_map=name_map)
    worker.client.register_plugin(plugin, name=name)


def run_batch(workflow: Workflow, candidates: Sequence[Candidate], catch: tuple[type[Exception], ...] = ()) -> List[dict]:
    with torch.no_grad():
        from dsbo.worker.dask import WorkerContext

        worker = WorkerContext.get_current_worker()
        warm_container_dependencies(workflow)
        if not candidates:
            return []

        count = len(candidates)
        task_graph = workflow.task_graph
        subdirs = candidate_subdirs(count)

        task_groups = list(get_dependent_task_groups(task_graph))
        progress = tqdm(total=num_tasks_for_batch(workflow, count), desc="Running workflow", unit="tasks")
        get_random_engine().seed_all(workflow)

        def execution_plan():
            for depth, group in enumerate(task_groups):
                for task in group:
                    for index in (None,) if is_batch_fused(task) else range(count):
                        targets = tuple(range(count)) if index is None else (index,)
                        yield depth, task, index, targets

        if worker is None:
            batch_results = [[] for _ in candidates]

            for _, task, index, targets in execution_plan():
                try:
                    result = task.run(
                        design=candidates[0 if index is None else index].values,
                        subdir="batch_reduce" if index is None else subdirs[index],
                        file_overlays=file_overlays(task_graph, task, index, subdirs),
                        is_forward_ad=fw._current_level > -1,
                        is_grad_enabled=torch.is_grad_enabled(),
                        unpack_design=True,
                    )
                    if is_batch_fused(task):
                        #Create symlinks for visual inspection. NOTE: The framework uses overlays to resolve
                        #the original file and is not dependent on this functionality.
                        create_candidate_symlinks(task, subdirs)
                    else:
                        #Batch fused modules cannot return a dictionary result therefore it is ignored here.
                        #targets is therefore (index, )
                        batch_results[targets[0]].append(dict(result))
                except catch:
                    for t in targets:
                        batch_results[t].append(None)
                progress.update()

            return [dict(chain.from_iterable(r.items() for r in results if r))
                    for results in batch_results]

        #Create exec_ids for task stream plugin
        batch_path = Workdir.get_current_dir()
        exec_ids = [os.path.join(batch_path, s) if s else batch_path for s in subdirs]

        batch_task_ids = [[] for _ in candidates]
        scheduled_tasks = []
        delayed_map, task_map, name_map = {}, {}, {}

        for depth, task, index, targets in execution_plan():
            dependencies = [delayed_map[p] for p in task_graph.predecessors(task)]
            subdir = "batch_reduce" if index is None else subdirs[index]
            overlays = file_overlays(task_graph, task, index, subdirs)

            with Workdir.create_subdir(subdir) if subdir else nullcontext():
                task_id, delayed = worker.schedule(
                    task,
                    #TODO: this is still hacky as hell. There should be some better strategy 
                    #TODO: how to handle  candidates for batch_fuse modules.
                    candidates[0 if index is None else index].values,
                    dependencies,
                    priority=depth,
                    file_overlays=overlays,
                )

            delayed_map[task] = delayed
            scheduled_tasks.append((task_id, task, delayed))
            if not is_batch_fused(task):
                for t in targets:
                    # Ignore batch_task_ids for fused modules to avoid aggregating   
                    # fused results in the return statement
                    batch_task_ids[t].append(task_id)

            eids = tuple(exec_ids[t] for t in targets)
            task_map[delayed.key] = eids[0] if len(eids) == 1 else eids
            name_map[delayed.key] = task.name

        register_task_stream(worker, task_graph, exec_ids, batch_path, task_map, name_map, "dsbo_task_stream")

        futures = worker.compute([d for *_, d in scheduled_tasks])

        #Register progress bar update on future done callbacks.
        for f in futures:
            f.add_done_callback(lambda _: progress.update())

        results_flat = worker.gather(futures, catch=catch)
        lookup = {task_id: result for task_id, result in results_flat}

        #Create symlinks for visual inspection. NOTE: The framework uses overlays to resolve
        #the original file and is not dependent on this functionality.
        for task_id, task, _ in scheduled_tasks:
            if task_id in lookup and is_batch_fused(task):
                create_candidate_symlinks(task, subdirs)

        return [dict(chain.from_iterable(lookup.get(tid, {}).items() for tid in tids))
                for tids in batch_task_ids]


def run_batch_reverse_autograd(
    workflow: Workflow,
    candidates: Sequence[Candidate],
    grad_outputs: List[dict],
    catch: tuple[type[Exception], ...] = (),
) -> List[torch.Tensor]:
    from dsbo.worker.dask import WorkerContext
    from dsbo.config import get_active_file_backend

    worker = WorkerContext.get_current_worker()

    G = workflow.task_graph
    tasks_rev = list(reversed(list(nx.topological_sort(G))))
    pbar = tqdm(total=len(candidates) * len(tasks_rev))

    if worker is None:
        gradients = []
        for candidate, candidate_grad_output in zip(candidates, grad_outputs):
            try:
                grad_candidate = candidate.detach().requires_grad_()
                with torch.enable_grad():
                    get_active_file_backend().reset_gradient_state()
                    for task in tasks_rev:
                        is_sink = len(list(G.successors(task))) == 0
                        if task._no_grad_op:
                            continue

                        task.run_backward(grad_candidate.values, candidate_grad_output if is_sink else None)
                gradients.append(
                    grad_candidate.grad.clone() if grad_candidate.grad is not None else torch.zeros_like(candidate.tensor)
                )
            except Exception as e:
                gradients.append(torch.zeros_like(candidate.tensor))
                if not isinstance(e, catch):
                    raise
        return gradients
    else:
        batch_delays = []
        batch_task_ids = []
        task_map: dict[str, str] = {}
        name_map: dict[str, str] = {}
        exec_ids = []
        batch_path = getattr(get_workdir(), "root", None)

        for i, candidate in enumerate(candidates):
            candidate_grad_output = grad_outputs[i]
            with Workdir.create_subdir(f"batch_{i}") if len(candidates) > 1 else nullcontext():
                delayed_map = {}
                task_ids = []
                wdir = getattr(get_workdir(), "root", None)
                subdir = get_subdir()
                exec_id = os.path.join(wdir, subdir) if (wdir and subdir) else wdir
                exec_ids.append(exec_id)

                for depth, task in enumerate(tasks_rev):
                    if task._no_grad_op:
                        continue

                    forward_succs = list(G.successors(task))
                    is_sink = len(forward_succs) == 0
                    dependencies = [delayed_map[s] for s in forward_succs if s in delayed_map]
                    task_grad_output = candidate_grad_output if is_sink else None

                    task_id, delayed_obj = worker.schedule_backward(
                        task, candidate, grad_output=task_grad_output,
                        dependencies=dependencies, priority=depth,
                    )
                    delayed_map[task] = delayed_obj
                    task_ids.append(task_id)
                    batch_delays.append(delayed_obj)
                    task_map[delayed_obj.key] = exec_id
                    name_map[delayed_obj.key] = task.name
                batch_task_ids.append(task_ids)

        register_task_stream(
            worker, G, exec_ids, batch_path,
            task_map, name_map, "dsbo_task_stream_bwd",
        )

        futures_computed = worker.compute(batch_delays)
        for f in futures_computed:
            f.add_done_callback(lambda _: pbar.update())
        results_flat = worker.gather(futures_computed, catch=catch)
        lookup = {tid: grad for tid, grad in results_flat if grad is not None}

        gradients = []
        for candidate, task_ids in zip(candidates, batch_task_ids):
            combined = sum(
                (lookup[tid] for tid in task_ids if tid in lookup),
                start=torch.zeros_like(candidate.tensor),
            )
            gradients.append(combined)
        return gradients


def candidates_from_tensor(tensor: torch.Tensor, space: ParameterSpace) -> List[Candidate]:
    return [Candidate(space, tensor[i]) for i in range(tensor.shape[0])]


def to_1d_tensor(v: Any, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(v, torch.Tensor): t = v.detach().to(dtype)
    else: t = torch.tensor(float(v), dtype=dtype)
    if t.dim() > 1:
        raise ValueError(
            f"Result values must be scalar or 1-D tensors, got shape {t.shape}. "
            "Flatten higher-dimensional outputs before returning them from a Module."
        )
    return t.reshape(-1)


class RunBatchAutogradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, workflow, design_tensors: torch.Tensor, space: ParameterSpace, catch: tuple[type[Exception], ...]):
        ctx.workflow = workflow
        ctx.space = space
        ctx.save_for_backward(design_tensors)

        candidates = candidates_from_tensor(design_tensors, space)

        cache = workflow._get_cache()
        use_cache = (
            cache.config.should_cache(workflow.task_graph_mode)
            if cache and not any(is_batch_fused(task) for task in workflow.task_graph)
            else False
        )

        if use_cache:
            results, misses, indices = cache.match_partial(candidates)
            if len(misses) > 0:
                batch_results = run_batch(workflow, misses, catch)
                cache.save_all(misses, batch_results)
                for i, result in zip(indices, batch_results):
                    results[i] = result
        else:
            results = run_batch(workflow, candidates, catch)

        ctx.result_keys = list(results[0].keys()) if results else []

        if not ctx.result_keys:
            return design_tensors.new_zeros(design_tensors.shape[0]), [], {}, {}

        dtype = design_tensors.dtype
        first = results[0]

        flat = [[to_1d_tensor(r[k], dtype) for k in ctx.result_keys] for r in results]

        ctx.splits = [t.shape[0] for k, t in zip(ctx.result_keys, flat[0])]
        ctx.result_is_scalar = {
            k: (not isinstance(first[k], torch.Tensor) or first[k].dim() == 0)
            for k in ctx.result_keys
        }

        vals = torch.stack([torch.cat(r) for r in flat])
        return vals, ctx.result_keys, ctx.splits, ctx.result_is_scalar


    @staticmethod
    def backward(ctx, grad_output, _, __, ___):
        (design_tensors,) = ctx.saved_tensors
        candidates = candidates_from_tensor(design_tensors, ctx.space)
        n = len(candidates)

        grad_flat = grad_output.view(n, -1)
        grad_outputs = [
            {
                k: (chunk.squeeze(0) if ctx.result_is_scalar[k] else chunk)
                for chunk, k in zip(torch.split(grad_flat[i], ctx.splits), ctx.result_keys)
            }
            for i in range(n)
        ]

        gradients = run_batch_reverse_autograd(ctx.workflow, candidates, grad_outputs, catch=(RuntimeError,))
        return None, torch.stack(gradients, dim=0), None, None


def implements_backward(module: Module) -> bool:
    return (type(module).backward is not Module.backward) or \
           (getattr(module, "_backward", None) is not None)


def assert_workflow_supports_backward_ad(workflow: Workflow) -> None:
    unsupported = [m for m in workflow.task_graph.nodes()
                   if isinstance(m, Module) and not implements_backward(m) if not m._no_grad_op]
    if unsupported:
        raise AssertionError(
            f"Modules {unsupported} do not implement backward(), "
            "which is required for reverse-mode AD."
        )


class Workflow:
    def __init__(self, dependencies: dict[Module, Iterable[Module]] | Module) -> None:

        #Convert single module into DependencyDict format
        if isinstance(dependencies, Module):
            dependencies = DependencyDict({dependencies: []})

        self._dependencies = dependencies
        self._dependency_groups = dependencies.groups

        self._mode: GraphMode | None = None
        self._surrogate: WorkflowSurrogate | None = None
        self._default_task_graph: nx.DiGraph | None = None

        assert_no_self_loops(self.task_graph)
        assert_no_duplicate_nodes(self.task_graph)

    def patch(self, surrogate: WorkflowSurrogate):
        self._surrogate = surrogate

    @contextmanager
    def mode(self, mode: GraphMode):
        old = self._mode
        self._mode = mode
        try: yield
        finally: self._mode = old

    def task_graph_for(self, mode: GraphMode) -> nx.DiGraph:
        if mode == GraphMode.MODULE_SURROGATE_EVAL:
            return build_module_surrogate_eval_graph(self)
        if mode == GraphMode.MODULE_SURROGATE_TRAIN:
            return build_module_surrogate_train_graph(self)
        if mode == GraphMode.WORKFLOW_SURROGATE:
            return nx.DiGraph({self._surrogate: []})

        if self._default_task_graph is None:
            self._default_task_graph = nx.DiGraph(self._dependencies)
        return self._default_task_graph

    @property
    def module_surrogates(self) -> dict[Module, WorkflowSurrogate]:
        with self.mode(GraphMode.DEFAULT):
            has_module_surrogate = lambda module: hasattr(module, "surrogate") and module.surrogate is not None
            return {module: module.surrogate for module in self.task_graph.nodes if has_module_surrogate(module)}

    @property
    def uses_module_surrogates(self) -> bool:
        return len(self.module_surrogates) > 0


    def run(self, design: Candidate, catch: tuple[type[Exception], ...] = ()) -> dict[str, float]:
        self._ensure_surrogates_trained(design, catch)
        return self.run_batch([design], catch)[0]

    def run_batch(self, candidates: Sequence[Candidate], catch: tuple[type[Exception], ...] = ()):
        if self._surrogate is not None and self._surrogate._is_pretrained:
            if len(candidates) > 1:
                raise NotImplementedError("Batch execution for workflow surrogates is not supported yet.")
            return ResultsBatch([self._surrogate(c) for c in candidates])
        
        if self.uses_module_surrogates:
            if len(candidates) > 1:
                raise NotImplementedError("Batch execution for module surrogates is not supported yet.")
            with self.mode(GraphMode.MODULE_SURROGATE_EVAL):
                return self._execute_batch(candidates, catch)
            
        return self._execute_batch(candidates, catch)
    
    def _ensure_surrogates_trained(self, candidate: Candidate, catch: tuple[type[Exception], ...] = ()):
        if self._surrogate is not None:
            if not self._surrogate.is_in_trust_region(candidate) or self._surrogate.surrogate is None:
                with self.mode(GraphMode.DEFAULT):
                    self._surrogate.fit(candidate, workflow_callback=self._execute_batch)

            return

        if self.uses_module_surrogates:
            with Workdir.create_subdir("module_surrogate_training"):
                for _, surrogate in self.module_surrogates.items():
                    if not surrogate.is_in_trust_region(candidate) or not surrogate.is_available:
                        with self.mode(GraphMode.MODULE_SURROGATE_TRAIN):
                            from dsbo.utils.io.sharded import ShardedHDF5Storage
                            with config.use(file_backend=ShardedHDF5Storage):
                                surrogate.fit(candidate, workflow_callback=self._execute_batch)

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

    def _execute_batch(self, candidates: Sequence[Candidate], catch: tuple[type[Exception], ...] = ()):
        if not candidates:
            return ResultsBatch([])

        cache = self._get_cache()
        use_cache = (
            cache.config.should_cache(self.task_graph_mode)
            if cache and not any(is_batch_fused(task) for task in self.task_graph)
            else False
        )

        if fw._current_level > -1 or not torch.is_grad_enabled():
            with torch.no_grad():
                if use_cache:
                    results, misses, indices = cache.match_partial(candidates)
                    if len(misses) > 0:
                        batch_results = run_batch(self, misses, catch)
                        cache.save_all(misses, batch_results)
                        for i, result in zip(indices, batch_results):
                            results[i] = result
                        
                    return results

                results = run_batch(self, candidates, catch)
                if use_cache:
                    cache.save_all(candidates, results)

                return results

        design_tensors = torch.stack([c.tensor for c in candidates])
        space = candidates[0].space
        vals, keys, splits, is_scalar = RunBatchAutogradFn.apply(self, design_tensors, space, catch)
        if not keys: return [{} for _ in candidates]

        return ResultsBatch([
            {
                k: (chunk.squeeze(0) if is_scalar[k] else chunk)
                for chunk, k in zip(torch.split(vals[i], splits), keys)
            }
            for i in range(len(candidates))
        ])


    def _iterate_dependent_task_groups(self):
        for batch in get_dependent_task_groups(self.task_graph):
            yield batch

    def get_unique_workers(self) -> set[WorkerSpec]:
        workers = set()
        for node in self.task_graph.nodes():
            if node.resources is not None:
                workers.add(node.resources)
        return workers

    def visualize(self, filename=None, format='png'):
        from dsbo.utils.visualize import visualize_workflow
        return visualize_workflow(self, filename=filename, format=format)

    def warm_containers(self, verbose: bool = True) -> dict:
        return warm_container_dependencies(self, verbose=verbose)

    @property
    def task_graph(self):
        return self.task_graph_for(self.task_graph_mode)
    
    @property
    def task_graph_mode(self) -> GraphMode:
        return self._mode or (
            GraphMode.WORKFLOW_SURROGATE
            if self._surrogate is not None and self._surrogate._is_pretrained
            else GraphMode.DEFAULT
        )

    @property
    def levels(self):
        return task_levels(self.task_graph)
    
    @property
    def max_depth(self):
        return max(self.levels.values()) + 1 if self.levels else 0
    
    @property
    def num_tasks(self):
        return self.task_graph.number_of_nodes()


def assert_no_mixed_surrogate_usage(workflow: Workflow) -> None:
    if workflow._surrogate is not None and workflow.uses_module_surrogates:
        raise ValueError("Workflow cannot use both a workflow surrogate and module surrogates simultaneously.")


def assert_leaf_surrogates(workflow: Workflow) -> None:
    for module in workflow.task_graph.nodes:
        if hasattr(module, "surrogate") and module.surrogate is not None:
            if len(list(workflow.task_graph.successors(module))) > 0:
                raise ValueError("Module surrogates must be leaf nodes in the task graph.")
    

def mergeable_components(task_graph: Workflow, attr="resources") -> List[List[Module]]:
    G = task_graph.task_graph
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    contractible_edges = [
        (u, v)
        for u, v in G.edges()
        if (getattr(u, attr) == getattr(v, attr)
            and out_deg[u] == 1
            and in_deg[v] == 1)
    ]

    H = nx.DiGraph(contractible_edges)
    node_order = list(G.nodes())
    return [
        [n for n in node_order if n in comp]
        for comp in nx.weakly_connected_components(H)
        if len(comp) > 1
    ]



def validate_workflow_dependencies(workflow: Workflow, on_error: str = "warn"):
    #TODO: change default on_error from "warn" to "raise" once the implementation is stable
    assert on_error in ["raise", "warn", "ignore"]

    G = workflow.task_graph
    outputs = set()

    def handle_on_error(node, input):
        msg = f"Node {node} requires {input} but it is not available"

        match on_error:
            case "raise":  raise Exception(msg)
            case "warn":   warnings.warn(msg)
            case "ignore": pass

    for node in nx.topological_sort(G):
        if G.in_degree(node) != 0:
            for input in node.registered_inputs:
                if input not in outputs:
                    handle_on_error(node, input)

        outputs.update(node.registered_outputs)
