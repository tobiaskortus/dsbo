from __future__ import annotations

import os

import torch
import networkx as nx
from tqdm.auto import tqdm
from itertools import chain, groupby
from contextlib import nullcontext
import torch.autograd.forward_ad as fw
from typing import Any, List, Sequence, TYPE_CHECKING

from dsbo.modules import Module
from dsbo.parameters import Candidate
from dsbo.utils.io import Workdir, get_subdir, get_workdir
from dsbo.utils.random import get_random_engine
from dsbo.storage import get_current_storage
from dsbo.modules.fused import (
    file_overlays,
    is_batch_fused,
    create_candidate_symlinks,
)

if TYPE_CHECKING:
    from dsbo.workflow import Workflow, ResultsBatch
    from dsbo.parameters import ParameterSpace


def get_dependent_task_groups(G: nx.DiGraph):
    from dsbo.workflow import task_levels

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


def _update_task_stream_plugin(name, writer, task_map, name_map, tag, dask_scheduler=None):
    plugin = dask_scheduler.plugins.get(name)
    if plugin is None:
        return False
    plugin.update(writer, task_map, name_map, tag)
    return True


def register_task_stream(worker, graph, exec_ids, batch_path, task_map, name_map, name, tag=""):
    storage = get_current_storage()
    if storage is None:
        return
    workflow_record = storage.workflow.add_workflow(graph)
    storage.workflow.add_runs(workflow_record, exec_ids, batch_path)
    writer = storage.workflow._make_transition_writer()
    if worker.client.run_on_scheduler(_update_task_stream_plugin, name, writer, task_map, name_map, tag):
        return
    plugin = storage.workflow.create_task_stream_plugin(task_map, name_map=name_map, tag=tag)
    worker.client.register_plugin(plugin, name=name)


def run_batch(workflow: Workflow, candidates: Sequence[Candidate], catch: tuple[type[Exception], ...] = ()) -> List[dict]:
    with torch.no_grad():
        from dsbo.worker.dask import WorkerContext

        worker = WorkerContext.get_current_worker()
        if not candidates:
            return []

        count = len(candidates)
        task_graph = workflow.task_graph
        subdirs = candidate_subdirs(count)

        task_groups = list(get_dependent_task_groups(task_graph))
        progress = tqdm(total=num_tasks_for_batch(workflow, count), desc=f"Running workflow [{workflow.graph_context.mode.value}]", unit="tasks")
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
                    task_kwargs = dict(
                        design=candidates[0 if index is None else index].values,
                        subdir="batch_reduce" if index is None else subdirs[index],
                        file_overlays=file_overlays(task_graph, task, index, subdirs),
                        is_forward_ad=fw._current_level > -1,
                        is_grad_enabled=torch.is_grad_enabled(),
                        unpack_design=True,
                    )
                    if task.requires_design_tensor:
                        task_kwargs["design_tensor"] = candidates[0 if index is None else index].tensor
                    result = task.run(**task_kwargs)
                    if is_batch_fused(task) and workflow.graph_context.training_plan is None:
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
            progress.close()

            return [dict(chain.from_iterable(r.items() for r in results if r))
                    for results in batch_results]

        #Create exec_ids for task stream plugin
        batch_path = Workdir.get_current_dir()
        exec_ids = [os.path.join(batch_path, s) if s else batch_path for s in subdirs]

        batch_task_ids = [[] for _ in candidates]
        scheduled_tasks = []
        delayed_map, task_map, name_map = {}, {}, {}

        for depth, task, index, targets in execution_plan():
            deps = []
            for p in task_graph.predecessors(task):
                if is_batch_fused(p):
                    deps.append(delayed_map[(p, None)])
                elif index is None:
                    deps.extend(delayed_map[(p, i)] for i in range(count))
                else:
                    deps.append(delayed_map[(p, index)])

            subdir = "batch_reduce" if index is None else subdirs[index]
            overlays = file_overlays(task_graph, task, index, subdirs)

            with Workdir.create_subdir(subdir) if subdir else nullcontext():
                task_id, delayed = worker.schedule(
                    task,
                    candidates[0 if index is None else index].values,
                    deps,
                    design_tensor=(
                        candidates[0 if index is None else index].tensor
                        if task.requires_design_tensor
                        else None
                    ),
                    priority=depth,
                    file_overlays=overlays,
                )

            delayed_map[(task, index)] = delayed
            scheduled_tasks.append((task_id, task, index, delayed))
            for t in targets:
                batch_task_ids[t].append(task_id)

            eids = tuple(exec_ids[t] for t in targets)
            task_map[delayed.key] = eids[0] if len(eids) == 1 else eids
            name_map[delayed.key] = task.name

        register_task_stream(worker, task_graph, exec_ids, batch_path, 
                             task_map, name_map, "dsbo_task_stream", 
                             tag=workflow.graph_context.mode.value)

        futures = worker.compute([d for *_, d in scheduled_tasks])

        #Register progress bar update on future done callbacks.
        for f in futures:
            f.add_done_callback(lambda _: progress.update())

        results_flat = worker.gather(futures, catch=catch)
        progress.close()
        lookup = {task_id: result for task_id, result in results_flat}

        for _, task, _, _ in scheduled_tasks:
            if is_batch_fused(task):
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
    subdirs = candidate_subdirs(len(candidates))
    progress = tqdm(total=len(candidates) * len(tasks_rev), desc=f"Running workflow (reverse) [{workflow.graph_context.mode.value}]", unit="tasks")

    if worker is None:
        gradients = []
        for index, (candidate, candidate_grad_output) in enumerate(zip(candidates, grad_outputs)):
            try:
                grad_candidate = candidate.detach().requires_grad_()
                with torch.enable_grad():
                    get_active_file_backend().reset_gradient_state()
                    for task in tasks_rev:
                        is_sink = len(list(G.successors(task))) == 0
                        if task._no_grad_op:
                            continue

                        design_tensor = grad_candidate.tensor if getattr(task, "requires_design_tensor", False) else None
                        task.run_backward(
                            grad_candidate.values,
                            candidate_grad_output if is_sink else None,
                            design_tensor=design_tensor,
                            subdir=subdirs[index],
                            file_overlays=file_overlays(G, task, index, subdirs),
                        )
                        progress.update()
                gradients.append(
                    grad_candidate.grad.clone() if grad_candidate.grad is not None else torch.zeros_like(candidate.tensor)
                )
            except Exception as e:
                gradients.append(torch.zeros_like(candidate.tensor))
                if not isinstance(e, catch):
                    raise
        progress.close()
        return gradients
    else:
        batch_delays = []
        batch_task_ids = []
        task_map: dict[str, str] = {}
        name_map: dict[str, str] = {}
        exec_ids = []
        batch_path = getattr(get_workdir(), "workdir_root", None)

        for i, candidate in enumerate(candidates):
            candidate_grad_output = grad_outputs[i]
            with Workdir.create_subdir(f"batch_{i}") if len(candidates) > 1 else nullcontext():
                delayed_map = {}
                task_ids = []
                wdir = getattr(get_workdir(), "workdir_root", None)
                subdir = get_subdir()
                exec_id = os.path.join(wdir, subdir) if (wdir and subdir) else wdir
                exec_id = os.path.join(exec_id, "backward")  # Add a separate backward tag.
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
                        file_overlays=file_overlays(G, task, i, subdirs),
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
            tag=workflow.graph_context.mode.value,
        )

        futures_computed = worker.compute(batch_delays)
        for f in futures_computed:
            f.add_done_callback(lambda _: progress.update())
            
        results_flat = worker.gather(futures_computed, catch=catch)
        progress.close()
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
        ctx.graph_state = workflow.graph_context.current

        candidates = candidates_from_tensor(design_tensors, space)

        cache = workflow._get_cache()
        use_cache = (
            cache.config.should_cache(workflow.graph_context.mode)
            if cache and not any(is_batch_fused(task) for task in workflow.task_graph)
            else False
        )

        if use_cache:
            results = cache.load_or_compute(
                candidates,
                lambda misses: run_batch(workflow, misses, catch),
            )
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
        from dsbo.workflow import ResultsBatch

        (design_tensors,) = ctx.saved_tensors
        candidates = candidates_from_tensor(design_tensors, ctx.space)
        n = len(candidates)

        grad_flat = grad_output.view(n, -1)
        grad_outputs = ResultsBatch.from_tensor(
            grad_flat, ctx.result_keys, ctx.splits, ctx.result_is_scalar
        )

        with ctx.workflow.graph_context.use(ctx.graph_state):
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
