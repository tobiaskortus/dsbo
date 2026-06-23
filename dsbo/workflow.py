from __future__ import annotations

import os
import torch
import warnings
import networkx as nx

from copy import deepcopy
from itertools import chain
from itertools import groupby
import matplotlib.pyplot as plt
from contextlib import nullcontext
import matplotlib.colors as mcolors
from contextlib import contextmanager
import matplotlib.patches as mpatches
import torch.autograd.forward_ad as fw
from typing import Any, Iterable, List, Sequence, TYPE_CHECKING, overload

from dsbo.modules import Module
from dsbo.modules.dsl import DependencyDict
from dsbo.storage import get_current_storage
from dsbo.config import get_active_file_backend, config
from dsbo.parameters import Candidate
from dsbo.parameters import Candidate, ParameterSpace
from dsbo.backends.aido.surrogate import LocalSurrogate
from dsbo.utils.io import Workdir, get_subdir, get_workdir
from dsbo.surrogates.graph_transformations import (GraphMode, 
                                                   build_module_surrogate_eval_graph, 
                                                   build_module_surrogate_train_graph)

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

        values: list[Any] = []
        values = list(chain.from_iterable(item[key] for item in self))

        if values and isinstance(values[0], torch.Tensor):
            if all(isinstance(v, torch.Tensor) for v in values):
                return torch.stack(values)

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


class FusedModule(Module):
    """A group of modules with identical resources, fused into a single graph node."""

    def __init__(self, *modules: Module):
        super().__init__()
        self.modules = modules
        for m in modules:
            assert self.modules[0].resources == m.resources, \
                    "All modules in a FusedModule must have the same resources."

    def __call__(self, design: dict, **kwargs) -> dict:
        results = {}
        for m in self.modules:
            results.update(m(design=design))
        return results

    def run(self, design: dict, **kwargs) -> dict:
        results = {}
        for m in self.modules:
            results.update(m.run(design=design, **{k: v for k, v in kwargs.items()}))
        return results

    @property
    def resources(self):
        return self.modules[0].resources

    @resources.setter
    def resources(self, value):
        pass  # derived from child modules; ignore assignment from super().__init__()

    @property
    def name(self):
        return "[" + ",".join(m.name for m in self.modules) + "]"

    
    
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


def run_batch(workflow: Workflow, candidates: Sequence[Candidate], catch: tuple[Exception, ...] = ()) -> List[dict]:
    with torch.no_grad():
        from dsbo.worker.dask import WorkerContext
        worker = WorkerContext.get_current_worker()
        is_forward_ad = fw._current_level > -1
        is_grad_enabled = torch.is_grad_enabled()

        if worker is None:
            batch_results = []
            for i, candidate in enumerate(candidates):
                with Workdir.create_subdir(f"batch_{i}") if len(candidates) > 1 else nullcontext():
                    try:
                        results = {}
                        for batch in workflow.iterate_dependent_task_groups():
                            for task in batch:
                                result = task.run(
                                    design=candidate.values,
                                    is_forward_ad=is_forward_ad,
                                    is_grad_enabled=is_grad_enabled,
                                    unpack_design=True,
                                )
                                results.update(result)
                        batch_results.append(results)
                    except Exception as e:
                        batch_results.append({})
                        if not isinstance(e, catch):
                            raise
            return batch_results
        else:
            batch_delays = []
            batch_task_ids = []
            task_map: dict[str, str] = {}
            name_map: dict[str, str] = {}
            exec_ids = []
            batch_path = getattr(get_workdir(), "root", None)
            for i, candidate in enumerate(candidates):
                with Workdir.create_subdir(f"batch_{i}") if len(candidates) > 1 else nullcontext():
                    delayed_map = {}
                    task_ids = []
                    wdir = getattr(get_workdir(), "root", None)
                    subdir = get_subdir()
                    exec_id = os.path.join(wdir, subdir) if (wdir and subdir) else wdir
                    exec_ids.append(exec_id)
                    for depth, batch in enumerate(workflow.iterate_dependent_task_groups()):
                        for task in batch:
                            dependencies = [delayed_map[p] for p in workflow.task_graph.predecessors(task)]
                            task_id, delayed_obj = worker.schedule(task, candidate.values, dependencies, priority=depth)
                            delayed_map[task] = delayed_obj
                            task_ids.append(task_id)
                            batch_delays.append(delayed_obj)
                            task_map[delayed_obj.key] = exec_id
                            name_map[delayed_obj.key] = task.name
                    batch_task_ids.append(task_ids)

            storage = get_current_storage()
            if storage is not None:
                wf_record = storage.workflow.add_workflow(workflow.task_graph)
                storage.workflow.add_runs(wf_record, exec_ids, batch_path)
                plugin = storage.workflow.create_task_stream_plugin(task_map, name_map=name_map)
                worker.client.register_plugin(plugin, name="dsbo_task_stream")

            futures_computed = worker.compute(batch_delays)
            results_flat = worker.gather(futures_computed, catch=catch)
            lookup = {task_id: result for task_id, result in results_flat}
            return [dict(chain.from_iterable(lookup.get(tid, {}).items() for tid in tids))
                    for tids in batch_task_ids]


def run_batch_reverse_autograd(
    workflow: Workflow,
    candidates: Sequence[Candidate],
    grad_outputs: List[dict],
    catch: tuple[Exception, ...] = (),
) -> List[torch.Tensor]:
    from dsbo.worker.dask import WorkerContext
    from dsbo.config import get_active_file_backend
    worker = WorkerContext.get_current_worker()

    G = workflow.task_graph
    tasks_rev = list(reversed(list(nx.topological_sort(G))))

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

        storage = get_current_storage()
        if storage is not None:
            wf_record = storage.workflow.add_workflow(G)
            storage.workflow.add_runs(wf_record, exec_ids, batch_path)
            plugin = storage.workflow.create_task_stream_plugin(task_map, name_map=name_map)
            worker.client.register_plugin(plugin, name="dsbo_task_stream_bwd")

        futures_computed = worker.compute(batch_delays)
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
    def forward(ctx, workflow, design_tensors: torch.Tensor, space: ParameterSpace):
        ctx.workflow = workflow
        ctx.space = space
        ctx.save_for_backward(design_tensors)

        candidates = candidates_from_tensor(design_tensors, space)
        results = run_batch(workflow, candidates)
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
        return None, torch.stack(gradients, dim=0), None


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
        
        self._task_graph = None
        self._dependencies = dependencies
        self._dependency_groups = dependencies.groups

        assert_no_self_loops(self.task_graph)
        assert_no_duplicate_nodes(self.task_graph)


    def run(self, design: Candidate, catch: tuple[Exception, ...] = ()) -> dict[str, float]:
        return self.run_batch([design], catch=catch)[0]

    def run_batch(self, candidates: Sequence[Candidate], catch: tuple[Exception, ...] = ()):
        if fw._current_level > -1 or not torch.is_grad_enabled():
            with torch.no_grad():
                return ResultsBatch(run_batch(self, candidates, catch))
        else:
            design_tensors = torch.stack([c.tensor for c in candidates])
            space = candidates[0].space
            vals, keys, splits, is_scalar = RunBatchAutogradFn.apply(self, design_tensors, space)
            if not keys: return [{} for _ in candidates]

            return ResultsBatch([
                {
                    k: (chunk.squeeze(0) if is_scalar[k] else chunk)
                    for chunk, k in zip(torch.split(vals[i], splits), keys)
                }
                for i in range(len(candidates))
            ])


    def iterate_dependent_task_groups(self):
        for batch in get_dependent_task_groups(self.task_graph):
            yield batch

    def get_unique_workers(self) -> set[WorkerSpec]:
        workers = set()
        for node in self.task_graph.nodes():
            if node.resources is not None:
                workers.add(node.resources)
        return workers

    def visualize(self, filename=None, format='png'):
        import io
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.colors as mcolors
        from graphviz import Digraph
        from IPython.display import Image, display

        def _res_key(n):
            res = getattr(n, 'resources', None)
            return 'default' if res is None else f"mem={res.memory}, cpu={res.job_cpu}, gpu={res.gpu_resources}"

        def _node_label(n):
            name = n.name if hasattr(n, 'name') else str(n)
            return f"{name}\n{_res_key(n)}"

        def _build_color_map(nodes):
            keys = list(dict.fromkeys(_res_key(n) for n in nodes))
            cmap = matplotlib.colormaps['Set3']
            color_map = {k: mcolors.to_hex(cmap(i % cmap.N)) for i, k in enumerate(keys)}
            keys = [k for k in keys if k != 'default'] + (['default'] if 'default' in keys else [])
            return keys, color_map

        def _build_dep_group_key(nodes):
            lookup = {
                node: gi
                for gi, dg in enumerate(self._dependency_groups)
                for node in dg.nodes
                if node in nodes
            }
            return lambda n: ('dg', lookup[n]) if n in lookup else ('single', id(n))

        def _no_grad(n):
            return getattr(n, '_no_grad_op', False)

        def _faded(hex_color, blend=0.7):
            """Blend a hex colour toward white to signal disabled gradients."""
            r, g, b = mcolors.to_rgb(hex_color)
            r2 = r + (1 - r) * blend
            g2 = g + (1 - g) * blend
            b2 = b + (1 - b) * blend
            return mcolors.to_hex((r2, g2, b2))

        def _add_node(sg, n):
            nid = str(id(n))
            fill     = _faded(_color(n))        if _no_grad(n) else _color(n)
            style    = 'filled,dashed'          if _no_grad(n) else 'filled'
            sg.node(nid, label=_node_label(n), style=style, fillcolor=fill,
                    shape='box', fontcolor='black', color='black')
            rep_entry[n] = rep_exit[n] = nid

        def _add_replica_group(sg, first, last, count):
            _add_node(sg, first)
            _add_node(sg, last)
            dot.edge(
                str(id(first)), str(id(last)),
                label=f'×{count}', style='dashed', arrowhead='none',
                constraint='false', color='black', fontcolor='black',
                penwidth='1', fontsize='11', labelfloat='false',
            )

        def _render_with_legend(out_format):
            graph_bytes = dot.pipe(format='png')
            if not keys:
                return graph_bytes if out_format == 'png' else dot.pipe(format=out_format)

            graph_img = plt.imread(io.BytesIO(graph_bytes))
            h, w = graph_img.shape[:2]
            dpi, legend_h = 100, 0.55
            fig_w, fig_h = w / dpi, h / dpi + legend_h

            fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
            ax_g = fig.add_axes([0, legend_h / fig_h, 1, h / dpi / fig_h])
            ax_g.imshow(graph_img)
            ax_g.axis('off')

            ax_l = fig.add_axes([0, 0, 1, legend_h / fig_h])
            ax_l.axis('off')
            worker_patches = [mpatches.Patch(facecolor=color_map[k], edgecolor='#aaaaaa', linewidth=0.5, label=k) for k in keys]

            has_no_grad = any(_no_grad(n) for n in nodes)
            has_grad    = any(not _no_grad(n) for n in nodes)

            worker_legend = ax_l.legend(
                handles=worker_patches, ncol=min(4, len(worker_patches)), loc='center',
                frameon=False, fontsize=8, handlelength=1.2, handleheight=0.9,
                columnspacing=1.2, handletextpad=0.5,
                title='Compute Resources', title_fontsize=8,
            )
            worker_legend.get_title().set_fontweight('bold')

            if has_no_grad and has_grad:
                grad_patches = [
                    mpatches.Patch(facecolor='white', edgecolor='black',
                                   linewidth=1, linestyle='solid', label='grad'),
                    mpatches.Patch(facecolor='white', edgecolor='black',
                                   linewidth=1, linestyle='dashed', label='no grad'),
                ]
                ax_l.add_artist(worker_legend)
                ax_l.legend(
                    handles=grad_patches, ncol=1, loc='center right',
                    frameon=False, fontsize=8, handlelength=1.2, handleheight=0.9,
                    columnspacing=1.2, handletextpad=0.5,
                )

            buf = io.BytesIO()
            fig.savefig(buf, format=out_format, bbox_inches='tight', dpi=dpi)
            plt.close(fig)
            buf.seek(0)
            return buf.read()

        fmt = format
        if filename:
            base, ext = os.path.splitext(filename)
            if ext:
                filename, fmt = base, ext.lstrip('.')

        dot = Digraph(format=fmt)
        dot.attr(rankdir='LR')
        dot.attr('node', fontsize='10')
        dot.attr('edge', arrowsize='0.6')

        nodes = set(self.task_graph.nodes())
        keys, color_map = _build_color_map(nodes)
        _color = lambda n: color_map.get(_res_key(n), '#d3d3d3')
        _dep_group_key = _build_dep_group_key(nodes)
        levels = task_levels(self.task_graph)

        rep_entry: dict = {}
        rep_exit:  dict = {}
        hidden_nodes: set = set()

        level_groups: dict[int, list] = {}
        for n in nodes:
            level_groups.setdefault(levels.get(n, 0), []).append(n)

        for lvl, group in sorted(level_groups.items()):
            group.sort(key=lambda n: (
                getattr(n, 'i', None) if isinstance(getattr(n, 'i', None), int) else float('inf'),
                n.name if hasattr(n, 'name') else '',
            ))
            with dot.subgraph(name=f"rank_{lvl}") as sg:
                sg.attr(rank='same')
                for (kind, _), pg in groupby(group, key=_dep_group_key):
                    pg = list(pg)
                    if kind == 'dg' and len(pg) > 2:
                        hidden_nodes.update(pg[1:-1])
                        _add_replica_group(sg, pg[0], pg[-1], len(pg))
                    else:
                        for n in pg:
                            _add_node(sg, n)

        for u, v in self.task_graph.edges():
            if u in hidden_nodes or v in hidden_nodes:
                continue
            dot.edge(rep_exit.get(u, str(id(u))), rep_entry.get(v, str(id(v))))

      
        if filename is not None:
            out_path = f"{filename}.{fmt}"
            with open(out_path, 'wb') as f:
                f.write(_render_with_legend(fmt))
            return out_path
        
        from IPython.display import Image, display
        display(Image(_render_with_legend(format)))

    @property
    def task_graph(self) -> nx.DiGraph:
        if self._task_graph == None:
            self._task_graph = nx.DiGraph(self._dependencies) 

        return self._task_graph

    @property
    def levels(self):
        return task_levels(self.task_graph)
    
    @property
    def max_depth(self):
        return max(self.levels.values()) + 1 if self.levels else 0


class PatchableWorkflow(Workflow):
    def __init__(self, dependencies: dict[Module, Iterable[Module]] | Module) -> None:
        self.surrogate: WorkflowSurrogate | None = None
        self._mode: GraphMode | None = None
        self._default_task_graph: nx.DiGraph | None = None
        super().__init__(dependencies)

    def patch(self, surrogate: WorkflowSurrogate):
        self.surrogate = surrogate

    @contextmanager
    def mode(self, mode: GraphMode):
        old = self._mode
        self._mode = mode
        try: yield 
        finally: self._mode = old


    @property
    def task_graph(self):
        return self.task_graph_for(self._mode or (
            GraphMode.WORKFLOW_SURROGATE
            if self.surrogate is not None and self.surrogate._is_pretrained
            else GraphMode.DEFAULT
        ))

    def task_graph_for(self, mode: GraphMode) -> nx.DiGraph:
        if mode == GraphMode.MODULE_SURROGATE_EVAL:
            return build_module_surrogate_eval_graph(self)
        if mode == GraphMode.MODULE_SURROGATE_TRAIN:
            return build_module_surrogate_train_graph(self)
        if mode == GraphMode.WORKFLOW_SURROGATE:
            return nx.DiGraph({self.surrogate: []})

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

    def run_batch(self, candidates: list[Candidate], catch: tuple[Exception, ...] = ()) -> list[ResultsBatch]:
        if self.surrogate is not None and self.surrogate._is_pretrained:
            return [ResultsBatch([self.surrogate(candidate)]) for candidate in candidates]
        return super().run_batch(candidates, catch)

    def run(self, candidate: Candidate, catch: tuple[Exception, ...] = (), use_surrogate: bool = True) -> dict[str, float]:
        if self.surrogate and use_surrogate:
            if not self.surrogate.is_in_trust_region(candidate) or self.surrogate.surrogate is None:
                with self.mode(GraphMode.DEFAULT):
                    self.surrogate.fit(candidate, workflow_callback=super(PatchableWorkflow, self).run_batch)

            return super(PatchableWorkflow, self).run(candidate, catch)

        if self.uses_module_surrogates:
            with Workdir.create_subdir("module_surrogate_training"):
                for _, surrogate in self.module_surrogates.items():
                    if not surrogate.is_in_trust_region(candidate):
                        with self.mode(GraphMode.MODULE_SURROGATE_TRAIN):
                            from dsbo.utils.io.sharded import ShardedNumpyStorage
                            with config.use(file_backend=ShardedNumpyStorage):
                                surrogate.fit(candidate, workflow_callback=super(PatchableWorkflow, self).run_batch)

        with self.mode(GraphMode.DEFAULT):
            return super(PatchableWorkflow, self).run(candidate, catch)



def assert_no_mixed_surrogate_usage(workflow: PatchableWorkflow) -> None:
    if workflow.surrogate is not None and workflow.uses_module_surrogates:
        raise ValueError("Workflow cannot use both a workflow surrogate and module surrogates simultaneously.")


def assert_leaf_surrogates(workflow: PatchableWorkflow) -> None:
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


def update_task_graph(task_graph: Workflow, groups: list[list[Module]]) -> Workflow:
    G = task_graph.task_graph.copy()
    for group in groups:
        merged = FusedModule(*group)
        preds = set().union(*(G.predecessors(t) for t in group)) - set(group)
        succs = set().union(*(G.successors(t) for t in group)) - set(group)

        G.remove_nodes_from(group)
        G.add_node(merged)

        for p in preds: G.add_edge(p, merged)
        for s in succs: G.add_edge(merged, s)

    updated_task_graph= deepcopy(task_graph)
    updated_task_graph.task_graph = G

    return updated_task_graph


def optimize(workflow: Workflow) -> Workflow:
    return update_task_graph(workflow, mergeable_components(workflow))


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
