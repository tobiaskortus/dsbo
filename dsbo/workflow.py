from __future__ import annotations

import os
import torch
import warnings
import matplotlib
import networkx as nx
from copy import deepcopy
from typing import Sequence
from itertools import chain
from graphviz import Digraph
from itertools import groupby
from typing import Iterable, List
from contextlib import nullcontext
import matplotlib.colors as mcolors
import torch.autograd.forward_ad as fw
from IPython.display import Image, display

from dsbo.modules import Module
from dsbo.utils.compile import dry_run
from dsbo.modules.dsl import DependencyDict
from dsbo.utils.io import Workdir, get_subdir
from dsbo.storage import get_current_storage
from dsbo.worker.config import SlurmWorkerConfig
from dsbo.parameters import Candidate, ParameterSpace
from dsbo.backends.aido.surrogate import LocalSurrogate
from dsbo.utils.io import Workdir, get_subdir, get_workdir


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
                            dependencies = [delayed_map[p] for p in workflow._representation.predecessors(task)]
                            task_id, delayed_obj = worker.schedule(task, candidate.values, dependencies, priority=depth)
                            delayed_map[task] = delayed_obj
                            task_ids.append(task_id)
                            batch_delays.append(delayed_obj)
                            task_map[delayed_obj.key] = exec_id
                            name_map[delayed_obj.key] = task.name
                    batch_task_ids.append(task_ids)

            storage = get_current_storage()
            if storage is not None:
                wf_record = storage.workflow.add_workflow(workflow._representation)
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
    from dsbo.utils.io.autograd import NumpyFile
    worker = WorkerContext.get_current_worker()

    G = workflow._representation
    tasks_rev = list(reversed(list(nx.topological_sort(G))))

    if worker is None:
        gradients = []
        for candidate, candidate_grad_output in zip(candidates, grad_outputs):
            try:
                grad_candidate = candidate.detach().requires_grad_()
                with torch.enable_grad():
                    NumpyFile.reset_gradient_state()
                    for task in tasks_rev:
                        is_sink = len(list(G.successors(task))) == 0
                        task.backward(grad_candidate.values, candidate_grad_output if is_sink else None)
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
                    forward_succs = list(G.successors(task))
                    is_sink = len(forward_succs) == 0
                    dependencies = [delayed_map[s] for s in forward_succs]
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


class RunBatchAutogradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, workflow, design_tensors: torch.Tensor, space: ParameterSpace):
        ctx.workflow = workflow
        ctx.space = space
        ctx.save_for_backward(design_tensors)

        candidates = candidates_from_tensor(design_tensors, space)
        results = run_batch(workflow, candidates)
        ctx.result_keys = list(results[0].keys()) if results else []

        #Handle cases where all tasks failed by throwing an error
        if not ctx.result_keys:
            return design_tensors.new_zeros(design_tensors.shape[0])

        vals = torch.tensor(
            [[float(r[k]) for k in ctx.result_keys] for r in results],
            dtype=design_tensors.dtype,
        )
        return vals.squeeze(-1) if len(ctx.result_keys) == 1 else vals

    @staticmethod
    def backward(ctx, grad_output):
        (design_tensors,) = ctx.saved_tensors
        candidates = candidates_from_tensor(design_tensors, ctx.space)

        grad_output_tensor = grad_output.view(len(candidates), -1)
        grad_outputs = [
            {k: grad_output_tensor[i, j] for j, k in enumerate(ctx.result_keys)}
            for i in range(len(candidates))
        ]

        gradients = run_batch_reverse_autograd(ctx.workflow, candidates, grad_outputs, catch=(RuntimeError,))
        return None, torch.stack(gradients, dim=0), None


def implements_backward(module: Module) -> bool:
    return type(module).backward is not Module.backward


def assert_workflow_supports_backward_ad(workflow: Workflow) -> None:
    unsupported = [m for m in workflow._representation.nodes()
                   if isinstance(m, Module) and not implements_backward(m)]
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

        self._representation = nx.DiGraph(dependencies)

        assert_no_self_loops(self._representation)
        assert_no_duplicate_nodes(self._representation)


    def run(self, design: Candidate, catch: tuple[Exception, ...] = ()) -> dict[str, float]:
        return self.run_batch([design], catch=catch)[0]

    def run_batch(self, candidates: Sequence[Candidate], catch: tuple[Exception, ...] = ()):
        if fw._current_level > -1 or not torch.is_grad_enabled():
            with torch.no_grad():
                return run_batch(self, candidates, catch)
        else:
            design_tensors = torch.stack([c.tensor for c in candidates])
            space = candidates[0].space
            return RunBatchAutogradFn.apply(self, design_tensors, space)


    def iterate_dependent_task_groups(self):
        for batch in get_dependent_task_groups(self._representation):
            yield batch

    def get_unique_workers(self) -> set[SlurmWorkerConfig]:
        workers = set()
        for node in self._representation.nodes():
            if node.resources is not None:
                workers.add(node.resources)
        return workers

    def visualize(self, filename=None, format='png'):
        
        def _node_label(node):
            res = getattr(node, 'resources', None)
            name = ','.join(m.name for m in node.modules) if isinstance(node, FusedModule) else (node.name if hasattr(node, 'name') else str(node))
            wk = 'default' if res is None else f'gpu={res.gpu_resources}, cpu={res.job_cpu}, mem={res.memory}'
            return f"{name}\n{wk}"
        
        fmt = format
        if filename:
            base, ext = os.path.splitext(filename)
            if ext:
                filename = base
                fmt = ext.lstrip('.')
        dot = Digraph(format=fmt)
        dot.attr(rankdir='LR')

        def res_key(n):
            res = getattr(n, 'resources', None)
            return 'default' if res is None else f"mem={res.memory}, cpu={res.job_cpu}, gpu={res.gpu_resources}"

        nodes = list(self._representation.nodes())
        keys = []
        groups = {}
        for n in nodes:
            k = res_key(n)
            groups.setdefault(k, []).append(n)
            if k not in keys:
                keys.append(k)

        cmap = matplotlib.colormaps['Set3']
        color_map = {k: mcolors.to_hex(cmap(i % cmap.N)) for i, k in enumerate(keys)}

        rep_entry = {}
        rep_exit = {}

        # Compute topological level for each node (determines the column in LR layout).
        levels = task_levels(self._representation)

        # Group nodes by level; within each level sort by (.i, name) so replicas of
        # the same module and repeat end up adjacent vertically.
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
                for n in group:
                    color = color_map.get(res_key(n), '#d3d3d3')
                    if isinstance(n, FusedModule):
                        sub_ids = []
                        for j, sub in enumerate(n.modules):
                            sid = f"{id(n)}_sub_{j}"
                            sg.node(sid, label=_node_label(sub), style='filled', fillcolor=color, shape='box', color='black', penwidth='1')
                            sub_ids.append(sid)
                        for a, b in zip(sub_ids, sub_ids[1:]):
                            dot.edge(a, b)
                        if sub_ids:
                            rep_entry[n], rep_exit[n] = sub_ids[0], sub_ids[-1]
                        else:
                            rep_entry[n] = rep_exit[n] = str(id(n))
                    else:
                        nid = str(id(n))
                        sg.node(nid, label=_node_label(n), style='filled', fillcolor=color, shape='box', fontcolor='black')
                        rep_entry[n] = rep_exit[n] = nid

        for u, v in self._representation.edges():
            src = rep_exit.get(u, str(id(u)))
            dst = rep_entry.get(v, str(id(v)))
            dot.edge(src, dst)
            #u_i = getattr(u, 'i', None)
            #v_i = getattr(v, 'i', None)
            # Cross-path edges (unindexed → indexed, or different .i) must not
            # constrain rank to avoid pulling replica columns out of alignment.
            #cross = (u_i is None and isinstance(v_i, int)) or (
            #    isinstance(u_i, int) and isinstance(v_i, int) and u_i != v_i
            #)
            #dot.edge(src, dst, constraint='false' if cross else 'true')

        if keys:
            rows = ''.join([f"<TR><TD BGCOLOR='{color_map[k]}' FIXEDSIZE='TRUE' WIDTH='20' HEIGHT='14'></TD><TD ALIGN='LEFT'>&nbsp;{k}</TD></TR>" for k in keys])
            table = f"<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='4'>{rows}</TABLE>"
            dot.node('legend', label=f'<{table}>', shape='none')

        if filename is not None:
            return dot.render(filename, cleanup=True)
        img = dot.pipe(format=format)
        display(Image(img))


    @property
    def levels(self):
        return task_levels(self._representation)
    
    @property
    def max_depth(self):
        return max(self.levels.values()) + 1 if self.levels else 0


class PatchableWorkflow(Workflow):
    """TODO: Integrate PatchableWorkflow into the main Workflow class, allowing for optional surrogate support without requiring a separate class."""
    def __init__(self, dependencies: dict[Module, Iterable[Module]] | Module) -> None:
        super().__init__(dependencies)
        self.surrogate = None

    def patch(self, surrogate: LocalSurrogate):
        self.surrogate = surrogate

    def run(self, candidate: Candidate, use_surrogate: bool = True) -> dict[str, float]:
        if self.surrogate and use_surrogate:
            if not self.surrogate.trust_region(candidate) or self.surrogate.model is None:
                #TODO: Accessing the _strategy attribute directly is a bit hacky!
                self.surrogate._strategy.update(candidate)
                self.surrogate.fit(candidate, workflow_callback=super().run_batch)
            
            print("Using surrogate predictions")
            with dry_run():
                self.surrogate.eval()
                scheme = super().run(candidate)
                x = candidate.tensor.unsqueeze(0).repeat(64, 1)
                y = self.surrogate(x).mean(dim=0)
                if y.numel() == 1: y = [y]
                return {k: v for k, v in  zip(scheme.keys(), y)}

        return super().run(candidate)
    

def mergeable_components(task_graph: Workflow, attr="resources") -> List[List[Module]]:
    G = task_graph._representation
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
    G = task_graph._representation.copy()
    for group in groups:
        merged = FusedModule(*group)
        preds = set().union(*(G.predecessors(t) for t in group)) - set(group)
        succs = set().union(*(G.successors(t) for t in group)) - set(group)

        G.remove_nodes_from(group)
        G.add_node(merged)

        for p in preds: G.add_edge(p, merged)
        for s in succs: G.add_edge(merged, s)

    updated_task_graph= deepcopy(task_graph)
    updated_task_graph._representation = G

    return updated_task_graph


def optimize(workflow: Workflow) -> Workflow:
    return update_task_graph(workflow, mergeable_components(workflow))


def validate_workflow_dependencies(workflow: Workflow, on_error: str = "warn"):
    #TODO: change default on_error from "warn" to "raise" once the implementation is stable
    assert on_error in ["raise", "warn", "ignore"]

    G = workflow._representation
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
