from contextlib import nullcontext
import math
import os
import matplotlib
import networkx as nx
from typing import List
from copy import deepcopy
from graphviz import Digraph
from itertools import groupby
import matplotlib
import matplotlib.colors as mcolors
from dask.distributed import Future, performance_report, as_completed
from matplotlib import pyplot as plt
from dsbo.logger import WorkflowTaskLogger
from dsbo.modules.compile import is_dry_run
from dsbo.utils.io import get_workdir, get_workdir_path, with_remote_workdir
from dsbo.worker.config import SlurmWorkerConfig
from typing import Callable, Optional, Sequence
from IPython.display import Image, display


class WorkflowTask:
    def __init__(self, name: str, task: Callable, worker_config: Optional[SlurmWorkerConfig] = None) -> None:
        self.name = name
        self.task = task
        self.worker_config = worker_config

    @with_remote_workdir
    def run(self, *args, **kwargs):
        with WorkflowTaskLogger(self) if not is_dry_run() else nullcontext() as logger:
            return self.task(*args, **kwargs)


def workflow_task(name, config)-> Callable[..., Callable[..., WorkflowTask]]:
    def decorator(obj) -> Callable[..., WorkflowTask]:
        task = None
        def wrapper(*args, **kwargs):
            nonlocal task
            if task is None:
                instance = obj(*args, **kwargs)
                task = WorkflowTask(name, instance, config)
            return task
        return wrapper
    return decorator


class FusedWorkflowTask(WorkflowTask):
    def __init__(self, *tasks):
        self.tasks = tasks
        for task in tasks:
            assert self.tasks[0].worker_config == task.worker_config, \
                    "All tasks in CompositeWorkflowTask must have the same worker configuration."

    def run(self, *args, **kwargs):
        results = {}
        for task in self.tasks:
            results.update(task.run(*args, **kwargs))

        return results
    
    @property
    def worker_config(self):
        return self.tasks[0].worker_config
    
    @property
    def name(self):
        return "[" + ",".join(task.name for task in self.tasks) + "]"

class DaskWorkflowRepeater:
    def __init__(self, task: WorkflowTask, repetitions: int) -> None:
        self.task = task
        self.repetitions = repetitions

    def run(self, design: dict) -> Sequence[Future]:
        raise NotImplementedError("DaskWorkflowRepeater.run is not implemented yet.")
    
class DynamicDaskWorkflowRepeater:
    def __init__(self, task: WorkflowTask, file_pattern: str) -> None:
        self.task = task
        self.file = None
    
    def run(self, design: dict) -> Sequence[Future]:
        raise NotImplementedError("DynamicDaskWorkflowRepeater.run is not implemented yet.")
    
    
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

def is_output_task(G: nx.DiGraph, task: WorkflowTask) -> bool:
    return G.out_degree(task) == 0

def get_dependent_task_groups(G: nx.DiGraph):
    levels_dict = task_levels(G)
    return [
        [node for node, _ in group]
        for _, group in groupby(sorted(levels_dict.items(), key=lambda kv: kv[1]), key=lambda kv: kv[1])
    ]


class WorkflowTaskGraph:
    def __init__(self, dependencies: dict[WorkflowTask, list[WorkflowTask]]) -> None:
        self._repr = nx.DiGraph(dependencies)

        assert_no_self_loops(self._repr)
        assert_no_duplicate_nodes(self._repr)
        #assert_no_disconnected_nodes(self._repr)

    def run(self, design: dict) -> dict[str, float]:
        """ Sequential execution of the task graph without worker distribution 
        For distributed execution, use TODO: Something?!?!?!
        
        Args:
            design (dict): Design parameters that should be used to evaluate the defined workflow.
        """
        results = {}
        from dsbo.worker.dask import WorkerContext
        worker = WorkerContext.get_current_worker()
        print(f"Using worker: {type(worker)}")

        future_map: dict[WorkflowTask, Future] = {}

        for depth, batch in enumerate(self.iterate_dependent_task_groups()):
            for task in batch:
                if worker is None:
                    result = task.run(design=design)
                    results.update(result)
                else:
                    pred_futures = [future_map[p] for p in self._repr.predecessors(task)]

                    future = worker.submit(task, design=design, priority=depth, dependencies=pred_futures)
                    future_map[task] = future

        if worker is not None:
            futures = future_map.values()
            for f in as_completed(futures):
                r = f.result()
                if r is not None:
                    results.update(r)
                            
                            
        return results

    def iterate_dependent_task_groups(self):
        for batch in get_dependent_task_groups(self._repr):
            yield batch

    def get_unique_workers(self):
        workers = set()
        for node in self._repr.nodes():
            if node.worker_config is not None:
                workers.add(node.worker_config)
        return workers

    def visualize(self, filename=None, format='png'):
        
        def _node_label(node):
            wc = getattr(node, 'worker_config', None)
            name = ','.join(t.name for t in node.tasks) if isinstance(node, FusedWorkflowTask) else (node.name if isinstance(node, WorkflowTask) else str(node))
            wk = 'default' if wc is None else f'gpu={wc.gpu_resources}, cpu={wc.job_cpu}, mem={wc.memory}'
            return f"{name}\n{wk}"
        
        fmt = format
        if filename:
            base, ext = os.path.splitext(filename)
            if ext:
                filename = base
                fmt = ext.lstrip('.')
        dot = Digraph(format=fmt)
        dot.attr(rankdir='LR')

        def wc_key(n):
            wc = getattr(n, 'worker_config', None)
            return 'default' if wc is None else f"mem={wc.memory}, cpu={wc.job_cpu}, gpu={wc.gpu_resources}"

        nodes = list(self._repr.nodes())
        keys = []
        groups = {}
        for n in nodes:
            k = wc_key(n)
            groups.setdefault(k, []).append(n)
            if k not in keys:
                keys.append(k)

        cmap = matplotlib.colormaps['Set3']
        color_map = {k: mcolors.to_hex(cmap(i % cmap.N)) for i, k in enumerate(keys)}

        rep_entry = {}
        rep_exit = {}

        for n in nodes:
            color = color_map.get(wc_key(n), '#d3d3d3')
            if isinstance(n, FusedWorkflowTask):
                cluster_name = f"cluster_{id(n)}"
                sub_ids = []
                with dot.subgraph(name=cluster_name) as c:
                    c.attr(label='', rank='same', style="dashed")
                    for i, sub in enumerate(n.tasks):
                        sid = f"{id(n)}_sub_{i}"
                        c.node(sid, label=_node_label(sub), style='filled', fillcolor=color, shape='box', color='black', penwidth='1')
                        sub_ids.append(sid)
                    for a, b in zip(sub_ids, sub_ids[1:]):
                        c.edge(a, b)
                if sub_ids:
                    rep_entry[n], rep_exit[n] = sub_ids[0], sub_ids[-1]
                else:
                    rep_entry[n] = rep_exit[n] = str(id(n))
            else:
                nid = str(id(n))
                dot.node(nid, label=_node_label(n), style='filled', fillcolor=color, shape='box', fontcolor='black')
                rep_entry[n] = rep_exit[n] = nid

        for u, v in self._repr.edges():
            src = rep_exit.get(u, str(id(u)))
            dst = rep_entry.get(v, str(id(v)))
            dot.edge(src, dst)

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
        return task_levels(self._repr)
    
    @property
    def max_depth(self):
        return max(self.levels.values()) + 1 if self.levels else 0
    

class DefaultSimRecoWorkflow(WorkflowTaskGraph):
    def __init__(self, sim_task: WorkflowTask, transform_task: WorkflowTask, reco_task: WorkflowTask) -> None:
        dependencies = {
            sim_task: [transform_task],
            transform_task: [reco_task]
        }
        super().__init__(dependencies)


def mergeable_components(task_graph: WorkflowTaskGraph, attr="worker_config") -> List[List[WorkflowTask]]:
    G = task_graph._repr
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


def update_task_graph(task_graph: WorkflowTaskGraph, groups: list[list[WorkflowTask]]) -> WorkflowTaskGraph:
    G = task_graph._repr.copy()
    for group in groups:
        merged = FusedWorkflowTask(*group)
        preds = set().union(*(G.predecessors(t) for t in group)) - set(group)
        succs = set().union(*(G.successors(t) for t in group)) - set(group)

        G.remove_nodes_from(group)
        G.add_node(merged)

        for p in preds: G.add_edge(p, merged)
        for s in succs: G.add_edge(merged, s)

    updated_task_graph= deepcopy(task_graph)
    updated_task_graph._repr = G

    return updated_task_graph


def optimize(workflow: WorkflowTaskGraph) -> WorkflowTaskGraph:
    return update_task_graph(workflow, mergeable_components(workflow))


