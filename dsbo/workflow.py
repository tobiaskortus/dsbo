from contextlib import nullcontext
import math
import os
import matplotlib
import networkx as nx
from typing import List
from copy import deepcopy
from itertools import groupby
from dask.distributed import Future, performance_report
from matplotlib import pyplot as plt
from dsbo.logger import WorkflowTaskLogger
from dsbo.utils.io import get_workdir, get_workdir_path, with_remote_workdir
from dsbo.worker.config import SlurmWorkerConfig
from typing import Callable, Optional, Sequence


class WorkflowTask:
    def __init__(self, name: str, task: Callable, worker_config: Optional[SlurmWorkerConfig] = None) -> None:
        self.name = name
        self.task = task
        self.worker_config = worker_config

    @with_remote_workdir
    def run(self, *args, **kwargs):
        with WorkflowTaskLogger(self) as logger:
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
        for task in self.tasks:
            task.run(*args, **kwargs)
    
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

def task_layout(G: nx.DiGraph, x_spacing=1, y_spacing=0.5):
    levels = task_levels(G)

    pos = {}
    for level in sorted(set(levels.values())):
        nodes = [n for n, l in levels.items() if l == level]

        # center nodes vertically
        offset = (len(nodes) - 1) / 2
        for i, node in enumerate(nodes):
            x = level * x_spacing
            y = (offset - i) * y_spacing
            pos[node] = (x, y)

    return pos

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

    def run(self, design: dict) -> tuple[float]:
        """ Sequential execution of the task graph without worker distribution 
        For distributed execution, use TODO: Something?!?!?!
        
        Args:
            design (dict): Design parameters that should be used to evaluate the defined workflow.
        """
        #TODO: improve aggregation of results from output tasks
        results = []
        from dsbo.worker.dask import WorkerContext
        worker = WorkerContext.get_current_worker()
        print(f"Using worker: {type(worker)}")

        for depth, batch in enumerate(self.iterate_dependent_task_groups()):
            futures = []
            for task in batch:
                if worker is None:
                    result = task.run(design=design)
                    if is_output_task(self._repr, task):
                        #TODO: well...
                        if result is not None:
                            if isinstance(result, (list, tuple)):
                                results.extend([r for r in result if r is not None])
                            else:
                                results.append(result)
                else:
                    future = worker.submit(task, design=design, priority=depth)
                    futures.append(future)

            if worker is not None:
                #TODO: well ...
                batch_results = tuple(f.result() for f in futures if f is not None)
                for r in batch_results:
                    if r is not None:
                        if isinstance(r, (list, tuple)):
                            results.extend([x for x in r if x is not None])
                        else:
                            results.append(r)
        return tuple(results)

    def iterate_dependent_task_groups(self):
        for batch in get_dependent_task_groups(self._repr):
            yield batch

    def get_unique_workers(self):
        workers = set()
        for node in self._repr.nodes():
            if node.worker_config is not None:
                workers.add(node.worker_config)
        return workers

    def visualize(self, fname: Optional[str] = None) -> None:
        max_nodes_in_level = max((len([n for n, l in self.levels.items() if l == lvl]) \
                                  for lvl in set(self.levels.values())), default=1)

        width = min(max(2 + self.max_depth * 2.5, 6), 30)
        height = min(max(1.5 + max_nodes_in_level * 1.2, 3.5), 20)

        fig, ax = plt.subplots(figsize=(width, height))

        pos = task_layout(self._repr, y_spacing=1, x_spacing=0.6)
        
        worker_groups = {}
        for node in self._repr.nodes():
            if node.worker_config is not None:
                wc = node.worker_config
                key = f"mem={wc.memory}, cpu={wc.job_cpu}, gpu={wc.gpu_resources}"
            else:
                key = "default"
            worker_groups.setdefault(key, []).append(node)

        cmap = matplotlib.colormaps["Set3"]
        keys = list(worker_groups.keys())
        color_map = {k: cmap(i % cmap.N) for i, k in enumerate(keys)}

        #Remove arrows for now. TODO: fix the positioning otf arrows to avoid overlap
        nx.draw_networkx_edges(self._repr, pos, arrowsize=0.001, ax=ax, connectionstyle='arc3,rad=0.0')

        for node in self._repr.nodes():
            x, y = pos[node]
            node_key = None
            for k, ns in worker_groups.items():
                if node in ns:
                    node_key = k
                    break
            color = color_map.get(node_key, (0.8, 0.8, 0.8, 1.0))
            ax.text(x, y, node.name, ha='center', va='center', zorder=3, fontsize=12,
                bbox=dict(boxstyle='square,pad=0.3', facecolor=color, edgecolor='black', linewidth=0.8))

        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=color_map[k], label=k) for k in keys]
        if patches:
            ncol = 2
            nrows = math.ceil(len(keys) / ncol)
            bottom_pad = 0.1 + 0.06 * nrows
            fig.legend(handles=patches, title="Worker Configuration (mem, cpu, gpu)", loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05), 
                       frameon=False, title_fontproperties={'weight':'bold'})
            
            fig.subplots_adjust(bottom=bottom_pad)

        ax.axis('scaled')
        ax.axis("off")

        if fname is not None:
            plt.savefig(fname, bbox_inches='tight', dpi=300)

    @property
    def levels(self):
        return task_levels(self._repr)
    
    @property
    def max_depth(self):
        return max(self.levels.values()) + 1 if self.levels else 0
    

class DefaultSimRecoWorkflow(WorkflowTaskGraph):
    def __init__(self, sim_task: WorkflowTask, transform_task: WorkflowTaskGraph, reco_task: WorkflowTask) -> None:
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


