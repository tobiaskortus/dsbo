import math
import time
from matplotlib import pyplot as plt
import matplotlib
import networkx as nx
from itertools import groupby
from dask.distributed import Future
from dsbo.worker.config import SlurmWorkerConfig
from concurrent.futures import as_completed
from typing import Callable, Optional, Sequence


class WorkflowTask:
    def __init__(self, name: str, task: Callable, worker_config: Optional[SlurmWorkerConfig] = None) -> None:
        self.name = name
        self.task = task
        self.worker_config = worker_config

    def run(self, *args, **kwargs):
        return self.task(*args, **kwargs)
    
    
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

def get_parallel_batches(G: nx.DiGraph):
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
        results = []
        from dsbo.worker.dask import WorkerContext
        worker = WorkerContext.get_current_worker()
        print(f"Using worker: {type(worker)}")
        
        for batch in self.iterate_task_batches():
            futures = []
            
            for task in batch:
                if worker is None:
                    result = task.run(design=design)
                    if is_output_task(self._repr, task):
                        if result is not None:
                            results.extend(result)
                else:
                    future = worker.submit(task, design=design)
                    futures.append(future)
                    
            if worker is not None:
                results = [f.result() for f in futures]
                
        return tuple(results)

    def iterate_task_batches(self):
        for batch in get_parallel_batches(self._repr):
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