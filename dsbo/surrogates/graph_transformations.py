from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, TYPE_CHECKING
import networkx as nx

from dsbo.modules import Module
from dsbo.utils.io import Workdir
from dsbo.modules.dsl import ModuleDslMixin
from dsbo.modules.utils import ResultRecorder
from dsbo.modules.fused import BatchFusedModule, batch_fuse, is_batch_fused
from dsbo.utils.io.workdir import AbstractFilePath, DefaultFilenames, DefaultPaths

if TYPE_CHECKING:
    from dsbo.parameters import CandidateBatch
    from dsbo.surrogates.module_surrogate import ModuleSurrogate
    from dsbo.surrogates.workflow_surrogate import WorkflowSurrogate


class GraphMode(Enum):
    DEFAULT = "default"
    WORKFLOW_SURROGATE_EVAL = "workflow_surrogate"
    WORKFLOW_SURROGATE_TRAIN = "workflow_surrogate_train"
    MODULE_SURROGATE_EVAL = "module_surrogate_eval"
    MODULE_SURROGATE_TRAIN = "module_surrogate_train"


@dataclass
class SurrogateTrainingPlan:
    graph: nx.DiGraph
    trainer: BatchFusedModule | None
    train_submodule: Module | None
    record_path: AbstractFilePath | None = None

    @property
    def candidates(self):
        if self.train_submodule is None:
            raise RuntimeError("Surrogate training plan has no lifecycle task wired.")
        
        return self.train_submodule.binding("candidates")

    @property
    def model_file(self) -> str:
        if self.trainer is None or self.train_submodule is None:
            raise RuntimeError("Surrogate training plan has no trainer wired.")
        
        return self.trainer.resolve_file(
            self.train_submodule.model_file_path.path,
            subdir="batch_reduce",
        )


def _has_surrogate(module: Module) -> bool:
    from dsbo.surrogates.module_surrogate import ModuleSurrogate
    if isinstance(module, ModuleSurrogate):
        return False
    return hasattr(module, "surrogate") and module.surrogate is not None


def _surrogate_nodes(dependency_groups: Iterable, G: nx.DiGraph) -> set[Module]:
    nodes: set[Module] = set()
    for g in dependency_groups:
        if _has_surrogate(g.template):
            nodes.update(g.nodes)
    nodes.update(n for n in G.nodes() if _has_surrogate(n) and n not in nodes)
    return nodes


def _replicate(template: Module, counts: dict[int, int]) -> Module:
    key = id(template)
    counts[key] = counts.get(key, -1) + 1
    return ModuleDslMixin.replicate(template, counts[key])


def _surrogate_replicator():
    """Returns a callable that hands out a distinct ``.i`` index per surrogate template."""
    counts: dict[int, int] = {}

    def make(template: Module) -> Module:
        return _replicate(template, counts)

    return make


def _validate_training_inputs(surrogate, training_task, candidates, *, workflow: bool):
    if workflow:
        if surrogate is None:
            raise RuntimeError(
                "Cannot build a workflow surrogate training plan without a "
                "workflow surrogate. Use workflow.patch(surrogate) first."
            )
        if training_task is None:
            raise RuntimeError("Workflow surrogate training requires a selected lifecycle task.")
        if candidates is None:
            raise RuntimeError("Workflow surrogate training requires generated candidates.")
    else:
        if surrogate is None:
            raise RuntimeError("Module surrogate training requires a surrogate target.")
        if training_task is None:
            raise RuntimeError("Module surrogate training requires a selected lifecycle task.")
        if candidates is None:
            raise RuntimeError("Module surrogate training requires generated candidates.")


def _batch_fuse_trainer(train_submodule, surrogate) -> BatchFusedModule:
    train_submodule.resources = getattr(surrogate, "resources", None)
    return batch_fuse(train_submodule)

def _bind_candidates(train_submodule, candidates) -> None:
    train_submodule.bind(candidates=candidates)
    return train_submodule


def _build_module_surrogate_trainer(G, producers, training_task, candidates) -> BatchFusedModule:
    _bind_candidates(training_task, candidates)
    trainer = _batch_fuse_trainer(training_task, producers[0].surrogate)
    trainer.i = 0
    trainer.set_alias(f"{producers[0].name} surrogate trainer")
    G.add_node(trainer)
    G.add_edges_from((producer, trainer) for producer in producers)
    return trainer


def _module_surrogate_subgraph(
    base_graph: nx.DiGraph, dependency_groups: Iterable, target: ModuleSurrogate
) -> nx.DiGraph:
    G = base_graph.copy()
    surrogated = {
        node
        for node in _surrogate_nodes(dependency_groups, G)
        if node.surrogate is target
    }
    keep = surrogated | set().union(*(nx.ancestors(G, n) for n in surrogated))
    G = nx.subgraph(G, keep).copy()
    return G


def _contract_equivalent_leaves(G: nx.DiGraph, dependency_groups: Iterable) -> nx.DiGraph:
    """Only applied when the selected subgraph is a set of independent leaves: collapses
    replicates sharing a template (or type, when no template) into a single representative
    so one trainer producer remains."""
    if G.nodes() and all(G.out_degree(n) == 0 for n in G.nodes()):
        block_of: dict[tuple[str, int], list[Module]] = defaultdict(list)
        grouped = set()

        # Inner groups precede outer groups; assign each node only once.
        for g in dependency_groups:
            for n in g.nodes:
                if n in G.nodes() and n not in grouped:
                    block_of[("template", id(g.template))].append(n)
                    grouped.add(n)

        for n in G.nodes():
            if n not in grouped:
                block_of[("type", type(n))].append(n)

        partition = [set(v) for v in block_of.values() if len(v) > 1]

        for block in partition:
            block = sorted(block, key=lambda x: getattr(x, "i", 0))
            for node in block[1:]:
                G = nx.contracted_nodes(G, block[0], node, self_loops=False)
    return G


def _wrap_with_recorders(base_graph: nx.DiGraph, record_path) -> tuple[nx.DiGraph, dict[Module, ResultRecorder]]:
    sinks = {node for node in base_graph.nodes() if base_graph.out_degree(node) == 0}
    node_to_recorder: dict[Module, ResultRecorder] = {
        node: ResultRecorder(node, record_path, required=node in sinks)
        for node in base_graph.nodes()
        if not is_batch_fused(node)
    }
    return nx.relabel_nodes(base_graph, node_to_recorder), node_to_recorder


def _attach_training_sink(G: nx.DiGraph, train_fused):
    G.add_node(train_fused)
    for node in list(G.nodes()):
        if node is not train_fused and G.out_degree(node) == 0:
            G.add_edge(node, train_fused)


def build_module_surrogate_eval_graph(base_graph: nx.DiGraph, dependency_groups: Iterable) -> nx.DiGraph:
    G = base_graph.copy()

    make_surrogate = _surrogate_replicator()
    mapping: dict[Module, Module] = {}

    def _map(module, nodes):
        surrogate = make_surrogate(module.surrogate)
        surrogate.set_alias(f"{module.name} (surrogate)")
        for node in nodes:
            mapping[node] = surrogate

    for group in dependency_groups:
        if _has_surrogate(group.template):
            _map(group.template, group.nodes)

    for node in G.nodes():
        if node not in mapping and _has_surrogate(node):
            _map(node, [node])

    return nx.relabel_nodes(G, mapping, copy=False)


def build_module_surrogate_training_plan(
    base_graph: nx.DiGraph,
    dependency_groups: Iterable,
    target: ModuleSurrogate,
    training_task: Module | None,
    candidates: CandidateBatch | None = None) -> SurrogateTrainingPlan:
    
    _validate_training_inputs(target, training_task, candidates, workflow=False)

    G = _module_surrogate_subgraph(base_graph, dependency_groups, target)
    G = _contract_equivalent_leaves(G, dependency_groups)

    producers = [n for n in G.nodes() if _has_surrogate(n) and n.surrogate is target]
    trainer = _build_module_surrogate_trainer(G, producers, training_task, candidates) if producers else None
    return SurrogateTrainingPlan(G, trainer, training_task if trainer else None)


def build_workflow_surrogate_training_plan(
    base_graph: nx.DiGraph,
    dependency_groups: Iterable,
    surrogate: WorkflowSurrogate,
    training_task: Module | None,
    candidates: CandidateBatch | None = None,
) -> SurrogateTrainingPlan:
    _validate_training_inputs(surrogate, training_task, candidates, workflow=True)

    record_path = Workdir.register_file(DefaultPaths.TRAINING_RECORDS, DefaultFilenames.RESULTS)
    G, node_to_recorder = _wrap_with_recorders(base_graph, record_path)

    _bind_candidates(training_task, candidates)
    train_fused = _batch_fuse_trainer(training_task, surrogate)
    train_fused.set_alias(training_task.name)
    
    _attach_training_sink(G, train_fused)
    G._dependency_groups = tuple(dg.relabel(node_to_recorder) for dg in dependency_groups)

    return SurrogateTrainingPlan(G, train_fused, training_task, record_path)
