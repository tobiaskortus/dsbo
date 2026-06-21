from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING
import networkx as nx

from dsbo.modules import Module
from dsbo.modules.dsl import ModuleDslMixin
from dsbo.surrogates.module_surrogate import ModuleSurrogate

if TYPE_CHECKING:
    from dsbo.workflow import Workflow



class GraphMode(Enum):
    DEFAULT = "default"
    WORKFLOW_SURROGATE = "workflow_surrogate"
    MODULE_SURROGATE_EVAL = "module_surrogate_eval"
    MODULE_SURROGATE_TRAIN = "module_surrogate_train"


def has_surrogate(module: Module) -> bool:
    if isinstance(module, ModuleSurrogate):
        return False
    return hasattr(module, "surrogate") and module.surrogate is not None


def surrogate_nodes(workflow: Workflow, G: nx.DiGraph) -> set[Module]:
    """Return all nodes that have an attached surrogate."""
    nodes: set[Module] = set()
    for g in workflow._dependency_groups:
        if has_surrogate(g.template):
            nodes.update(g.nodes)
    nodes.update(n for n in G.nodes() if has_surrogate(n) and n not in nodes)
    return nodes


def replicate(template: Module, counts: dict[int, int]) -> Module:
    key = id(template)
    counts[key] = counts.get(key, -1) + 1
    return ModuleDslMixin.replicate(template, counts[key])


def get_surrogate(surrogate_template: ModuleSurrogate, usage_counts: dict[int, int],
    replicate_counts: dict[int, int], original_name: str = "surrogate") -> ModuleSurrogate:
    key = id(surrogate_template)
    node = surrogate_template if usage_counts.get(key, 0) == 1 \
        else replicate(surrogate_template, replicate_counts)
    node.set_alias(original_name)
    return node


def build_module_surrogate_eval_graph(workflow: Workflow) -> nx.DiGraph:
    G = workflow.task_graph_for(GraphMode.DEFAULT).copy()

    replicate_counts: dict[int, int] = {}
    mapping: dict[Module, Module] = {}
    for group in workflow._dependency_groups:
        if not has_surrogate(group.template):
            continue
        surrogate = replicate(group.template.surrogate, replicate_counts)
        surrogate.set_alias(f"{group.template.name} (surrogate)")
        for node in group.nodes:
            mapping[node] = surrogate

    for node in G.nodes():
        if node in mapping or not has_surrogate(node):
            continue
        surrogate = replicate(node.surrogate, replicate_counts)
        surrogate.set_alias(f"{node.name} (surrogate)")
        mapping[node] = surrogate

    nx.relabel_nodes(G, mapping, copy=False)
    return G


def build_module_surrogate_train_graph(workflow: Workflow) -> nx.DiGraph:
    G = workflow.task_graph_for(GraphMode.DEFAULT).copy()
    surrogated = surrogate_nodes(workflow, G)
    if not surrogated:
        return G


    keep = surrogated | set().union(*(nx.ancestors(G, n) for n in surrogated))
    G = nx.subgraph(G, keep)

    if not G.nodes() or not all(G.out_degree(n) == 0 for n in G.nodes()):
        return G

    nodes = list(G.nodes())
    block_of: dict[tuple[str, int], list[Module]] = {}

    for g in workflow._dependency_groups:
        for n in g.nodes:
            if n in G.nodes():
                block_of.setdefault(("template", id(g.template)), []).append(n)

    for n in nodes:
        if not any(n in v for v in block_of.values()):
            block_of.setdefault(("type", type(n)), []).append(n)

    partition = [set(v) for v in block_of.values() if len(v) > 1]

    for block in partition:
        block = sorted(block, key=lambda x: getattr(x, "i", 0))
        for node in block[1:]:
            G = nx.contracted_nodes(G, block[0], node, self_loops=False)

    return G
