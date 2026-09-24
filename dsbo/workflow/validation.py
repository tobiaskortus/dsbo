from __future__ import annotations

import warnings

import networkx as nx

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dsbo.workflow.workflow import Workflow


def assert_no_mixed_surrogate_usage(workflow: Workflow) -> None:
    if (workflow.surrogate_execution.workflow_surrogate is not None and
        workflow.surrogate_execution.uses_module_surrogates):
        raise ValueError("Workflow cannot use both a workflow surrogate and module surrogates simultaneously.")


def assert_leaf_surrogates(workflow: Workflow) -> None:
    for module in workflow.task_graph.nodes:
        if hasattr(module, "surrogate") and module.surrogate is not None:
            if len(list(workflow.task_graph.successors(module))) > 0:
                raise ValueError("Module surrogates must be leaf nodes in the task graph.")


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
