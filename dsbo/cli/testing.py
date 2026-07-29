from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any

import networkx as nx

from dsbo.modules import BatchFusedModule, FusedModule, Module
from dsbo.workflow import Workflow, validate_workflow_dependencies


ModuleCheck = Callable[[type, type], None]


def _construct(module_class: type) -> Any:
    try:
        return module_class()
    except Exception as error:
        raise AssertionError(
            f"{module_class.__name__} must be constructible without arguments."
        ) from error


def check_inherits_expected_base(module_class: type, expected_base: type) -> None:
    assert issubclass(module_class, expected_base), (
        f"{module_class.__name__} must inherit from {expected_base.__name__}."
    )


def check_module_is_concrete(module_class: type, expected_base: type) -> None:
    del expected_base
    missing = sorted(getattr(module_class, "__abstractmethods__", ()))
    assert not inspect.isabstract(module_class), (
        f"{module_class.__name__} must implement abstract methods: "
        f"{', '.join(missing)}."
    )


def check_default_constructor(module_class: type, expected_base: type) -> None:
    del expected_base
    _construct(module_class)


def check_module_base_state(module_class: type, expected_base: type) -> None:
    if not issubclass(expected_base, Module):
        return

    module = _construct(module_class)
    assert isinstance(module.registered_files, dict)
    assert isinstance(module.registered_inputs, list)
    assert isinstance(module.registered_outputs, list)
    assert hasattr(module, "resources"), (
        f"{module_class.__name__}.__init__() must call super().__init__()."
    )


def check_module_execution_signature(module_class: type, expected_base: type) -> None:
    if not issubclass(expected_base, Module):
        return

    if expected_base is Module:
        assert module_class.__call__ is not Module.__call__, (
            f"{module_class.__name__} must implement __call__(self, design, **kwargs)."
        )

    signature = inspect.signature(module_class.__call__)
    try:
        signature.bind(None, {})
    except TypeError as error:
        raise AssertionError(
            f"{module_class.__name__}.__call__ must accept a design argument."
        ) from error


def check_module_name_and_representation(module_class: type, expected_base: type) -> None:
    if not issubclass(expected_base, Module):
        return

    module = _construct(module_class)
    assert module.name == module_class.__name__
    assert repr(module) == module.name


def check_module_replication(module_class: type, expected_base: type) -> None:
    if not issubclass(expected_base, Module):
        return

    module = _construct(module_class)
    replica = module.replicate(0)
    assert replica is not module
    assert isinstance(replica, module_class)
    assert replica.i == 0
    assert replica.name == f"{module.name}[0]"


MODULE_CHECKS: tuple[ModuleCheck, ...] = (
    check_inherits_expected_base,
    check_module_is_concrete,
    check_default_constructor,
    check_module_base_state,
    check_module_execution_signature,
    check_module_name_and_representation,
    check_module_replication,
)


def parametrize_with_checks(
    module_classes: Iterable[type],
    *,
    expected_base: type = Module,
):
    """Parametrize a pytest test with DSBO module contract checks."""
    import pytest

    parameters = []
    for module_class in module_classes:
        for check in MODULE_CHECKS:
            bound_check = partial(check, expected_base=expected_base)
            parameters.append(
                pytest.param(
                    module_class,
                    bound_check,
                    id=f"{module_class.__name__}-{check.__name__}",
                )
            )
    return pytest.mark.parametrize("module_class,check", parameters)


def check_workflow(workflow: Workflow, *, require_fused: bool = False) -> None:
    """Validate the structural contract of a generated workflow definition."""
    assert isinstance(workflow, Workflow), (
        "Workflow factories must return a dsbo.workflow.Workflow instance."
    )

    graph = workflow.task_graph
    assert graph.number_of_nodes() > 0, "A workflow must contain at least one module."
    assert all(isinstance(node, Module) for node in graph.nodes), (
        "Every workflow graph node must be a dsbo.modules.Module instance."
    )
    assert nx.is_directed_acyclic_graph(graph), "A workflow graph must be acyclic."
    validate_workflow_dependencies(workflow, on_error="raise")

    if require_fused:
        fused_types = (FusedModule, BatchFusedModule)
        assert any(isinstance(node, fused_types) for node in graph.nodes), (
            "The fused workflow must contain a FusedModule or a module wrapped "
            "with batch_fuse()."
        )
