import pytest
import networkx as nx
from unittest.mock import MagicMock
from dsbo.workflow import (
    implements_backward,
    assert_workflow_supports_backward_ad,
    assert_no_mixed_surrogate_usage,
    assert_leaf_surrogates,
    validate_workflow_dependencies,
    mergeable_components,
    Workflow,
)
from dsbo.modules import Module
from dsbo.modules.dsl import DependencyDict
from tests.workflow.conftest import (
    SimpleModule, BackwardModule, NoGradModule,
)


class TestImplementsBackward:
    def test_module_without_backward_returns_false(self):
        m = SimpleModule()
        assert not implements_backward(m)

    def test_module_with_backward_returns_true(self):
        m = BackwardModule()
        assert implements_backward(m)


class TestAssertWorkflowSupportsBackwardAd:
    def test_passes_when_all_modules_support_backward(self):
        m1 = BackwardModule()
        m2 = BackwardModule()
        wf = Workflow(m1 >> m2)
        assert_workflow_supports_backward_ad(wf)

    def test_raises_on_unsupported_module(self):
        wf = Workflow(SimpleModule())
        with pytest.raises(AssertionError, match="do not implement backward"):
            assert_workflow_supports_backward_ad(wf)

    def test_skips_no_grad_modules(self):
        m1 = NoGradModule()
        wf = Workflow(m1)
        assert_workflow_supports_backward_ad(wf)


class TestValidateWorkflowDependencies:
    def test_passes_when_inputs_satisfied(self):
        node = MagicMock(spec=Module)
        node.registered_inputs = []
        node.registered_outputs = ["out"]
        G = nx.DiGraph()
        G.add_edge(node, node)
        validate_workflow_dependencies(Workflow(SimpleModule()), on_error="raise")

    def test_raises_on_invalid_on_error(self):
        with pytest.raises(AssertionError):
            validate_workflow_dependencies(Workflow(SimpleModule()), on_error="invalid")

    def test_handle_on_error_warn_missing_input(self, recwarn):
        class NeedsInput(Module):
            def __init__(self):
                super().__init__()
                self._registered_inputs = ["missing_input"]
                self._registered_outputs = []
            def __call__(self, design, **kwargs):
                return {}

        m1 = SimpleModule()
        m2 = NeedsInput()
        wf = Workflow(m1 >> m2)
        validate_workflow_dependencies(wf, on_error="warn")
        assert len(recwarn) >= 1

    def test_handle_on_error_raise_missing_input(self):
        class NeedsInput(Module):
            def __init__(self):
                super().__init__()
                self._registered_inputs = ["missing_input"]
                self._registered_outputs = []
            def __call__(self, design, **kwargs):
                return {}

        m1 = SimpleModule()
        m2 = NeedsInput()
        wf = Workflow(m1 >> m2)
        with pytest.raises(Exception, match="not available"):
            validate_workflow_dependencies(wf, on_error="raise")


class TestMergeableComponents:
    def test_returns_empty_for_single_node(self, single_module_workflow):
        components = mergeable_components(single_module_workflow)
        assert components == []

    def test_chain_with_same_resources(self):
        m1 = SimpleModule()
        m2 = SimpleModule()
        wf = Workflow(m1 >> m2)
        components = mergeable_components(wf)
        assert len(components) == 1
        assert components[0] == [m1, m2]

    def test_chain_with_different_resources(self):
        m1 = SimpleModule()
        m2 = SimpleModule()
        m2.resources = MagicMock()
        wf = Workflow(m1 >> m2)
        components = mergeable_components(wf)
        assert components == []

    def test_diamond_not_contractible(self):
        m1 = SimpleModule()
        m2 = SimpleModule()
        m3 = SimpleModule()
        m4 = SimpleModule()
        wf = Workflow(DependencyDict({m1: [m2, m3], m2: [m4], m3: [m4]}))
        components = mergeable_components(wf)
        assert components == []

    def test_chain_is_contractible(self):
        m1 = SimpleModule()
        m2 = SimpleModule()
        m1.resources = None
        m2.resources = None
        wf = Workflow(m1 >> m2)
        components = mergeable_components(wf)
        assert len(components) == 1


class TestAssertNoMixedSurrogateUsage:
    def test_passes_without_surrogates(self, single_module_workflow):
        assert_no_mixed_surrogate_usage(single_module_workflow)

    def test_raises_with_mixed_surrogates(self):
        m = SimpleModule()
        m.surrogate = MagicMock()
        wf = Workflow(m)
        wf._surrogate = MagicMock()
        with pytest.raises(ValueError, match="cannot use both"):
            assert_no_mixed_surrogate_usage(wf)


class TestAssertLeafSurrogates:
    def test_passes_with_leaf_surrogate(self):
        m = SimpleModule()
        m.surrogate = MagicMock()
        wf = Workflow(m)
        assert_leaf_surrogates(wf)

    def test_raises_on_non_leaf_surrogate(self):
        m1 = SimpleModule()
        m2 = SimpleModule()
        m1.surrogate = MagicMock()
        wf = Workflow(m1 >> m2)
        with pytest.raises(ValueError, match="must be leaf"):
            assert_leaf_surrogates(wf)
