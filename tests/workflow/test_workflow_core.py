import pytest
import torch
import networkx as nx
from unittest.mock import MagicMock, patch, PropertyMock
from dsbo.workflow import Workflow
from dsbo.modules.dsl import DependencyDict
from dsbo.surrogates.graph_transformations import GraphMode
from dsbo.worker.spec import WorkerSpec
from tests.workflow.conftest import SimpleModule


class TestWorkflowConstruction:
    def test_init_from_single_module(self):
        wf = Workflow(SimpleModule())
        assert wf.task_graph.number_of_nodes() == 1

    def test_init_from_dependency_dict(self):
        m1 = SimpleModule()
        m2 = SimpleModule()
        wf = Workflow(DependencyDict({m1: [m2]}))
        assert wf.task_graph.number_of_nodes() == 2

    def test_init_rejects_self_loops(self):
        m = SimpleModule()
        dd = DependencyDict({m: [m]})
        with pytest.raises(AssertionError, match="self-loop"):
            Workflow(dd)

    def test_dependency_groups_propagated(self):
        m1 = SimpleModule()
        m2 = SimpleModule()
        dd = DependencyDict({m1: [m2]})
        dd.groups = [MagicMock()]
        wf = Workflow(dd)
        assert len(wf._dependency_groups) == 1


class TestWorkflowTaskGraph:
    def test_default_mode_returns_default_graph(self, single_module_workflow):
        assert single_module_workflow.task_graph_mode == GraphMode.DEFAULT

    def test_mode_resets_on_context_exit(self, single_module_workflow):
        with single_module_workflow.mode(GraphMode.MODULE_SURROGATE_EVAL):
            assert single_module_workflow.task_graph_mode == GraphMode.MODULE_SURROGATE_EVAL
        assert single_module_workflow.task_graph_mode == GraphMode.DEFAULT

    def test_task_graph_changes_with_mode(self, single_module_workflow):
        with single_module_workflow.mode(GraphMode.MODULE_SURROGATE_EVAL):
            G = single_module_workflow.task_graph
            assert G.number_of_nodes() == 1

    def test_workflow_surrogate_mode(self):
        m = SimpleModule()
        wf = Workflow(m)
        surrogate = MagicMock()
        surrogate._is_pretrained = True
        wf.patch(surrogate)
        assert wf.task_graph_mode == GraphMode.WORKFLOW_SURROGATE

    def test_task_graph_for_module_surrogate_train(self, single_module_workflow):
        G = single_module_workflow.task_graph_for(GraphMode.MODULE_SURROGATE_TRAIN)
        assert isinstance(G, nx.DiGraph)

    def test_task_graph_for_workflow_surrogate(self):
        m1 = SimpleModule()
        m2 = SimpleModule()
        wf = Workflow(m1 >> m2)
        surrogate = MagicMock()
        surrogate._is_pretrained = True
        wf._surrogate = surrogate
        G = wf.task_graph_for(GraphMode.WORKFLOW_SURROGATE)
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() == 1
        assert list(G.nodes)[0] == surrogate

    def test_num_tasks(self, chain_workflow):
        assert chain_workflow.num_tasks == 2

    def test_max_depth(self, chain_workflow):
        assert chain_workflow.max_depth == 2

    def test_max_depth_single_node(self, single_module_workflow):
        assert single_module_workflow.max_depth == 1


class TestGetUniqueWorkers:
    def test_empty_when_no_resources(self, single_module_workflow):
        workers = single_module_workflow.get_unique_workers()
        assert workers == set()

    def test_returns_resources(self):
        m1 = SimpleModule()
        m2 = SimpleModule()
        m3 = SimpleModule()
        spec1 = MagicMock(spec=WorkerSpec, name="spec_a")
        spec2 = MagicMock(spec=WorkerSpec, name="spec_b")
        m1.resources = spec1
        m2.resources = spec2
        m3.resources = spec2
        wf = Workflow(m1 >> m2 >> m3)
        workers = wf.get_unique_workers()
        assert len(workers) == 2
        assert spec1 in workers
        assert spec2 in workers

