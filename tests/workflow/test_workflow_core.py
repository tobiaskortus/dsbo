import pytest
import torch
import networkx as nx
from unittest.mock import MagicMock, patch
from dsbo.workflow import Workflow
from dsbo.modules.dsl import DependencyDict
from dsbo.surrogates import SurrogateStrategy
from dsbo.surrogates.workflow_surrogate import WorkflowSurrogate
from dsbo.surrogates.trainer import TrainingConfig
from dsbo.surrogates.graph_transformations import (
    GraphMode,
    build_module_surrogate_training_plan,
    build_workflow_surrogate_training_plan,
)
from dsbo.workflow.graph_context import WorkflowGraphExecutionStateState
from dsbo.worker.spec import WorkerSpec
from tests.workflow.conftest import SimpleModule
from tests.surrogates.conftest import TinySurrogateModel


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
        assert len(wf.surrogate_execution.dependency_groups) == 1


class TestWorkflowTaskGraph:
    def test_default_mode_returns_default_graph(self, single_module_workflow):
        assert single_module_workflow.graph_context.mode == GraphMode.DEFAULT

    def test_context_resets_on_exit(self, single_module_workflow):
        graph = single_module_workflow.task_graph.copy()
        with single_module_workflow.graph_context.use(
            WorkflowGraphExecutionStateState(GraphMode.MODULE_SURROGATE_EVAL, graph),
        ):
            assert single_module_workflow.graph_context.mode == GraphMode.MODULE_SURROGATE_EVAL
        assert single_module_workflow.graph_context.mode == GraphMode.DEFAULT

    def test_task_graph_changes_with_context(self, single_module_workflow):
        graph = single_module_workflow.task_graph.copy()
        with single_module_workflow.graph_context.use(
            WorkflowGraphExecutionStateState(GraphMode.MODULE_SURROGATE_EVAL, graph),
        ):
            G = single_module_workflow.task_graph
            assert G.number_of_nodes() == 1

    def test_graph_reads_do_not_implicitly_select_workflow_surrogate(self):
        class _Strategy(SurrogateStrategy):
            def is_in_trust_region(self, candidate):
                return False

            def generate_candidates(self, candidate, space, finetune=False):
                return [candidate]

        class StorageBackedSurrogate(WorkflowSurrogate):
            @property
            def has_initial_model(self):
                raise AssertionError("Graph reads must not query artifact storage.")

        m = SimpleModule()
        wf = Workflow(m)
        surrogate = StorageBackedSurrogate(
            TinySurrogateModel(),
            _Strategy(),
            TrainingConfig(max_epochs=1, grad_clip=1.0, batch_size=2, val_split=0.0),
        )
        with patch.object(type(surrogate), "bind_artifact_name", MagicMock()) as bind:
            wf.patch(surrogate)
        bind.assert_called_once()
        assert wf.graph_context.mode == GraphMode.DEFAULT
        assert list(wf.task_graph) == [m]

    def test_workflow_surrogate_eval_state(self):
        m1 = SimpleModule()
        m2 = SimpleModule()
        wf = Workflow(m1 >> m2)
        surrogate = MagicMock()
        wf.patch(surrogate)
        with wf.surrogate_execution.use_surrogates_if_available([None]):
            G = wf.task_graph
            assert isinstance(G, nx.DiGraph)
            assert G.number_of_nodes() == 1
            assert list(G.nodes)[0] == surrogate

    def test_nested_state_restoration(self, single_module_workflow):
        default_graph = single_module_workflow.task_graph

        with single_module_workflow.graph_context.use(
            WorkflowGraphExecutionStateState(GraphMode.MODULE_SURROGATE_EVAL, default_graph.copy()),
        ):
            with single_module_workflow.graph_context.use(
                WorkflowGraphExecutionStateState(GraphMode.DEFAULT, default_graph)
            ):
                assert single_module_workflow.graph_context.mode == GraphMode.DEFAULT
            assert single_module_workflow.graph_context.mode == GraphMode.MODULE_SURROGATE_EVAL
        assert single_module_workflow.graph_context.mode == GraphMode.DEFAULT

    def test_context_restoration_after_exception(self, single_module_workflow):
        with pytest.raises(RuntimeError, match="failed"):
            with single_module_workflow.graph_context.use(
                WorkflowGraphExecutionStateState(
                    GraphMode.MODULE_SURROGATE_EVAL,
                    single_module_workflow.task_graph.copy(),
                )
            ):
                raise RuntimeError("failed")

        assert single_module_workflow.graph_context.mode == GraphMode.DEFAULT

    def test_training_transitions_validate_inputs(self, single_module_workflow):
        graph = single_module_workflow.task_graph
        groups = single_module_workflow.surrogate_execution.dependency_groups
        with pytest.raises(RuntimeError, match="without a workflow surrogate"):
            build_workflow_surrogate_training_plan(graph, groups, None, MagicMock(), [])
        with pytest.raises(RuntimeError, match="lifecycle task"):
            build_workflow_surrogate_training_plan(graph, groups, MagicMock(), None, [])
        with pytest.raises(RuntimeError, match="generated candidates"):
            build_workflow_surrogate_training_plan(graph, groups, MagicMock(), MagicMock(), None)
        with pytest.raises(RuntimeError, match="surrogate target"):
            build_module_surrogate_training_plan(graph, groups, None, MagicMock(), [])
        with pytest.raises(RuntimeError, match="lifecycle task"):
            build_module_surrogate_training_plan(graph, groups, MagicMock(), None, [])
        with pytest.raises(RuntimeError, match="generated candidates"):
            build_module_surrogate_training_plan(graph, groups, MagicMock(), MagicMock(), None)

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
