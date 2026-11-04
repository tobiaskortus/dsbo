"""Regression guards for the graph-transformation plan builders.

These lock in behavior parity for the decomposed helpers in
``dsbo/surrogates/graph_transformations.py``: module-subgraph selection and
leaf contraction, workflow recorder wrapping, and dependency-group remapping.
They are intentionally construction-level (no execution), so they stay fast.
"""

import pytest
import torch

from dsbo.parameters import ParameterSpace, Continuous, Candidate
from dsbo.modules import dynamic_submodule, batch_fuse
from dsbo.modules.dsl import dynamic
from dsbo.workflow import Workflow
from dsbo.utils.io import Workdir, Input
from dsbo.utils.io.workdir import DefaultPaths, DefaultFilenames
from dsbo.surrogates import SurrogateStrategy
from dsbo.surrogates.module_surrogate import ModuleSurrogate
from dsbo.surrogates.workflow_surrogate import WorkflowSurrogate
from dsbo.worker.spec import LocalWorkerSpec
from dsbo.surrogates.graph_transformations import (
    build_module_surrogate_eval_graph,
    build_module_surrogate_training_plan,
    build_workflow_surrogate_training_plan,
)

from tests.surrogates.conftest import SimpleModule


class Strategy(SurrogateStrategy):
    def is_in_trust_region(self, candidate):
        return False

    def generate_candidates(self, candidate, space, finetune=False):
        return [candidate]


class _ModuleSurrogate(ModuleSurrogate):
    """Minimal concrete module surrogate; ``train`` mirrors how production
    surrogates expose their lifecycle task."""

    def __init__(self):
        super().__init__(strategy=Strategy())
        self.data_file = Input(Workdir.register_file("simulation", "events.h5"))

    @dynamic_submodule
    def train(self, design, **kwargs):
        return {}

    def state_dict(self):
        return {}

    def from_state_dict(self, state):
        return None


class _WorkflowSurrogate(WorkflowSurrogate):
    def __init__(self):
        super().__init__(strategy=Strategy())

    @dynamic_submodule
    def train(self, design, **kwargs):
        return {}

    def state_dict(self):
        return {}

    def from_state_dict(self, state):
        return None


def _candidate(value=1.0):
    return Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([value]))


class TestModuleSurrogateTrainingPlan:
    @pytest.mark.parametrize("dynamic_outer", [False, True])
    @pytest.mark.parametrize("with_ancestors", [False, True])
    def test_nested_replication_uses_copied_nodes(self, dynamic_outer, with_ancestors):
        target = SimpleModule()
        surrogate = _ModuleSurrogate()
        target.surrogate = surrogate
        inner = (
            SimpleModule() * 2 >> target * dynamic()
            if with_ancestors else target * 2
        )
        dependencies = (
            SimpleModule() * 2 >> inner * dynamic()
            if dynamic_outer else inner * 2
        )
        workflow = Workflow(dependencies)

        plan = build_module_surrogate_training_plan(
            workflow.task_graph,
            dependencies.groups,
            surrogate,
            surrogate.train,
            [_candidate()],
        )

        assert plan.trainer is not None
        producers = list(plan.graph.predecessors(plan.trainer))
        assert len(producers) == (1 if not dynamic_outer and not with_ancestors else 4)
        assert all(node in workflow.task_graph for node in producers)

    def test_selects_ancestor_subgraph_and_wires_one_trainer(self):
        producer = SimpleModule()
        consumer = SimpleModule()
        surrogate = _ModuleSurrogate()
        consumer.surrogate = surrogate
        wf = Workflow(producer >> consumer)
        task = surrogate.train

        plan = build_module_surrogate_training_plan(
            wf.task_graph,
            wf.surrogate_execution.dependency_groups,
            surrogate,
            task,
            [_candidate()],
        )

        assert plan.trainer is not None
        assert plan.train_submodule is task
        node_types = [type(n).__name__ for n in plan.graph.nodes()]
        assert node_types.count("BatchFusedModule") == 1
        assert any(
            type(u).__name__ == "SimpleModule"
            and type(v).__name__ == "BatchFusedModule"
            for u, v in plan.graph.edges()
        )

    def test_contracts_replicated_leaves_to_single_representative(self):
        template = SimpleModule()
        surrogate = _ModuleSurrogate()
        template.surrogate = surrogate
        wf = Workflow(template * 3)
        task = surrogate.train

        plan = build_module_surrogate_training_plan(
            wf.task_graph,
            wf.surrogate_execution.dependency_groups,
            surrogate,
            task,
            [_candidate()],
        )

        assert plan.trainer is not None
        node_types = [type(n).__name__ for n in plan.graph.nodes()]
        assert node_types.count("SimpleModule") == 1
        assert node_types.count("BatchFusedModule") == 1

    def test_empty_selection_returns_plan_without_trainer(self):
        template = SimpleModule()
        template.surrogate = _ModuleSurrogate()
        wf = Workflow(template * 3)
        other = _ModuleSurrogate()
        task = other.train

        plan = build_module_surrogate_training_plan(
            wf.task_graph,
            wf.surrogate_execution.dependency_groups,
            other,
            task,
            [_candidate()],
        )

        assert plan.trainer is None
        assert plan.train_submodule is None


def test_dynamic_module_replicas_share_one_surrogate():
    source, target = SimpleModule(), SimpleModule()
    target.surrogate = _ModuleSurrogate()
    dependencies = source * 3 >> target * dynamic()
    workflow = Workflow(dependencies)

    graph = build_module_surrogate_eval_graph(workflow.task_graph, dependencies.groups)

    assert len(graph) == 4
    assert sum(isinstance(node, _ModuleSurrogate) for node in graph) == 1
    assert len(graph.edges) == 3


@pytest.mark.parametrize("dynamic_outer", [False, True])
@pytest.mark.parametrize("grouped_inner", [False, True])
def test_nested_surrogate_eval_preserves_branch_groups(dynamic_outer, grouped_inner):
    target = SimpleModule()
    target.surrogate = _ModuleSurrogate()
    inner = target * 2 if grouped_inner else target | SimpleModule()
    dependencies = (
        SimpleModule() * 2 >> inner * dynamic()
        if dynamic_outer else inner * 2
    )
    workflow = Workflow(dependencies)

    graph = build_module_surrogate_eval_graph(workflow.task_graph, dependencies.groups)

    surrogates = [node for node in graph if isinstance(node, _ModuleSurrogate)]
    assert len(surrogates) == 2
    assert all(graph.in_degree(node) == (1 if dynamic_outer else 0) for node in surrogates)


class TestWorkflowSurrogateTrainingPlan:
    def test_recorders_preserve_wrapped_node_resources(self):
        resource = LocalWorkerSpec(cores=1, processes=1, memory="1GB")
        module = SimpleModule().to(resource)
        wf = Workflow(module)
        surrogate = _WorkflowSurrogate().to(resource)

        plan = build_workflow_surrogate_training_plan(
            wf.task_graph,
            wf.surrogate_execution.dependency_groups,
            surrogate,
            surrogate.train,
            [_candidate()],
        )

        recorder = next(n for n in plan.graph if type(n).__name__ == "ResultRecorder")
        assert recorder.resources is resource
        assert module.resources is resource
        assert all(node.resources is resource for node in plan.graph)

        other = LocalWorkerSpec(cores=2, processes=1, memory="2GB")
        recorder.to(other)
        assert recorder.resources is other
        assert module.resources is other

    def test_wraps_every_node_in_recorder_and_attaches_trainer(self):
        wf = Workflow(SimpleModule() * 3)
        surrogate = _WorkflowSurrogate()
        task = surrogate.train

        plan = build_workflow_surrogate_training_plan(
            wf.task_graph,
            wf.surrogate_execution.dependency_groups,
            surrogate,
            task,
            [_candidate(), _candidate(2.0)],
        )

        node_types = [type(n).__name__ for n in plan.graph.nodes()]
        assert node_types.count("ResultRecorder") == 3
        assert node_types.count("BatchFusedModule") == 1
        trainer = next(n for n in plan.graph.nodes() if type(n).__name__ == "BatchFusedModule")
        sink_edges = [u for u, v in plan.graph.edges() if v is trainer]
        assert len(sink_edges) == 3
        assert all(type(u).__name__ == "ResultRecorder" for u in sink_edges)
        assert all(
            node._required
            for node in plan.graph.nodes()
            if type(node).__name__ == "ResultRecorder"
        )

    def test_marks_only_sink_recorders_as_required(self):
        first, second = SimpleModule(), SimpleModule()
        wf = Workflow(first >> second)
        surrogate = _WorkflowSurrogate()

        plan = build_workflow_surrogate_training_plan(
            wf.task_graph,
            wf.surrogate_execution.dependency_groups,
            surrogate,
            surrogate.train,
            [_candidate()],
        )

        recorders = {
            recorder.inner: recorder
            for recorder in plan.graph.nodes()
            if type(recorder).__name__ == "ResultRecorder"
        }
        assert recorders[first]._required is False
        assert recorders[second]._required is True

    def test_does_not_wrap_batch_fused_nodes_in_recorder(self):
        fused = batch_fuse(_ModuleSurrogate())
        wf = Workflow(fused >> SimpleModule())
        surrogate = _WorkflowSurrogate()
        task = surrogate.train

        plan = build_workflow_surrogate_training_plan(
            wf.task_graph,
            wf.surrogate_execution.dependency_groups,
            surrogate,
            task,
            [_candidate(), _candidate(2.0)],
        )

        node_types = [type(n).__name__ for n in plan.graph.nodes()]
        assert node_types.count("ResultRecorder") == 1
        assert node_types.count("BatchFusedModule") == 2
        assert any(n is fused for n in plan.graph.nodes())

    def test_remaps_dependency_groups_to_recorder_keys(self):
        wf = Workflow(SimpleModule() * 3)
        surrogate = _WorkflowSurrogate()
        task = surrogate.train
        original = wf.surrogate_execution.dependency_groups

        plan = build_workflow_surrogate_training_plan(
            wf.task_graph,
            original,
            surrogate,
            task,
            [_candidate(), _candidate(2.0)],
        )

        groups = plan.graph._dependency_groups
        assert len(groups) == len(original)
        assert len(groups) == 1
        assert len(groups[0].nodes) == 3
        assert all(type(k).__name__ == "ResultRecorder" for k in groups[0].nodes)
