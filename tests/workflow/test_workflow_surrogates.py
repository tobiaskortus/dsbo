import pytest
import torch
from unittest.mock import MagicMock, patch
from dsbo.parameters import ParameterSpace, Continuous, Candidate
from dsbo.modules import Module, dynamic_submodule
from dsbo.config import config
from dsbo.utils.io import Workdir, Input
from dsbo.utils.cache import CacheConfig, CachePolicy
from dsbo.surrogates import SurrogateBase, SurrogateStrategy
from dsbo.surrogates.graph_transformations import GraphMode
from dsbo.workflow import Workflow, RunBatchAutogradFn
from dsbo.surrogates.workflow_surrogate import DefaultWorkflowSurrogate
from dsbo.surrogates.diffusion import MlpDiffusion1D
from tests.workflow.conftest import SimpleModule, SurrogateModule, FailingModule
from dsbo.workflow.workflow import ResultsBatch


class WorkflowSurrogateModule(Module):
    _is_pretrained = True
    requires_design_tensor = True

    def __init__(self):
        super().__init__()
        self.surrogate = object()
        self.keys = ["out"]
        self.n = 1

    def is_in_trust_region(self, candidate):
        return True

    def select_training_task(self, candidate):
        return None

    def __call__(self, design, **kwargs):
        assert isinstance(design, dict)
        assert isinstance(kwargs["design_tensor"], torch.Tensor)
        return {"out": torch.tensor(42.0)}



class TestModuleSurrogateChecks:
    def test_finds_modules_with_surrogates_empty_when_none(self, single_module_workflow):
        assert single_module_workflow.surrogate_execution.module_surrogates == {}

    def test_finds_modules_with_surrogates(self):
        m = SimpleModule()
        m.surrogate = MagicMock()
        wf = Workflow(m)
        surrogates = wf.surrogate_execution.module_surrogates
        assert len(surrogates) == 1
        assert m in surrogates

    def test_uses_module_surrogates_true(self):
        m = SimpleModule()
        m.surrogate = MagicMock()
        wf = Workflow(m)
        assert wf.surrogate_execution.uses_module_surrogates

    def test_uses_module_surrogates_false(self, single_module_workflow):
        assert not single_module_workflow.surrogate_execution.uses_module_surrogates

    def test_discovery_does_not_replace_active_state(self):
        module = SimpleModule()
        module.surrogate = SurrogateModule()
        workflow = Workflow(module)
        with workflow.surrogate_execution.use_surrogates_if_available([None]):
            assert module in workflow.surrogate_execution.module_surrogates
            assert workflow.graph_context.mode == GraphMode.MODULE_SURROGATE_EVAL


class TestSurrogateTrainingSessions:
    def test_workflow_surrogate_executes_selected_task(self):
        m = SimpleModule()
        wf = Workflow(m)
        surrogate = MagicMock()
        surrogate.surrogate = None
        training_task = TrainOnlySurrogate().train
        surrogate.select_training_task.return_value = training_task
        surrogate.generate_training_candidates.return_value = []
        surrogate.training_stage.return_value = "trained"
        wf.patch(surrogate)

        cand = Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([3.0]))
        plan = MagicMock(
            model_file="model.pt",
            candidates=[],
            graph=wf.graph_context.graph,
        )
        with patch(
            "dsbo.surrogates.lifecycle.build_workflow_surrogate_training_plan",
            return_value=plan,
        ) as build_state, patch.object(wf, "run_batch") as run_training_batch:
            session = next(wf.surrogate_execution.pending_training_sessions(cand))
            with session.activate() as training_candidates:
                wf.run_batch(training_candidates)

            surrogate.select_training_task.assert_called_once_with(cand)
            surrogate.generate_training_candidates.assert_called_once_with(
                cand, training_task,
            )
            build_state.assert_called_once_with(
                wf.graph_context.graph,
                wf.surrogate_execution.dependency_groups,
                surrogate,
                training_task,
                [],
            )
            run_training_batch.assert_called_once_with([])
            surrogate.training_stage.assert_called_once_with(training_task)
            surrogate.publish_trained_model.assert_called_once_with("model.pt", stage="trained")

    def test_workflow_surrogate_skips_when_no_task_selected(self):
        m = SimpleModule()
        wf = Workflow(m)
        surrogate = MagicMock()
        surrogate.select_training_task.return_value = None
        wf.patch(surrogate)

        cand = Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([3.0]))
        sessions = list(wf.surrogate_execution.pending_training_sessions(cand))
        surrogate.select_training_task.assert_called_once_with(cand)
        assert sessions == []

    def test_module_surrogate_executes_selected_task(self):
        m = SimpleModule()
        m.surrogate = MagicMock()
        training_task = TrainOnlySurrogate().train
        m.surrogate.select_training_task.return_value = training_task
        m.surrogate.generate_training_candidates.return_value = []
        m.surrogate.training_stage.return_value = "trained"
        wf = Workflow(m)

        cand = Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([3.0]))
        plan = MagicMock(
            model_file="model.pt",
            candidates=[],
            graph=wf.graph_context.graph,
        )
        with patch(
            "dsbo.surrogates.lifecycle.build_module_surrogate_training_plan",
            return_value=plan,
        ) as build_state, patch.object(wf, "run_batch") as run_training_batch:
            session = next(wf.surrogate_execution.pending_training_sessions(cand))
            with session.activate() as training_candidates:
                wf.run_batch(training_candidates)

            m.surrogate.select_training_task.assert_called_once_with(cand)
            m.surrogate.generate_training_candidates.assert_called_once_with(
                cand, training_task,
            )
            build_state.assert_called_once_with(
                wf.graph_context.graph,
                wf.surrogate_execution.dependency_groups,
                m.surrogate,
                training_task,
                [],
            )
            run_training_batch.assert_called_once_with([])
            m.surrogate.training_stage.assert_called_once_with(training_task)
            m.surrogate.publish_trained_model.assert_called_once_with("model.pt", stage="trained")

    def test_module_surrogate_skips_when_no_task_selected(self):
        m = SimpleModule()
        m.surrogate = MagicMock()
        m.surrogate.select_training_task.return_value = None
        wf = Workflow(m)

        cand = Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([3.0]))
        sessions = list(wf.surrogate_execution.pending_training_sessions(cand))
        m.surrogate.select_training_task.assert_called_once_with(cand)
        assert sessions == []


class TestRunWorkflowWithSurrogate:
    def _tiny_diffusion_surrogate(self, objective_dim=2, parameter_dim=1, sample_count=3):
        surrogate = DefaultWorkflowSurrogate(MagicMock())
        surrogate.keys = [f"k{i}" for i in range(objective_dim)]
        surrogate.n = 1
        surrogate.sample_count = sample_count
        surrogate.num_infer_timesteps = 4
        surrogate._mean_x = None
        surrogate._std_x = None
        surrogate._mean_y = None
        surrogate._std_y = None
        surrogate._diffusion_model = MlpDiffusion1D(
            objective_dim=objective_dim,
            parameter_dim=parameter_dim,
            emb_dim=16,
            hidden_dim=16,
            n_layers=2,
        )
        surrogate._create_schedulers()
        return surrogate

    def test_diffusion_surrogate_samples_n_values_per_key(self):
        surrogate = self._tiny_diffusion_surrogate(objective_dim=2, sample_count=3)
        design_tensor = torch.tensor([3.0])

        with patch.object(surrogate, "get_runtime_model", side_effect=lambda: None):
            result = surrogate({"x": torch.tensor(99.0)}, design_tensor=design_tensor)

        assert result["k0"].shape == (3,)
        assert result["k1"].shape == (3,)
        assert torch.isfinite(result["k0"]).all() and torch.isfinite(result["k1"]).all()

    def test_diffusion_surrogate_samples_are_distinct_given_distinct_conditions(self):
        surrogate = self._tiny_diffusion_surrogate(objective_dim=1, sample_count=1)
        d1 = torch.tensor([1.0])
        d2 = torch.tensor([5.0])

        with patch.object(surrogate, "get_runtime_model", side_effect=lambda: None):
            y1 = surrogate({"x": torch.tensor(0.0)}, design_tensor=d1)["k0"]
            y2 = surrogate({"x": torch.tensor(0.0)}, design_tensor=d2)["k0"]

        assert torch.isfinite(y1).all() and torch.isfinite(y2).all()

    def test_diffusion_surrogate_backward_flows_gradient_to_design(self):
        surrogate = self._tiny_diffusion_surrogate(objective_dim=1, parameter_dim=1, sample_count=1)
        design_tensor = torch.tensor([3.0], requires_grad=True)

        with patch.object(surrogate, "get_runtime_model", side_effect=lambda: None):
            surrogate.backward(
                {"x": torch.tensor(0.0)},
                {"k0": torch.tensor([1.0])},
                design_tensor=design_tensor,
            )

        assert design_tensor.grad is not None
        assert design_tensor.grad.shape == (1,)
        assert torch.isfinite(design_tensor.grad)

    def test_workflow_surrogate(self, single_module_workflow, candidate):
        surrogate = WorkflowSurrogateModule()
        surrogate._is_pretrained = True
        single_module_workflow.patch(surrogate)
        result = single_module_workflow.run(candidate)
        assert torch.allclose(result["out"], torch.tensor(42.0))

    def test_workflow_surrogate_batch(self, single_module_workflow, candidates):
        surrogate = WorkflowSurrogateModule()
        surrogate._is_pretrained = True
        single_module_workflow.patch(surrogate)

        with single_module_workflow.surrogate_execution.use_surrogates_if_available(len(candidates)):
            results = single_module_workflow.run_batch(candidates)
        assert len(results) == len(candidates)
        for result in results:
            assert torch.allclose(result["out"], torch.tensor(42.0))

    def test_workflow_surrogate_backward_routes_through_saved_context(self, single_module_workflow, candidate):
        surrogate = WorkflowSurrogateModule()
        surrogate._is_pretrained = True
        single_module_workflow.patch(surrogate)

        seen_modes = []

        def _tracking_reverse(workflow, *args, **kwargs):
            seen_modes.append(workflow.graph_context.mode)
            return [torch.zeros_like(candidate.tensor)]

        with patch("dsbo.workflow.runner.run_batch_reverse_autograd", side_effect=_tracking_reverse):
            x = torch.tensor([[3.0]], requires_grad=True)
            with single_module_workflow.surrogate_execution.use_surrogates_if_available([candidate]):
                vals, keys, _, _ = RunBatchAutogradFn.apply(
                    single_module_workflow, x, candidate.space, (),
                )
                vals.sum().backward()

        assert seen_modes == [GraphMode.WORKFLOW_SURROGATE_EVAL]
        assert single_module_workflow.graph_context.mode == GraphMode.DEFAULT

    def test_module_surrogate(self, candidate):
        m = SimpleModule()
        m.surrogate = SurrogateModule()
        wf = Workflow(m)
        result = wf.run(candidate)
        assert torch.allclose(result["out"], torch.tensor(99.0))

    def test_module_surrogate_batch_not_implemented(self, candidates):
        m = SimpleModule()
        m.surrogate = SurrogateModule()
        wf = Workflow(m)
        with pytest.raises(NotImplementedError, match="not supported"):
            wf.run_batch(candidates)

    def test_empty_module_surrogate_batch_does_not_transition(self):
        module = SimpleModule()
        module.surrogate = SurrogateModule()
        workflow = Workflow(module)

        with patch(
            "dsbo.surrogates.lifecycle.build_module_surrogate_eval_graph",
        ) as transition:
            assert workflow.run_batch([]) == []

        transition.assert_not_called()
        assert workflow.graph_context.mode == GraphMode.DEFAULT


class TestUseSurrogatesFlag:
    def test_run_false_skips_surrogate_training(self):
        m = SimpleModule()
        wf = Workflow(m)
        surrogate = MagicMock()
        training_task = TrainOnlySurrogate().train
        surrogate.select_training_task.return_value = training_task
        surrogate.generate_training_candidates.return_value = []
        wf.patch(surrogate)

        cand = Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([3.0]))
        with patch.object(wf, "run_batch") as run_batch, \
             patch.object(wf.surrogate_execution, "pending_training_sessions") as pending:
            wf.run(cand, use_surrogates=False)

        pending.assert_not_called()
        surrogate.select_training_task.assert_not_called()
        surrogate.generate_training_candidates.assert_not_called()
        run_batch.assert_called_once_with([cand], (), False)

    def test_run_false_stays_in_default_mode(self):
        module = SimpleModule()
        module.surrogate = SurrogateModule()
        workflow = Workflow(module)

        cand = Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([3.0]))
        with patch.object(workflow, "run_batch", return_value=ResultsBatch([{}])):
            workflow.run(cand, use_surrogates=False)

        assert workflow.graph_context.mode == GraphMode.DEFAULT

    def test_run_batch_false_does_not_trigger_surrogate_execution(self):
        module = SimpleModule()
        module.surrogate = SurrogateModule()
        workflow = Workflow(module)

        cand = Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([3.0]))
        with torch.no_grad(), \
             patch("dsbo.workflow.workflow.run_batch", return_value=ResultsBatch([])) as run_batch:
            results = workflow.run_batch([cand], use_surrogates=False)

        run_batch.assert_called_once()
        assert workflow.graph_context.mode == GraphMode.DEFAULT
        assert results == []


class _StubStrategy(SurrogateStrategy):
    def __init__(self):
        self.finetune_flags = []

    def is_in_trust_region(self, candidate):
        return False

    def generate_candidates(self, candidate, space, finetune=False):
        self.finetune_flags.append(finetune)
        return [candidate]


class NoOpSurrogate(SurrogateBase):
    def __init__(self):
        super().__init__(strategy=_StubStrategy())
        self.calls: list[str] = []
        self.data_file = Input(Workdir.register_file("simulation", "events.h5"))

    def state_dict(self) -> dict:
        return {}

    def from_state_dict(self, state: dict) -> None:
        pass


class TrainOnlySurrogate(NoOpSurrogate):
    @dynamic_submodule
    def train(self, design, **kwargs):
        self.calls.append("train")
        return {}


class PretrainFinetuneSurrogate(NoOpSurrogate):
    @dynamic_submodule
    def pretrain(self, design, **kwargs):
        self.calls.append("pretrain")
        return {}

    @dynamic_submodule
    def finetune(self, design, **kwargs):
        self.calls.append("finetune")
        return {}


class TestTrainingTaskSelection:
    candidate = Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([3.0]))

    def test_implements_reads_lifecycle_not_descriptor(self):
        single = TrainOnlySurrogate()
        assert single.implements_train
        assert not single.implements_pretrain
        assert not single.implements_finetune
        assert not single.implements_pretrain_and_finetune

        two = PretrainFinetuneSurrogate()
        assert not two.implements_train
        assert two.implements_pretrain
        assert two.implements_finetune
        assert two.implements_pretrain_and_finetune

    def test_no_lifecycle_implemented_rejected_at_construction(self):
        with pytest.raises(NotImplementedError, match="must implement train or pretrain"):
            NoOpSurrogate()

    def test_selects_single_shot_train_task(self):
        s = TrainOnlySurrogate()
        with patch.object(SurrogateBase, "_use_cached_models", return_value=False):
            task = s.select_training_task(self.candidate)
        assert task is s.train
        with pytest.raises(KeyError):
            task.binding("candidates")
        assert s.training_stage(task) == "trained"
        assert s.strategy.finetune_flags == []
        assert s.calls == []

    def test_selected_task_is_the_training_graph_node(self):
        s = TrainOnlySurrogate()
        wf = Workflow(SimpleModule())
        wf.patch(s)
        with patch.object(SurrogateBase, "_use_cached_models", return_value=False):
            task = s.select_training_task(self.candidate)
        with patch.object(s, "publish_trained_model"):
            session = next(wf.surrogate_execution.pending_training_sessions(self.candidate))
            with session.activate():
                assert wf.graph_context.mode == GraphMode.WORKFLOW_SURROGATE_TRAIN
                plan = wf.graph_context.training_plan

        assert plan is not None
        assert plan.train_submodule is task
        assert plan.trainer.inner is task
        assert plan.candidates == [self.candidate]
        assert plan.trainer in plan.graph
        assert s.calls == []

    def test_training_invocations_build_and_execute_distinct_plans(self):
        workflow = Workflow(SimpleModule())
        surrogate = MagicMock()
        first_task = TrainOnlySurrogate().train
        second_task = TrainOnlySurrogate().train
        first_candidate = self.candidate
        second_candidate = Candidate(
            self.candidate.space,
            torch.tensor([7.0]),
        )
        surrogate.generate_training_candidates.side_effect = [
            [first_candidate],
            [second_candidate],
        ]
        surrogate.training_stage.return_value = "trained"
        surrogate.select_training_task.side_effect = [first_task, second_task]
        workflow.patch(surrogate)

        executions = []

        for candidate in (first_candidate, second_candidate):
            session = next(workflow.surrogate_execution.pending_training_sessions(candidate))
            with session.activate() as candidates:
                executions.append((workflow.graph_context.training_plan, candidates))

        first_plan, first_candidates = executions[0]
        second_plan, second_candidates = executions[1]
        assert first_plan is not second_plan
        assert first_plan.train_submodule is first_task
        assert second_plan.train_submodule is second_task
        assert first_candidates == [first_candidate]
        assert second_candidates == [second_candidate]
        assert workflow.graph_context.mode == GraphMode.DEFAULT

    def test_module_surrogate_training_state(self):
        module = SimpleModule()
        target = MagicMock()
        target.contract = MagicMock(
            data_file=Workdir.register_file("simulation", "events.h5"),
            conditional_file=None,
        )
        module.surrogate = target
        target.select_training_task.return_value = task = TrainOnlySurrogate().train
        target.generate_training_candidates.return_value = [self.candidate]
        target.training_stage.return_value = "trained"
        workflow = Workflow(module)

        session = next(workflow.surrogate_execution.pending_training_sessions(self.candidate))
        with session.activate():
            assert workflow.graph_context.mode == GraphMode.MODULE_SURROGATE_TRAIN
            plan = workflow.graph_context.training_plan
            assert plan is not None
            assert plan.train_submodule is task
            assert plan.candidates == [self.candidate]
            assert plan.trainer in workflow.task_graph

    def test_two_phase_selects_pretrain_when_not_pretrained(self):
        s = PretrainFinetuneSurrogate()
        with patch.object(SurrogateBase, "_use_cached_models", return_value=False), \
             patch.object(SurrogateBase, "is_pretrained", new=False):
            task = s.select_training_task(self.candidate)
        assert task is s.pretrain
        with pytest.raises(KeyError):
            task.binding("candidates")
        assert s.training_stage(task) == "pretrained"
        assert s.strategy.finetune_flags == []
        assert s.generate_training_candidates(self.candidate, task) == [self.candidate]
        assert s.strategy.finetune_flags == [False]
        assert s.calls == []

    def test_two_phase_selects_finetune_when_not_finetuned(self):
        s = PretrainFinetuneSurrogate()
        with patch.object(SurrogateBase, "_use_cached_models", return_value=False), \
             patch.object(SurrogateBase, "is_pretrained", new=True), \
             patch.object(SurrogateBase, "is_finetuned", new=False):
            task = s.select_training_task(self.candidate)
        assert task is s.finetune
        with pytest.raises(AttributeError):
            task.candidates
        assert s.training_stage(task) == "finetuned"
        assert s.strategy.finetune_flags == []
        assert s.generate_training_candidates(self.candidate, task) == [self.candidate]
        assert s.strategy.finetune_flags == [True]
        assert s.calls == []

    def test_two_phase_reuses_finetuned_model_when_cached_and_in_trust_region(self):
        s = PretrainFinetuneSurrogate()
        s.strategy.is_in_trust_region = lambda c: True
        cfg = CacheConfig(cache_policy=CachePolicy.CACHE_EVERYTHING)
        with config.use(cache_config=cfg), \
             patch.object(SurrogateBase, "is_pretrained", new=True), \
             patch.object(SurrogateBase, "is_finetuned", new=True):
            d = s.select_training_task(self.candidate)
        assert d is None
        assert s.calls == []

    def test_two_phase_retrains_finetuned_model_when_retrain_forced(self):
        s = PretrainFinetuneSurrogate()
        s.strategy.is_in_trust_region = lambda c: True
        with patch.object(SurrogateBase, "_use_cached_models", return_value=False), \
             patch.object(SurrogateBase, "is_pretrained", new=True), \
             patch.object(SurrogateBase, "is_finetuned", new=True):
            task = s.select_training_task(self.candidate)
        assert task is s.finetune
        assert s.calls == []

    def test_two_phase_selects_finetune_when_finetuned_but_outside_trust_region(self):
        s = PretrainFinetuneSurrogate()
        s.strategy.is_in_trust_region = lambda c: False
        cfg = CacheConfig(cache_policy=CachePolicy.CACHE_EVERYTHING)
        with config.use(cache_config=cfg), \
             patch.object(SurrogateBase, "is_pretrained", new=True), \
             patch.object(SurrogateBase, "is_finetuned", new=True):
            task = s.select_training_task(self.candidate)
        assert task is s.finetune
        assert s.calls == []

    def test_selects_nothing_when_cached_and_in_trust_region(self):
        s = TrainOnlySurrogate()
        s.is_in_trust_region = lambda c: True
        cfg = CacheConfig(cache_policy=CachePolicy.CACHE_EVERYTHING)
        with config.use(cache_config=cfg):
            d = s.select_training_task(self.candidate)
        assert d is None
        assert s.calls == []

    def test_selects_nothing_without_cache_config_when_trained_and_in_trust_region(self):
        s = TrainOnlySurrogate()
        s.is_in_trust_region = lambda c: True
        d = s.select_training_task(self.candidate)
        assert d is None
        assert s.calls == []
