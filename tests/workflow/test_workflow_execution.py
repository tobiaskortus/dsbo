import pytest
import torch
from unittest.mock import MagicMock, patch
from itertools import chain
from dsbo.parameters import ParameterSpace, Continuous, Candidate
from dsbo.workflow import (
    run_batch, run_batch_reverse_autograd, RunBatchAutogradFn,
    ResultsBatch, to_1d_tensor, candidates_from_tensor, Workflow,
)
from dsbo.modules import Module
from dsbo.modules.dsl import DependencyDict
from tests.workflow.conftest import (
    SimpleModule, BackwardModule, NoGradModule, FailingModule,
    FailingBackwardModule,
)


class TestTo1dTensor:
    def test_scalar_float(self):
        result = to_1d_tensor(5.0, torch.float32)
        assert result.shape == (1,)
        assert result.item() == 5.0

    def test_tensor_input(self):
        result = to_1d_tensor(torch.tensor(3.0), torch.float64)
        assert result.dtype == torch.float64
        assert result.shape == (1,)

    def test_1d_tensor_passthrough(self):
        result = to_1d_tensor(torch.tensor([1.0, 2.0]), torch.float32)
        assert result.shape == (2,)

    def test_raises_on_2d_tensor(self):
        with pytest.raises(ValueError, match="Flatten"):
            to_1d_tensor(torch.ones(2, 3), torch.float32)

    def test_tensor_with_dtype_conversion(self):
        t = torch.tensor([1.0], dtype=torch.float32)
        result = to_1d_tensor(t, torch.float64)
        assert result.dtype == torch.float64


class TestCandidatesFromTensor:
    def test_single_candidate(self, simple_space):
        t = torch.tensor([[5.0]])
        cands = candidates_from_tensor(t, simple_space)
        assert len(cands) == 1
        assert cands[0].tensor.item() == 5.0

    def test_multiple_candidates(self, simple_space):
        t = torch.tensor([[1.0], [2.0], [3.0]])
        cands = candidates_from_tensor(t, simple_space)
        assert len(cands) == 3


class TestRunRunLocal:
    def test_run_single_candidate(self, single_module_workflow, candidate):
        with torch.no_grad():
            result = single_module_workflow.run(candidate)
        assert torch.allclose(result["out"], torch.tensor(10.0))

    def test_run_with_catch(self, simple_space):
        wf = Workflow(FailingModule())
        cand = Candidate(simple_space, torch.tensor([3.0]))
        with torch.no_grad():
            result = wf.run(cand, catch=(ValueError,))
        assert result == {}

    def test_run_re_raises_uncaught(self, simple_space):
        wf = Workflow(FailingModule())
        cand = Candidate(simple_space, torch.tensor([3.0]))
        with torch.no_grad():
            with pytest.raises(ValueError):
                wf.run(cand, catch=(RuntimeError,))


class TestRunBatchLocal:
    def test_runs_single_candidate(self, single_module_workflow, simple_space):
        cand = Candidate(simple_space, torch.tensor([3.0]))
        results = run_batch(single_module_workflow, [cand, cand])
        assert len(results) == 2
        assert torch.allclose(results[0]["out"], torch.tensor(6.0))
        assert torch.allclose(results[1]["out"], torch.tensor(6.0))

    def test_runs_multiple_candidates(self, single_module_workflow, candidates):
        results = run_batch(single_module_workflow, candidates)
        assert len(results) == 3
        assert torch.allclose(results[0]["out"], torch.tensor(2.0))
        assert torch.allclose(results[1]["out"], torch.tensor(4.0))
        assert torch.allclose(results[2]["out"], torch.tensor(6.0))

    def test_catches_exceptions(self, simple_space):
        wf = Workflow(FailingModule())
        cand = Candidate(simple_space, torch.tensor([3.0]))
        results = run_batch(wf, [cand, cand], catch=(ValueError,))
        assert len(results) == 2
        assert results[0] == {}
        assert results[1] == {}

    def test_re_raises_uncaught_exception(self, simple_space):
        wf = Workflow(FailingModule())
        cand = Candidate(simple_space, torch.tensor([3.0]))
        with pytest.raises(ValueError, match="simulated failure"):
            run_batch(wf, [cand, cand], catch=(RuntimeError,))


class TestRunBatchReverseAutograd:
    def test_gradients_for_backward_module(self, simple_space):
        m = BackwardModule()
        wf = Workflow(m)
        x = torch.tensor([3.0], requires_grad=True)
        cand = Candidate(simple_space, x)
        grad_outputs = [{"out": torch.tensor(1.0)}]
        grads = run_batch_reverse_autograd(wf, [cand], grad_outputs)
        assert len(grads) == 1
        assert grads[0].shape == (1,)

    def test_gradients_for_no_grad_module(self, simple_space):
        m = NoGradModule()
        wf = Workflow(m)
        x = torch.tensor([3.0], requires_grad=True)
        cand = Candidate(simple_space, x)
        grad_outputs = [{"out": torch.tensor(1.0)}]
        grads = run_batch_reverse_autograd(wf, [cand], grad_outputs)
        assert len(grads) == 1
        assert grads[0].shape == (1,)

    def test_multiple_candidates(self, simple_space):
        m = BackwardModule()
        wf = Workflow(m)
        cands = [Candidate(simple_space, torch.tensor([1.0], requires_grad=True)),
                 Candidate(simple_space, torch.tensor([2.0], requires_grad=True))]
        grad_outputs = [{"out": torch.tensor(1.0)}, {"out": torch.tensor(1.0)}]
        grads = run_batch_reverse_autograd(wf, cands, grad_outputs)
        assert len(grads) == 2

    def test_catches_exceptions(self, simple_space):
        #TODO: While this is the currently correct behaviour, this doesnt make much sense
        m = FailingBackwardModule()
        wf = Workflow(m)
        x = torch.tensor([3.0], requires_grad=True)
        cand = Candidate(simple_space, x)
        grad_outputs = [{"out": torch.tensor(1.0)}]
        grads = run_batch_reverse_autograd(wf, [cand], grad_outputs, catch=(ValueError,))
        assert len(grads) == 1

    def test_re_raises_uncaught_exception(self, simple_space):
        m = FailingBackwardModule()
        wf = Workflow(m)
        x = torch.tensor([3.0], requires_grad=True)
        cand = Candidate(simple_space, x)
        grad_outputs = [{"out": torch.tensor(1.0)}]
        with pytest.raises(ValueError, match="simulated backward failure"):
            run_batch_reverse_autograd(wf, [cand], grad_outputs, catch=(RuntimeError,))

    def test_backward_produces_gradients(self, simple_space):
        m = BackwardModule()
        wf = Workflow(m)
        x = torch.tensor([[3.0]], requires_grad=True)
        vals, _, _, _ = RunBatchAutogradFn.apply(wf, x, simple_space, ())
        loss = vals.sum()
        loss.backward()
        assert x.grad is not None

    def test_backward_diamond(self, simple_space):
        order = []

        class DiamondA(Module):
            def __call__(self, design, **kwargs):
                return {"out": design["x"] * 2.0}
            def backward(self, design, grad_output=None):
                order.append("A")
                x = design["x"]
                out = x * 2.0
                grad = grad_output["out"] if grad_output is not None else torch.ones_like(out)
                out.backward(grad)

        class DiamondB(Module):
            def __call__(self, design, **kwargs):
                return {"out": design["x"] * 3.0}
            def backward(self, design, grad_output=None):
                order.append("B")
                x = design["x"]
                out = x * 3.0
                grad = grad_output["out"] if grad_output is not None else torch.ones_like(out)
                out.backward(grad)

        class DiamondC(Module):
            def __call__(self, design, **kwargs):
                return {"out": design["x"] * 4.0}
            def backward(self, design, grad_output=None):
                order.append("C")
                x = design["x"]
                out = x * 4.0
                grad = grad_output["out"] if grad_output is not None else torch.ones_like(out)
                out.backward(grad)

        class DiamondD(Module):
            def __call__(self, design, **kwargs):
                return {"out": design["x"] * 5.0}
            def backward(self, design, grad_output=None):
                order.append("D")
                x = design["x"]
                out = x * 5.0
                out.backward(grad_output["out"])

        a, b, c, d = DiamondA(), DiamondB(), DiamondC(), DiamondD()
        dd = DependencyDict({a: [b, c], b: [d], c: [d]})
        wf = Workflow(dd)

        x = torch.tensor([[3.0]], requires_grad=True)
        vals, _, _, _ = RunBatchAutogradFn.apply(wf, x, simple_space, ())
        loss = vals.sum()
        loss.backward()

        assert order[:1] == ["D"]
        assert set(order[1:3]) == {"B", "C"}
        assert order[3:] == ["A"]

        expected_grad = 5.0 + 3.0 + 4.0 + 2.0
        assert x.grad is not None
        assert abs(x.grad.item() - expected_grad) < 1e-5


class TestWorkflowRunBatchNoAutograd:
    def test_executes_single_candidate(self, single_module_workflow, simple_space):
        cand = Candidate(simple_space, torch.tensor([3.0]))
        with torch.no_grad():
            results = single_module_workflow.run_batch([cand])
        assert torch.allclose(results[0]["out"], torch.tensor(6.0))

    def test_executes_multiple_candidates(self, single_module_workflow, candidates):
        with torch.no_grad():
            results = single_module_workflow.run_batch(candidates)
        assert len(results) == 3
        assert torch.allclose(results[0]["out"], torch.tensor(2.0))
        assert torch.allclose(results[1]["out"], torch.tensor(4.0))
        assert torch.allclose(results[2]["out"], torch.tensor(6.0))

    def test_executes_chain(self, chain_workflow, candidates):
        with torch.no_grad():
            results = chain_workflow.run_batch(candidates)
        assert len(results) == 3
        assert torch.allclose(results[0]["out"], torch.tensor(2.0))


class TestWorkflowRunBatchCacheIntegration:
    def test_all_hits_no_run_batch_called(self, workflow_with_mock_cache, candidates, mock_run_batch, mock_cache):
        mock_cache.match_partial.return_value = (
            [{"out": torch.tensor(10.0)}, {"out": torch.tensor(20.0)}, {"out": torch.tensor(30.0)}],
            [],
            [],
        )
        results = workflow_with_mock_cache.run_batch(candidates)
        mock_cache.match_partial.assert_called_once_with(candidates)
        mock_run_batch.assert_not_called()
        assert len(results) == 3
        assert torch.allclose(results[0]["out"], torch.tensor(10.0))
        assert torch.allclose(results[1]["out"], torch.tensor(20.0))
        assert torch.allclose(results[2]["out"], torch.tensor(30.0))

    def test_all_misses_run_batch_called(self, workflow_with_mock_cache, candidates, mock_run_batch, mock_cache):
        mock_cache.match_partial.return_value = (
            [None, None, None],
            candidates,
            [0, 1, 2],
        )
        mock_cache.save_all.return_value = None

        results = workflow_with_mock_cache.run_batch(candidates)

        mock_cache.match_partial.assert_called_once_with(candidates)
        mock_run_batch.assert_called_once()
        mock_cache.save_all.assert_called_once()
        assert len(results) == 3

    def test_partial_hits_misses_merged(self, workflow_with_mock_cache, candidates, mock_cache, mock_run_batch):
        miss_candidates = [candidates[2]]
        mock_cache.match_partial.return_value = (
            [{"out": torch.tensor(10.0)}, {"out": torch.tensor(20.0)}, None],
            miss_candidates,
            [2],
        )
        mock_run_batch.side_effect = lambda wf, misses, catch: [
            {"out": torch.tensor(30.0)} for _ in misses
        ]
        mock_cache.save_all.return_value = None

        results = workflow_with_mock_cache.run_batch(candidates)

        mock_cache.match_partial.assert_called_once_with(candidates)
        mock_run_batch.assert_called_once()
        mock_cache.save_all.assert_called_once_with(miss_candidates, [{"out": torch.tensor(30.0)}])
        assert len(results) == 3
        assert torch.allclose(results[0]["out"], torch.tensor(10.0))
        assert torch.allclose(results[1]["out"], torch.tensor(20.0))
        assert torch.allclose(results[2]["out"], torch.tensor(30.0))

    def test_cache_disabled_no_match_partial(self, workflow_with_mock_cache, candidates, mock_cache, mock_run_batch):
        mock_cache.config.should_cache.return_value = False

        results = workflow_with_mock_cache.run_batch(candidates)

        mock_cache.match_partial.assert_not_called()
        mock_run_batch.assert_called_once()
        assert len(results) == 3

    def test_no_cache_available(self, single_module_workflow, candidates, mock_run_batch):
        with patch.object(single_module_workflow, "_get_cache", return_value=None):
            results = single_module_workflow.run_batch(candidates)

        mock_run_batch.assert_called_once()
        assert len(results) == 3

    def test_result_reassembly_after_save(self, workflow_with_mock_cache, candidates, mock_cache, mock_run_batch):
        miss_candidates = [candidates[0], candidates[2]]
        mock_cache.match_partial.return_value = (
            [None, {"out": torch.tensor(20.0)}, None],
            miss_candidates,
            [0, 2],
        )
        mock_run_batch.side_effect = lambda wf, misses, catch: [
            {"out": torch.tensor(10.0)} for _ in misses
        ]
        mock_cache.save_all.return_value = None

        results = workflow_with_mock_cache.run_batch(candidates)

        assert len(results) == 3
        assert torch.allclose(results[0]["out"], torch.tensor(10.0))
        assert torch.allclose(results[1]["out"], torch.tensor(20.0))
        assert torch.allclose(results[2]["out"], torch.tensor(10.0))


class TestGetCache:
    def test_returns_none_when_no_cache_config(self, single_module_workflow):
        with patch("dsbo.workflow.config.get", return_value=None):
            cache = single_module_workflow._get_cache()
            assert cache is None

    def test_returns_cache_when_config_exists(self, single_module_workflow):
        from dsbo.utils.cache import CacheConfig
        config = CacheConfig()
        with patch("dsbo.workflow.config.get", return_value=config):
            with patch("dsbo.workflow.get_workdir") as mock_wd:
                mock_wd.return_value.root = "/tmp/workdir"
                cache = single_module_workflow._get_cache()
                assert cache is not None
