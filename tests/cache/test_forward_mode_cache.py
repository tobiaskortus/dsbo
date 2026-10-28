import pytest
import torch
import torch.autograd.forward_ad as fw

from dsbo.config import config
from dsbo.modules import Module
from dsbo.parameters import Candidate, Continuous, ParameterSpace
from dsbo.workflow import Workflow
from dsbo.utils.cache import CacheConfig
from dsbo.utils.io.workdir import Workdir
from dsbo.utils.random import get_random_engine


@pytest.fixture
def cached_directional_workflow(tmp_path):
    class DirectionalModule(Module):
        def __init__(self):
            super().__init__()
            self.executions = 0

        def __call__(self, design, **kwargs):
            self.executions += 1
            return {"out": design["x"] ** 2 + 3 * design["y"]}

        def backward(self, design, grad_output=None, **kwargs) -> None:
            out = design["x"] ** 2 + 3 * design["y"]
            out.backward(grad_output["out"])

    module = DirectionalModule()
    cached_workflow = Workflow(module)
    space = ParameterSpace(x=Continuous(0, 10), y=Continuous(0, 10))
    candidate = Candidate(space, torch.tensor([2.0, 4.0]))
    # The runner seeds modules before execution; seed now to keep the cache identity stable.
    get_random_engine().seed_all(cached_workflow)
    with Workdir.create(str(tmp_path), keep=True, error_if_exists=False), config.use(
        cache_config=CacheConfig(base_dir=str(tmp_path))
    ):
        yield cached_workflow, module, candidate


def test_forward_mode_cache_isolates_directions_and_reuses_hits(cached_directional_workflow):
    cached_workflow, module, candidate = cached_directional_workflow
    directions = [([1.0, 0.0], 4.0), ([0.0, 1.0], 3.0), ([2.0, 0.0], 8.0), ([0.0, 0.0], 0.0)]
    with fw.dual_level():
        for executions, (tangent, expected) in enumerate(directions, start=1):
            # Running twice and expecting the number of executions (module.executions) to remain the same
            for _ in range(2):
                dual_candidate = candidate.set_tangent(torch.tensor(tangent))
                result = cached_workflow.run(dual_candidate, use_surrogates=False)["out"]
                primal, result_tangent = fw.unpack_dual(result)
                torch.testing.assert_close(primal, torch.tensor(16.0))
                torch.testing.assert_close(result_tangent, torch.tensor(expected))
                assert module.executions == executions


def test_forward_mode_cache_reuses_hits_across_dual_levels(cached_directional_workflow):
    cached_workflow, module, candidate = cached_directional_workflow
    for _ in range(3):
        with fw.dual_level():
            dual_candidate = candidate.set_tangent(torch.tensor([1.0, 0.0]))
            result = cached_workflow.run(dual_candidate, use_surrogates=False)["out"]
            primal, tangent = fw.unpack_dual(result)
            torch.testing.assert_close(primal, torch.tensor(16.0))
            torch.testing.assert_close(tangent, torch.tensor(4.0))
            assert module.executions == 1


@pytest.mark.parametrize("ordinary_first", [False, True], ids=["forward-first", "ordinary-first"])
@pytest.mark.parametrize("grad_enabled", [False, True], ids=["no-grad", "reverse-autograd"])
def test_forward_mode_cache_isolated_from_ordinary_evaluation(
    cached_directional_workflow, ordinary_first, grad_enabled
):
    cached_workflow, module, candidate = cached_directional_workflow
    candidate.requires_grad_(grad_enabled)

    def run_primal():
        with torch.set_grad_enabled(grad_enabled):
            result = cached_workflow.run(candidate, use_surrogates=False)["out"]
        assert fw.unpack_dual(result).tangent is None
        torch.testing.assert_close(result, torch.tensor(16.0))
        return result

    if ordinary_first:
        run_primal()
        assert module.executions == 1

    with fw.dual_level():
        dual_candidate = candidate.set_tangent(torch.tensor([1.0, 0.0]))
        result = cached_workflow.run(dual_candidate, use_surrogates=False)["out"]
        primal, tangent = fw.unpack_dual(result)
        torch.testing.assert_close(primal, torch.tensor(16.0))
        torch.testing.assert_close(tangent, torch.tensor(4.0))
        assert module.executions == (2 if ordinary_first else 1)

    run_primal()
    assert module.executions == 2
    result = run_primal()
    assert module.executions == 2
    if grad_enabled:
        result.backward()
        torch.testing.assert_close(candidate.grad, torch.tensor([4.0, 3.0]))
        assert module.executions == 2
