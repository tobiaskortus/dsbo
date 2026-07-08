import pytest
import torch
from unittest.mock import MagicMock, patch
from dsbo.parameters import ParameterSpace, Continuous, Candidate
from dsbo.workflow import Workflow
from tests.workflow.conftest import SimpleModule, SurrogateModule, FailingModule


class TestModuleSurrogateChecks:
    def test_finds_modules_with_surrogates_empty_when_none(self, single_module_workflow):
        assert single_module_workflow.module_surrogates == {}

    def test_finds_modules_with_surrogates(self):
        m = SimpleModule()
        m.surrogate = MagicMock()
        wf = Workflow(m)
        surrogates = wf.module_surrogates
        assert len(surrogates) == 1
        assert m in surrogates

    def test_uses_module_surrogates_true(self):
        m = SimpleModule()
        m.surrogate = MagicMock()
        wf = Workflow(m)
        assert wf.uses_module_surrogates

    def test_uses_module_surrogates_false(self, single_module_workflow):
        assert not single_module_workflow.uses_module_surrogates


class TestEnsureSurrogatesTrained:
    def test_workflow_surrogate_trains_when_not_ready(self):
        m = SimpleModule()
        wf = Workflow(m)
        surrogate = MagicMock()
        surrogate.is_in_trust_region.return_value = False
        surrogate.surrogate = None
        wf._surrogate = surrogate

        cand = Candidate(ParameterSpace(x=Continuous(0, 10, 1)), torch.tensor([3.0]))
        wf._ensure_surrogates_trained(cand)
        surrogate.fit.assert_called_once()

    def test_workflow_surrogate_skips_when_ready(self):
        m = SimpleModule()
        wf = Workflow(m)
        surrogate = MagicMock()
        surrogate.is_in_trust_region.return_value = True
        surrogate.surrogate = MagicMock()
        wf._surrogate = surrogate

        cand = Candidate(ParameterSpace(x=Continuous(0, 10, 1)), torch.tensor([3.0]))
        wf._ensure_surrogates_trained(cand)
        surrogate.fit.assert_not_called()

    def test_module_surrogate_trains_when_not_ready(self):
        m = SimpleModule()
        m.surrogate = MagicMock()
        m.surrogate.is_in_trust_region.return_value = False
        m.surrogate.is_available = False
        wf = Workflow(m)

        cand = Candidate(ParameterSpace(x=Continuous(0, 10, 1)), torch.tensor([3.0]))
        with patch("dsbo.workflow.Workdir") as mock_wd:
            with patch("dsbo.workflow.config") as mock_config:
                wf._ensure_surrogates_trained(cand)
                m.surrogate.fit.assert_called_once()

    def test_module_surrogate_skips_when_ready(self):
        m = SimpleModule()
        m.surrogate = MagicMock()
        m.surrogate.is_in_trust_region.return_value = True
        m.surrogate.is_available = True
        wf = Workflow(m)

        cand = Candidate(ParameterSpace(x=Continuous(0, 10, 1)), torch.tensor([3.0]))
        wf._ensure_surrogates_trained(cand)
        m.surrogate.fit.assert_not_called()


class TestRunWorkflowWithSurrogate:
    def test_workflow_surrogate(self, single_module_workflow, candidate):
        surrogate = MagicMock()
        surrogate.side_effect = lambda c: {"out": torch.tensor(42.0)}
        single_module_workflow.patch(surrogate)
        result = single_module_workflow.run(candidate)
        assert torch.allclose(result["out"], torch.tensor(42.0))

    def test_workflow_surrogate_batch_not_implemented(self, single_module_workflow, candidates):
        surrogate = MagicMock()
        surrogate._is_pretrained = True
        surrogate.side_effect = lambda c: {"out": torch.tensor(42.0)}
        single_module_workflow.patch(surrogate)

        with pytest.raises(NotImplementedError, match="not supported"):
            single_module_workflow.run_batch(candidates)

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