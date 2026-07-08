import pytest
import torch
from unittest.mock import MagicMock, patch

from dsbo.parameters import ParameterSpace, Continuous, Candidate
from dsbo.modules import Module
from dsbo.modules.dsl import DependencyDict
from dsbo.workflow import Workflow


# ────────── Helper modules for testing ──────────

class SimpleModule(Module):
    def __init__(self):
        super().__init__()
    def __call__(self, design, **kwargs):
        return {"out": design["x"] * 2.0}

class MultiOutputModule(Module):
    def __init__(self):
        super().__init__()
    def __call__(self, design, **kwargs):
        return {"a": design["x"], "b": design["x"] * 2.0}

class BackwardModule(Module):
    def __init__(self):
        super().__init__()
    def __call__(self, design, **kwargs):
        return {"out": design["x"] * 2.0}
    def backward(self, design, grad_output=None):
        pass

class FailingBackwardModule(Module):
    def __init__(self):
        super().__init__()
    def __call__(self, design, **kwargs):
        return {"out": design["x"]}
    def backward(self, design, grad_output=None):
        raise ValueError("simulated backward failure")

class NoGradModule(Module):
    def __init__(self):
        super().__init__()
        self._no_grad_op = True
    def __call__(self, design, **kwargs):
        return {"out": design["x"]}

class FailingModule(Module):
    def __init__(self):
        super().__init__()
    def __call__(self, design, **kwargs):
        raise ValueError("simulated failure")

class SurrogateModule(Module):
    """A real Module used as a module surrogate in tests."""
    def __init__(self):
        super().__init__()
        self._alias = None
    def __call__(self, design, **kwargs):
        return {"out": torch.tensor(99.0)}
    def set_alias(self, name):
        self._alias = name
    @property
    def name(self):
        return self._alias or "SurrogateModule"
    def is_in_trust_region(self, candidate):
        return True
    @property
    def is_available(self):
        return True
    def fit(self, candidate, workflow_callback):
        pass


# ────────── Fixtures ──────────

@pytest.fixture
def simple_space():
    return ParameterSpace(x=Continuous(0, 10, 1))

@pytest.fixture
def candidate(simple_space):
    return Candidate(simple_space, torch.tensor([5.0]))

@pytest.fixture
def candidates(simple_space):
    return [
        Candidate(simple_space, torch.tensor([1.0])),
        Candidate(simple_space, torch.tensor([2.0])),
        Candidate(simple_space, torch.tensor([3.0])),
    ]

@pytest.fixture
def single_module_workflow():
    return Workflow(SimpleModule())

@pytest.fixture
def chain_workflow():
    m1 = SimpleModule()
    m2 = SimpleModule()
    return Workflow(m1 >> m2)

@pytest.fixture
def dag_workflow():
    a = SimpleModule()
    b = SimpleModule()
    c = SimpleModule()
    dd = DependencyDict({a: [], b: [a], c: [a]})
    return Workflow(dd)

@pytest.fixture
def mock_cache():
    instance = MagicMock()
    instance.config = MagicMock()
    instance.config.should_cache.return_value = True
    instance.match_partial.return_value = ([None, None, None], [], [])
    yield instance

@pytest.fixture
def mock_run_batch():
    with patch("dsbo.workflow.run_batch") as mock:
        mock.side_effect = lambda wf, misses, catch: [
            {"out": torch.tensor(float(c.tensor.item() * 10.0))} for c in misses
        ]
        yield mock

@pytest.fixture
def workflow_with_mock_cache(single_module_workflow, mock_cache):
    with patch.object(single_module_workflow, "_get_cache", return_value=mock_cache):
        yield single_module_workflow
