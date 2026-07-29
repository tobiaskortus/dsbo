import io
import os
import pytest
import sys
import torch
import tempfile
from unittest.mock import MagicMock, patch
from uuid import UUID

from dsbo.parameters import ParameterSpace, Continuous, Candidate
from dsbo.modules import Module
from dsbo.modules.dsl import DependencyDict
from dsbo.workflow import Workflow
from dsbo.worker.dask import DaskWorkerBase
from dsbo.utils.io import Workdir, Input, Output
from dsbo.utils.io.autograd import HDF5File
from dask_jobqueue.local import LocalCluster


# ────────── Helper modules for testing ──────────


class MockDaskWorker(DaskWorkerBase):
    def __init__(self):
        self.client = MagicMock()
        self.cluster = MagicMock()
        self.task_counter = 0
        self.scheduled = []
        self.backward_scheduled = []
        self._gather_results = None
        self._gather_forward_results = True  # if True, gather returns dicts

    def schedule(self, module, design, dependencies=(), priority=None, file_overlays=None):
        tid = UUID(int=self.task_counter, version=4)
        self.task_counter += 1
        delayed = MagicMock()
        delayed.key = f"mock-{tid}"
        self.scheduled.append((tid, delayed, module, priority))
        return tid, delayed

    def schedule_backward(self, module, candidate, grad_output=None,
                          dependencies=(), priority=None):
        tid = UUID(int=self.task_counter, version=4)
        self.task_counter += 1
        delayed = MagicMock()
        delayed.key = f"mock-bwd-{tid}"
        self.backward_scheduled.append((tid, delayed, module, priority))
        return tid, delayed

    def compute(self, objs, **kwargs):
        return objs

    def gather(self, futures, catch=()):
        if self._gather_results is not None:
            return self._gather_results
        # Default: return a dict with "out" for each scheduled task
        if self._gather_forward_results:
            return [(tid, {"out": torch.tensor(10.0)})
                    for tid, _, _, _ in self.scheduled]
        else:
            return [(tid, torch.tensor([10.0]))
                    for tid, _, _, _ in self.backward_scheduled]

    def set_gather_for_scheduled(self, results):
        """Set gather to return results keyed by the tid of each scheduled task.
        `results` should be a list of dicts (one per candidate) to assign per-task."""
        self._gather_results = [
            (tid, r) for tid, _, _, _ in self.scheduled
            for r in (results if isinstance(results, list) else [results])
        ]


def make_mock_worker(**attrs):
    worker = MockDaskWorker()
    for k, v in attrs.items():
        setattr(worker, k, v)
    return worker


class LocalDaskWorker(DaskWorkerBase):
    def __init__(self):
        self.cluster = LocalCluster(cores=1, memory="500MB")
        self.cluster.scale(2)
        super().__init__(self.cluster)

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


class DataProducer(Module):
    def __init__(self):
        super().__init__()
        self.output = Output(Workdir.register_file("simulation", "events.h5"))

    def __call__(self, design, **kwargs):
        HDF5File.write(self.output(), design["x"].reshape(1))


class FileTrainer(Module):
    def __init__(self):
        super().__init__()
        self.input = Input(Workdir.register_file("simulation", "events.h5"))
        self.model = Output(Workdir.register_file("reconstruction", "model.pt"))

    def __call__(self, design, **kwargs):
        loss = HDF5File.open(self.input(), "cpu").sum()
        torch.save(loss, self.model())
        return {"loss": loss}


class DataConsumer(Module):
    def __init__(self):
        super().__init__()
        self.input = Input(Workdir.register_file("simulation", "events.h5"))


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

@pytest.fixture(scope="function")
def local_worker():
    with LocalDaskWorker() as w:
        yield w

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


@pytest.fixture(autouse=True)
def _temp_workdir():
    """Wrap each test in a temporary workdir to isolate file operations (e.g. logs)."""
    with tempfile.TemporaryDirectory() as tmpdir, Workdir.create(tmpdir, keep=True, error_if_exists=False):
        yield


@pytest.fixture(autouse=True)
def _suppress_tqdm():
    """Redirect stderr to devnull to suppress tqdm progress bar output during tests."""
    with open(os.devnull, "w") as null, patch.object(sys, "stderr", null):
        yield
