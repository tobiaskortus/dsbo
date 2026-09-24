import torch
import pytest
import shutil
import tempfile
from pathlib import Path

from dsbo.modules import Module
from dsbo.utils.io import Input, Output
from dsbo.utils.io.workdir import Workdir
from dsbo.utils.io.storage.hdf5 import HDF5File
from dsbo.parameters import ParameterSpace, Vector, Candidate, CandidateBatch


class SimpleModule(Module):
    def __call__(self, design, **kwargs):
        return {"out": design["x"] * 2.0}


class AnotherModule(Module):
    def __call__(self, design, **kwargs):
        return {"out": design["x"] + 1.0}


class H5Producer(Module):
    def __init__(self):
        super().__init__()
        self.out = Output(Workdir.register_file("simulation", "events.h5"))

    def __call__(self, design, **kwargs):
        HDF5File.write(self.out(), design["x"].reshape(1))


class H5Consumer(Module):
    def __init__(self):
        super().__init__()
        self.inp = Input(Workdir.register_file("simulation", "events.h5"))

    def __call__(self, design, **kwargs):
        return {"loss": HDF5File.open(self.inp(), "cpu").sum()}


class H5InputModule(Module):
    def __init__(self):
        super().__init__()
        self.inp = Input(Workdir.register_file("simulation", "events.h5"))


@pytest.fixture
def simple_space():
    return ParameterSpace(x=Vector(0, 10, 1))


@pytest.fixture
def candidate(simple_space):
    return Candidate(simple_space, torch.tensor([5.0]))


@pytest.fixture
def candidate_batch(simple_space):
    t = torch.stack([torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])])
    return CandidateBatch([Candidate(simple_space, t[i]) for i in range(t.shape[0])])


@pytest.fixture
def results():
    return {"out": torch.tensor(10.0)}


@pytest.fixture
def results_with_tangent():
    import torch.autograd.forward_ad as fw
    primal = torch.tensor(10.0)
    tangent = torch.tensor(2.0)
    with fw.dual_level():
        yield {"out": fw.make_dual(primal, tangent)}


@pytest.fixture
def temp_dir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)
