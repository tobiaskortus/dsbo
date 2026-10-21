from dask_jobqueue.local import LocalCluster
import torch
import pytest

from dsbo.utils.io.storage.hdf5 import HDF5File
from dsbo.worker.dask import DaskWorkerBase
from dsbo.workflow import Workflow
from dsbo.utils.io.workdir import Workdir
from dsbo.modules import Module, Input, Output
from dsbo.parameters import Continuous, ParameterSpace, UniformSampler

class LocalDaskWorker(DaskWorkerBase):
    def __init__(self):
        self.cluster = LocalCluster(cores=1, memory="2GB")
        self.cluster.scale(2)
        super().__init__(self.cluster)


@pytest.fixture(scope="session", autouse=True)
def workflow():
    class ModuleA(Module):
        def __init__(self):
            super().__init__()
            self.out = Output(Workdir.register_file("a", "a_out.h5"))
        def __call__(self, design, **kwargs):
            out = design["x"] * design["y"]
            HDF5File.write(self.out(), out)
        def backward(self, design, grad_output=None):
            grad_out = HDF5File.read_backward_gradient(self.out())
            out = design["x"] * design["y"]
            out.backward(grad_out)

    class ModuleB(Module):
        def __init__(self, a_out_path):
            super().__init__()
            self.inp = Input(a_out_path)
            self.out = Output(Workdir.register_file("b", "b_out.h5"))

        def __call__(self, design, **kwargs):
            x = HDF5File.open(self.inp(), "cpu")
            y = torch.sin(x)
            HDF5File.write(self.out(), y)

        def backward(self, design, grad_output=None):
            grad_out = HDF5File.read_backward_gradient(self.out())
            x = HDF5File.open(self.inp(), "cpu")
            y = torch.sin(x)
            y.backward(grad_out)

    class ModuleC(Module):
        def __init__(self, a_out_path):
            super().__init__()
            self.inp = Input(a_out_path)
            self.out = Output(Workdir.register_file("c", "c_out.h5"))

        def __call__(self, design, **kwargs):
            x = HDF5File.open(self.inp(), "cpu")
            y = torch.exp(x)
            HDF5File.write(self.out(), y)

        def backward(self, design, grad_output=None):
            grad_out = HDF5File.read_backward_gradient(self.out())
            x = HDF5File.open(self.inp(), "cpu")
            y = torch.exp(x)
            y.backward(grad_out)

    class ModuleD(Module):
        def __init__(self, b_out_path, c_out_path):
            super().__init__()
            self.inp_b = Input(b_out_path)
            self.inp_c = Input(c_out_path)

        def __call__(self, design, **kwargs):
            b = HDF5File.open(self.inp_b(), "cpu")
            c = HDF5File.open(self.inp_c(), "cpu")
            return {"final": (b * c).sum()}

        def backward(self, design, grad_output=None):
            if grad_output is None:
                return
            b = HDF5File.open(self.inp_b(), "cpu")
            c = HDF5File.open(self.inp_c(), "cpu")
            outputs = {"final": (b * c).sum()}
            tensors = [outputs[k] for k in grad_output]
            grads = [grad_output[k] for k in grad_output]
            torch.autograd.backward(tensors, grads)

    a = ModuleA()
    b = ModuleB(a.out.path)
    c = ModuleC(a.out.path)
    d = ModuleD(b.out.path, c.out.path)

    return Workflow(a >> (b | c) >> d)


@pytest.fixture(scope="function")
def space():
    return ParameterSpace(x=Continuous(-2.0, 2.0), y=Continuous(-2.0, 2.0))


@pytest.fixture(scope="function")
def sampler():
    return UniformSampler()
