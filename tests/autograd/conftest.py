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

    def _task_resources(self, module):
        return {}


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


@pytest.fixture
def gradient_workflow_factory():
    def build(kind):
        class Source(Module):
            def __init__(self):
                super().__init__()
                self.out = Output(Workdir.register_file("reference", "source.h5"))

            def __call__(self, design, **kwargs):
                value = design["x"] ** 2
                HDF5File.write(self.out(), value)
                return {"direct": value} if kind == "shared" else {}

            def backward(self, design, grad_output=None):
                adjoint = HDF5File.read_backward_gradient(self.out())
                if kind == "shared":
                    adjoint = adjoint + grad_output["direct"]
                (design["x"] ** 2).backward(adjoint)

        class Branch(Module):
            def __init__(self, source, square):
                super().__init__()
                self.inp = Input(source)
                self.out = Output(Workdir.register_file("reference", "square.h5" if square else "scale.h5"))
                self.square = square

            def __call__(self, design, **kwargs):
                value = HDF5File.open(self.inp(), "cpu")
                HDF5File.write(self.out(), value ** 2 if self.square else 3 * value)

            def backward(self, design, grad_output=None):
                value = HDF5File.open(self.inp(), "cpu")
                output = value ** 2 if self.square else 3 * value
                output.backward(HDF5File.read_backward_gradient(self.out()))

        class Objective(Module):
            def __init__(self, left, right=None):
                super().__init__()
                self.left = Input(left)
                self.right = Input(right) if right is not None else None

            def value(self):
                left = HDF5File.open(self.left(), "cpu")
                return left + HDF5File.open(self.right(), "cpu") if self.right is not None else torch.sin(left)

            def __call__(self, design, **kwargs):
                return {"loss": self.value()}

            def backward(self, design, grad_output=None):
                self.value().backward(grad_output["loss"])

        source = Source()
        scaled = Branch(source.out.path, square=False)
        if kind == "diamond":
            squared = Branch(source.out.path, square=True)
            objective = Objective(scaled.out.path, squared.out.path)
            graph = source >> (scaled | squared) >> objective
            reference = lambda x: 3 * x ** 2 + x ** 4
            derivative = lambda x: 6 * x + 4 * x ** 3
        elif kind in ("chain", "shared"):
            objective = Objective(scaled.out.path)
            graph = source >> scaled >> objective
            reference = lambda x: torch.sin(3 * x ** 2) + (x ** 2 if kind == "shared" else 0)
            derivative = lambda x: 6 * x * torch.cos(3 * x ** 2) + (2 * x if kind == "shared" else 0)
        else:
            raise ValueError(f"Unknown analytic graph: {kind}")
        return Workflow(graph), reference, derivative

    return build
