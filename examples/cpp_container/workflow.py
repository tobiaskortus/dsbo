"""Python workflow orchestration around a Python-free native step.

Run with --image dsbo-native-example:latest, or use --executable with the local
CMake example binary. See docs/guide/native_io.md for building the image.
"""

import argparse
import subprocess
from pathlib import Path

import torch

from dsbo.container import ContainerConfig, ContainerWrapperBase, ImageRef
from dsbo.modules import Module
from dsbo.parameters import Candidate, Continuous, ParameterSpace
from dsbo.utils.io import Input, Output, Workdir
from dsbo.utils.io.storage import HDF5File
from dsbo.workflow import Workflow


class Source(Module):
    def __init__(self):
        super().__init__()
        self.output = Output(Workdir.register_file("native-example", "input.h5"))

    def __call__(self, design, **kwargs):
        HDF5File.write(self.output(), design["x"].reshape(1))

    def backward(self, design, grad_output=None):
        design["x"].reshape(1).backward(HDF5File.read_backward_gradient(self.output()))


class NativeDouble(Module):
    def __init__(self, source, executable: str | None):
        super().__init__()
        self.input = Input(source)
        self.output = Output(Workdir.register_file("native-example", "output.h5"))
        self.executable = executable

    def _launch(self, *arguments):
        subprocess.run([self.executable, *arguments], check=True)

    def __call__(self, design, **kwargs):
        self._launch("forward", self.input(), self.output())

    def backward(self, design, grad_output=None):
        self._launch("backward", self.output.gradient(), self.input.gradient())


class ContainerNativeDouble(NativeDouble, ContainerWrapperBase):
    def __init__(self, source, root: str, image: str):
        NativeDouble.__init__(self, source, executable=None)
        ContainerWrapperBase.__init__(
            self,
            ImageRef(uri=image),
            config=ContainerConfig(bind_mounts=((root, root),), workdir=root),
        )

    def _launch(self, *arguments):
        # The image entrypoint is the C++ executable. Preserve resolved host
        # paths with a same-path bind mount; do not infer gradient names.
        self.run_container(list(arguments))


class Objective(Module):
    def __init__(self, source):
        super().__init__()
        self.input = Input(source)

    def __call__(self, design, **kwargs):
        return {"loss": HDF5File.open(self.input(), "cpu").sum()}

    def backward(self, design, grad_output=None):
        HDF5File.open(self.input(), "cpu").sum().backward(grad_output["loss"])


def run(root: Path, image="dsbo-native-example:latest", executable=None):
    root = root.resolve()
    if executable:
        executable = str(Path(executable).resolve())
    space = ParameterSpace(x=Continuous(-10, 10))
    candidate = Candidate.from_values(space, {"x": 3.0}, requires_grad=True)
    with Workdir.create(str(root), error_if_exists=False, keep=True):
        source = Source()
        native = NativeDouble(source.output.path, executable) if executable else ContainerNativeDouble(source.output.path, str(root), image)
        objective = Objective(native.output.path)
        result = Workflow(source >> native >> objective).run(candidate)
        result["loss"].backward()
    torch.testing.assert_close(result["loss"].detach(), torch.tensor(6.0))
    torch.testing.assert_close(candidate.tensor.grad, torch.full_like(candidate.tensor, 2.0))
    print("Native workflow: loss=6, d(loss)/dx=2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default=Path("build/native-workflow"))
    parser.add_argument("--image", default="dsbo-native-example:latest")
    parser.add_argument("--executable", help="Run a locally compiled executable instead of a container")
    args = parser.parse_args()
    run(args.workdir, args.image, args.executable)
