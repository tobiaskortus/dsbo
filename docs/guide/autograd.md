# Differentiable Workflows

dsbo workflows support reverse- and forward-mode automatic differentiation using PyTorch's standard interfaces. Each module supplies its local derivatives, while the runtime propagates them between tasks through the declared-file interface. This connects workflow results to the candidate's parameter tensor, allowing computations in separate processes or external programs to participate in the same optimization loop.

## Tracking Parameter Gradients

Enable gradient tracking when constructing or sampling a candidate. Its tensor can then be passed to a standard PyTorch optimizer:

```python
import torch

from dsbo.parameters import Candidate, Continuous, ParameterSpace

space = ParameterSpace(scale=Continuous(0.5, 3.0))
candidate = Candidate.from_values(space, {"scale": 2.0}, requires_grad=True)

loss = (candidate.values["scale"] - 1.0).square()
loss.backward()
candidate.grad  # tensor([2.])
```

Decoded continuous values retain their connection to the candidate tensor. Scalar indexing, such as `candidate["scale"]`, converts the value to a Python number and discards this connection. Enabling gradients does not make integer rounding or categorical decoding differentiable.

## File-Based Autograd

A normal workflow forward pass does not record a reverse-mode PyTorch graph across its tasks. Files carry the values between modules; the backward pass uses corresponding gradient files to carry their adjoints in the opposite direction.

The provided `HDF5File` backend implements this channel. `write()` stores a tensor's values. When `open()` is called with gradient tracking enabled, it returns a tensor whose backward hook records a gradient contribution for that file:

```python
from dsbo.utils.io import Workdir
from dsbo.utils.io.storage import HDF5File

with Workdir.create("./file-gradient"):
    path = Workdir.register_file("data", "values.h5")()
    HDF5File.write(path, torch.tensor([1.0, 2.0]))
    values = HDF5File.open(path, device="cpu")

    loss = (values * torch.tensor([2.0, 3.0])).sum()
    loss.backward()

    gradient = HDF5File.read_backward_gradient(path)  # tensor([2., 3.])
```

Within a workflow pass, consumers write private gradient contributions. `read_backward_gradient()` resolves their sum so the producing module can use one incoming adjoint. The runtime scopes gradient state to each pass; modules do not need to share their forward tensors in memory.

## Implementing a Backward Pass

A producer recomputes its local expression during `backward()` and applies the incoming file gradient. This example writes a response controlled by one continuous parameter:

```python
from dsbo.modules import Module
from dsbo.utils.io import Input, Output

class ScaledResponse(Module):
    def __init__(self):
        super().__init__()
        self.output = Output(Workdir.register_file("response", "values.h5"))

    def response(self, design):
        return design["scale"] * torch.arange(1, 5, dtype=torch.float32)

    def __call__(self, design, **kwargs):
        HDF5File.write(self.output(), self.response(design))

    def backward(self, design, grad_output=None):
        with torch.enable_grad():
            response = self.response(design)
        response.backward(HDF5File.read_backward_gradient(self.output()))
```

The terminal module computes the objective. Its backward pass reopens the input and seeds the local derivative using the gradient of its returned result:

```python
class ResponseLoss(Module):
    def __init__(self, input_path):
        super().__init__()
        self.input = Input(input_path)

    def __call__(self, design, **kwargs):
        values = HDF5File.open(self.input(), device="cpu")
        return {"loss": values.square().mean()}

    def backward(self, design, grad_output=None):
        with torch.enable_grad():
            values = HDF5File.open(self.input(), device="cpu")
            loss = values.square().mean()
        loss.backward(grad_output["loss"])
```

`grad_output` contains derivatives with respect to returned workflow results, not the gradients of file contents. A module that contributes through both a returned result and a file must account for both paths.

### External Programs

An external program can participate in reverse-mode differentiation by exposing a derivative routine that the module calls from `backward()`. For a program computing `y = f(x)`, this routine receives the incoming output gradient `v = dL/dy` and computes the vector-Jacobian product `J_f(x).T @ v`, giving `dL/dx`. The routine may use the program's own automatic differentiation, an analytical derivative, or an adjoint solver; a full Jacobian need not be constructed.

The module connects this routine to the workflow in three steps:

1. Its forward call runs the program and writes the output through the declared-file interface. Any checkpoints required by the derivative routine are also saved to files, rather than retained only in module attributes.
2. Its `backward()` reads `dL/dy` with `HDF5File.read_backward_gradient(self.output())` and passes it, together with the design and any checkpoints, to the external derivative routine.
3. The returned derivatives are propagated into the corresponding design tensors with `torch.autograd.backward()`. For inputs supplied by upstream files, reopening those files with `HDF5File.open()` and calling `backward()` on the resulting tensors records gradient contributions for their producers.

For example, an external version of `ScaledResponse` can implement its backward method as follows. Here, `run_external_vjp()` is an application-defined adapter that launches the derivative program, transfers the incoming gradient, and reads back the derivative with respect to `scale`:

```python
def backward(self, design, grad_output=None):
    output_gradient = HDF5File.read_backward_gradient(self.output())
    scale_gradient = self.run_external_vjp(
        design["scale"].detach(), output_gradient
    )
    scale_gradient = torch.as_tensor(
        scale_gradient,
        dtype=design["scale"].dtype,
        device=design["scale"].device,
    ).reshape_as(design["scale"])
    torch.autograd.backward(design["scale"], scale_gradient)
```

For the response `scale * [1, 2, 3, 4]`, the external routine returns `(output_gradient * [1, 2, 3, 4]).sum()`. This is the derivative of the final objective with respect to `scale`, not merely the derivative of the response. The adapter must wait for the derivative program to finish before returning; the workflow then continues the backward pass through upstream tasks.

## Running a Differentiable Workflow

Construct a fresh candidate and connect the producer to the objective. Calling `backward()` on the result invokes the module backward methods in reverse task order and accumulates the design gradient:

```python
from dsbo.workflow import Workflow

candidate = Candidate.from_values(space, {"scale": 2.0}, requires_grad=True)
response = ScaledResponse()
workflow = Workflow(response >> ResponseLoss(response.output.path))

with Workdir.create("./workflow-gradient"):
    result = workflow.run(candidate)
    result["loss"].backward()

candidate.grad  # tensor([30.])
```

The objective is the mean of the squared response values, so its derivative is `15 * scale`. The computed gradient is therefore `30` at `scale = 2`.

## Next Steps

Continue with [Gradient-Based Optimization](gradient-based-optimization.md) for a complete optimization loop. [Reverse and Forward Gradient Flow](gradient-flow.md) covers branch/replica adjoints, the native HDF5 gradient-file interface, and workflow JVPs. [Stochastic Execution and Reproducibility](reproducibility.md) explains module seed assignment. See also [batch-fused workflow stages](batch-fusion.md) and the [file storage reference](io.md).
