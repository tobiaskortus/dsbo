# Gradient-Based Optimization

This example combines a [parameter space](parameters.md), [stateless modules](modules.md), [workflow composition](workflows.md), and [reverse-mode derivatives](autograd.md) into a complete optimization loop. It runs locally with a standard PyTorch optimizer; surrogates and distributed execution are not prerequisites.

The problem is intentionally synthetic: a response controlled by `gain` and `bias` is optimized until it matches a fixed target. Run the Python blocks in order in one script or session. The example requires no external simulation code and completes in seconds.

## Design Space

The design is described by a `ParameterSpace` of bounded, continuous parameters. A `Candidate` is instantiated from explicit start values; passing `requires_grad=True` makes the underlying tensor the optimization variable:

```python
import torch
from dsbo.parameters import Candidate, ParameterSpace, Continuous

space = ParameterSpace(gain=Continuous(-2.0, 2.0), bias=Continuous(-2.0, 2.0))
candidate = Candidate.from_values(space, {"gain": 0.3, "bias": -1.5}, requires_grad=True)
```

!!! note

    `candidate.tensor` is an ordinary `torch.Tensor` and can be passed to any PyTorch optimizer. To draw a random starting point from the bounds instead, import `UniformSampler` from `dsbo.parameters` and use `space.sample(UniformSampler(), requires_grad=True)`.

## Modules

Two modules make up the workflow. **ModuleA** computes the response from the parameters and writes it to a declared `Output` file. Its `backward()` reads the resolved file gradient and propagates it through the local expression, following the pattern in [Differentiable Workflows](autograd.md#implementing-a-backward-pass).

**ModuleB** represents the objective stage: it reads the response through a declared `Input`, compares it against the target, and returns the loss as a result. Its `backward()` follows the automatic hook pattern — it re-opens the input file under `torch.enable_grad()`, and the `post_accumulate_grad_hook` installed by `open()` records the input's gradient contribution.

```python
from dsbo.modules import Module, Input, Output
from dsbo.utils.io import Workdir
from dsbo.utils.io.storage import HDF5File
from dsbo.workflow import Workflow

TARGET = 0.8


class ModuleA(Module):
    """Writes the response of the design parameters to a file."""

    def __init__(self):
        super().__init__()
        self.out = Output(Workdir.register_file("response", "response.h5"))

    def __call__(self, design, **kwargs):
        response = design["gain"] * torch.tanh(design["bias"])
        HDF5File.write(self.out(), response)
        return {"response": response}

    def backward(self, design, grad_output=None):
        gradient = HDF5File.read_backward_gradient(self.out())
        if grad_output is not None:
            gradient = gradient + grad_output.get("response", 0.0)
        with torch.enable_grad():
            response = design["gain"] * torch.tanh(design["bias"])
        response.backward(gradient)


class ModuleB(Module):
    """Reads the response file and scores it against the target."""

    def __init__(self, response_path):
        super().__init__()
        self.inp = Input(response_path)

    def __call__(self, design, **kwargs):
        response = HDF5File.open(self.inp(), "cpu")
        return {"loss": (response - TARGET) ** 2}

    def backward(self, design, grad_output=None):
        with torch.enable_grad():
            response = HDF5File.open(self.inp(), "cpu")  # registers the gradient hook
            loss = (response - TARGET) ** 2
        loss.backward(grad_output["loss"])
```

ModuleB receives ModuleA's registered path and wraps it in `Input`. The task dependency is added explicitly in the next step; the runtime resolves both handles inside the active workdir. ModuleA also returns its response for inspection and for the later [surrogate-assisted extension](surrogate-assisted-optimization.md). Its backward method accounts for both the file gradient and any direct gradient on that returned response.

## Workflow

Composing the two modules with `>>` adds the dependency edge between them. During execution, tasks run in topological order in the forward pass and in reverse topological order in the backward pass:

```python
a = ModuleA()
b = ModuleB(a.out.path)
wf = Workflow(a >> b)
```

## Optimization Loop

The optimization loop is a standard PyTorch training loop; the workflow serves as the differentiable function. `wf.run(candidate)` executes the forward pass through the file channel and returns result tensors that carry a `grad_fn` into the file graph. Consequently, `loss.backward()` triggers the file-mediated reverse pass — consumers before producers — and accumulates the result into `candidate.tensor.grad` through the parameter channel:

```python
optimizer = torch.optim.Adam([candidate.tensor], lr=0.05)

for step in range(200):
    with Workdir.create(f"./example/step={step}"):
        result = wf.run(candidate)
        loss = result["loss"].sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            values = candidate.detach().values
            print(f"step {step:3d}  loss {loss.item():.6f}  "
                  f"gain {float(values['gain']):.3f}  bias {float(values['bias']):.3f}")

values = candidate.detach().values
response = float(values["gain"]) * torch.tanh(torch.tensor(float(values["bias"])))
print(f"final: loss {loss.item():.6f}, response = {response:.4f} (target {TARGET})")
```

Running the script converges to the target response. Convergence is quick in the `gain` direction but slower in `bias`, where the `tanh` response saturates and the loss surface becomes flat — the final iterations only polish the remaining digits:

```text
step   0  loss 1.148208  gain 0.250  bias -1.450
step  20  loss 0.068169  gain -0.646  bias -1.435
step  40  loss 0.016988  gain -0.976  bias -1.844
step  60  loss 0.000091  gain -0.845  bias -1.658
step  80  loss 0.000098  gain -0.850  bias -1.676
step 100  loss 0.000036  gain -0.861  bias -1.697
step 120  loss 0.000006  gain -0.854  bias -1.687
step 140  loss 0.000001  gain -0.857  bias -1.691
step 160  loss 0.000000  gain -0.856  bias -1.690
step 180  loss 0.000000  gain -0.856  bias -1.690
final: loss 0.000000, response = 0.8000 (target 0.8)
```

!!! note "Execution details"

    Each iteration performs a complete forward and reverse pass inside its own temporary workdir, `example/step=<i>`, which is removed automatically when the iteration completes:

    1. **Forward** — ModuleA writes `response/response.h5`; ModuleB opens it (resolved through the file overlay of the active workdir) and returns the loss.
    2. **Reverse** — the runner resets the per-pass gradient state and invokes ModuleB's `backward()` first: it re-opens `response.h5`, torch backpropagates, and the hook records a private gradient contribution. ModuleA's `backward()` then reads the resolved adjoint for `response.h5` and propagates the gradient into `candidate.tensor`.
    3. **Update** — the optimizer applies the accumulated design gradient. Because each step uses a fresh workdir, no stale sidecar can leak between iterations.
