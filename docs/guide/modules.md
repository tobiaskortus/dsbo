# Writing Modules

A **module** is the atomic unit of computation in DSBO — anything from a physics simulation to a reconstruction model to a simple data transform.

## Fundamentals

Every module must follow three rules:

1. **Call `super().__init__()` as the first line of `__init__`**
2. **Accept `design: dict` as the first positional argument of `__call__`** — this dictionary contains the parameter values from your design space
3. **Return `None` or a `dict[str, float | Tensor]`** of named metrics — the keys become objective names the optimizer can reference

```python
from dsbo.modules import Module

class SimpleModule(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, design, **kwargs):
        param = design["my_param"]
        result = compute(param)
        return {"metric_name": result}
```

!!! warning "Modules are stateless"

    Module tasks run on workers as short-lived processes. Any state you assign to `self` during a task — for example a trained model stored on `self` — lives only on that worker's process and is lost once the task finishes; mutations do not propagate back to the client. Treat modules as stateless: keep everything that must outlive a task in durable storage (write files via `Workdir` / artifact storage) and pass references (paths, artifact refs) to the next task, which loads them on demand.

## Module with File I/O

Most modules read and write files. Use `Workdir.register_file()` to declare the files your module uses, then call the returned callable at runtime to get the resolved path:

```python
from dsbo.modules import Module
from dsbo.utils.io import Workdir, DefaultPaths

class MySimulation(Module):
    def __init__(self):
        super().__init__()
        self.output_file = Workdir.register_file(DefaultPaths.SIMULATION, "showers.npz")

    def __call__(self, design, **kwargs):
        result = run_simulation(design)
        np.savez_compressed(self.output_file(), primal=result)
        return {}
```

`DefaultPaths` provides standard subdirectory names (`SIMULATION`, `RECONSTRUCTION`, `LOGS`) so modules that share data through the same subdirectory stay consistent. You can also pass a custom string — e.g. `"analysis"` — for module-specific directories.

!!! note "Path resolution"

    `self.output_file()` resolves differently depending on context — replica indices append suffixes (`showers_2.npz`), batch subdirectories nest paths, and dry-run mode returns `None`. See the [I/O Guide](io.md) for details.

### Declaring Input and Output Dependencies

Wrap your file paths in `Input` or `Output` to enable automatic dependency validation across your workflow:

```python
from dsbo.utils.io import Input, Output, DefaultPaths

class ConsumerModule(Module):
    def __init__(self):
        super().__init__()
        self.sim_data = Input(Workdir.register_file(DefaultPaths.SIMULATION, "showers.npz"))
        self.result = Output(Workdir.register_file("analysis", "metrics.npz"))

    def __call__(self, design, **kwargs):
        data = np.load(self.sim_data())
        metrics = analyze(data)
        np.savez(self.result(), **metrics)
        return metrics
```

!!! note "Dependency validation"

    The workflow validator checks that every `Input` has a matching `Output` produced by an upstream module. This catches misconfigured pipelines before execution.

## Pre-Defined Module Structures

DSBO provides base classes that implement common patterns. Instead of writing `__call__` yourself, you extend one of these templates and implement the specific methods they call.

### Simulation Module

`SimulationBase` is for modules that produce output (usually files) without returning metrics. Implement `simulate()`:

```python
from dsbo.modules.simulation import SimulationBase
from dsbo.utils.io import DefaultPaths

class Geant4Sim(SimulationBase):
    def __init__(self):
        super().__init__()
        self.output = Workdir.register_file(DefaultPaths.SIMULATION, "edep.npz")

    def simulate(self, design):
        result = run_geant4(design)
        np.savez_compressed(self.output(), primal=result)
```

!!! note "Return value"

    `SimulationBase` runs `simulate()` and returns an empty dict. Use this template when your module's purpose is to generate data for downstream modules rather than compute an objective.

### Reconstruction Module (Train/Validate Pattern)

`ReconstructionBase` is for modules that train a model and then evaluate it. Implement `train()` and `validate()` — `__call__` runs them in that order:

```python
from dsbo.modules.reconstruction import ReconstructionBase
from dsbo.utils.io import DefaultPaths

class RecoModel(ReconstructionBase):
    def __init__(self):
        super().__init__()
        self.model_path = Workdir.register_file(DefaultPaths.RECONSTRUCTION, "model.pt")
        self.training_data = Workdir.register_file(DefaultPaths.RECONSTRUCTION, "train.npz")
        self.val_data = Workdir.register_file(DefaultPaths.RECONSTRUCTION, "val.npz")

    def train(self, design, **kwargs):
        data = np.load(self.training_data())
        model = MyModel()
        train_model(model, data)
        torch.save(model.state_dict(), self.model_path())

    def validate(self, design, **kwargs):
        model = MyModel()
        model.load_state_dict(torch.load(self.model_path()))
        val_data = np.load(self.val_data())
        metrics = evaluate(model, val_data)
        return {"accuracy": metrics["acc"], "loss": metrics["loss"]}
```

## Containerized Modules

To run a module inside a container (Apptainer/Singularity), use a pre-built wrapper that combines a base class template with the container engine:

| Base class | Container wrapper | Methods to implement |
|---|---|---|
| `SimulationBase` | `SimulationContainerWrapperBase` | `simulate()` |
| `ReconstructionBase` | `ReconstructionContainerWrapper` | `train()`, `validate()` |

Call `self.call_container()` to run scripts inside the container:

```python
from dsbo.modules.simulation import SimulationContainerWrapperBase
from dsbo.container import python_script_args
from dsbo.utils.io import Workdir, DefaultPaths
from filelock import FileLock

class Geant4ContainerSim(SimulationContainerWrapperBase):
    def __init__(self, container_path):
        super().__init__(container_path)

        self.design_file = Workdir.register_file(
            DefaultPaths.SIMULATION, "design.json", unique_filename=False
        )
        self.output_file = Workdir.register_file(
            DefaultPaths.SIMULATION, "output.npz", unique_filename=True
        )

    def simulate(self, design):
        with FileLock(self.design_file() + ".lock"):
            json.dump(to_serializable(design), open(self.design_file(), "w"))

        args = python_script_args(
            "container_scripts/sim.py",
            [self.design_file(), self.output_file()],
        )
        self.call_container(args, is_exec=True)
```

!!! note "Custom container modules"

    If your computation doesn't fit simulation or reconstruction patterns, extend `ContainerWrapperBase` directly.

## Differentiable Module (with Backward Pass)

For gradient-based optimization, implement `backward()` alongside `__call__`:

```python
from dsbo.utils.io.autograd import NumpyFile
from dsbo.utils.io import Workdir, DefaultPaths

class DifferentiableSim(Module):
    def __init__(self):
        super().__init__()
        self.output = Workdir.register_file(DefaultPaths.SIMULATION, "edep.npz")

    def __call__(self, design, **kwargs):
        result = run_simulation(design["absorber"], design["gap"])
        NumpyFile.write(self.output(), result)
        return {}

    def backward(self, design, grad_output=None):
        bar_edep = NumpyFile.read_backward_gradient(self.output()).numpy()
        grad_absorber, grad_gap = compute_vjp(design, bar_edep)

        torch.autograd.backward(
            [design["absorber"], design["gap"]],
            [torch.from_numpy(grad_absorber), torch.from_numpy(grad_gap)],
        )
```

!!! note "When `backward` is called"

    - `grad_output` is `None` for intermediate modules — gradients flow through files, not through the `grad_output` parameter
    - `grad_output` is a dict like `{"metric_name": grad_tensor}` for the last modules in the graph

## Module with Dynamic Submodules

Split a module's methods into separately runnable workflow steps using `@dynamic_submodule`:

```python
from dsbo.modules import dynamic_submodule
from dsbo.utils.io import Workdir, DefaultPaths

class MyModel(Module):
    def __init__(self):
        super().__init__()
        self.model_file = Workdir.register_file(DefaultPaths.RECONSTRUCTION, "model.pt")

    @dynamic_submodule
    def train(self, design, **kwargs):
        model = MyModel()
        train_model(model)
        torch.save(model.state_dict(), self.model_file())
        return {}

    @dynamic_submodule
    def validate(self, design, **kwargs):
        model = MyModel()
        model.load_state_dict(torch.load(self.model_file()))
        return {"mse": evaluate(model)}

    @validate.backward
    def backward(self, design, grad_output):
        results = self.validate(design)
        tensors = [results[k] for k in grad_output]
        grad_tensors = [grad_output[k] for k in grad_output]
        torch.autograd.backward(tensors, grad_tensors)


model = MyModel()
wf = model.train >> model.validate
```

!!! note "Attribute delegation"

    Dynamic submodules share their parent module's file paths, configuration, and other state automatically.

### Dynamic Submodules and Training Loops

When a `@dynamic_submodule` runs a PyTorch training loop inside a forward-AD workflow, wrap it with `@train_op`:

```python
from dsbo.autograd import train_op

class MyModel(Module):
    @train_op
    @dynamic_submodule
    def train(self, design, **kwargs):
        ...
```‚‚