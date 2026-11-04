# Writing Modules

Modules are designed as stateless units of computation. Each module receives named parameter values, exchanges data through declared files, and optionally returns named results. Persistent state, such as a trained model, is [saved and loaded explicitly](artifacts.md) rather than passed between tasks through mutations to the module instance. This makes the computation portable across local execution, distributed workers, and containers.

!!! note "Containerized modules"

    Defining containerized modules is recommended to support reproducibility and long-term support by preserving the execution environment and its dependencies. See [Containerized Modules](containers.md) for details.

## Fundamentals

The default forward interface is `__call__(self, design, **kwargs)`, where `design` contains decoded [candidate values](parameters.md#explicit-candidates). Module initialization must call `super().__init__()` before assigning attributes: the base class intercepts assignments to register file declarations.

!!! note "Dynamic submodules"

    To expose individual methods as separate workflow stages, use `@dynamic_submodule`. See [Module with Dynamic Submodules](module-templates-and-dynamic-submodules.md#module-with-dynamic-submodules) for module templates and backward implementations.

```python
from dsbo.modules import Module


class CustomModule(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def __call__(self, design, **kwargs):
        # Implement computation using design and self.config here.
        return {}
```

Modules return a dictionary of named objective values. Each objective is either a scalar or a one-dimensional tensor containing one value per sample. Modules that only produce files may return `None`, which is treated as an empty dictionary.

The workflow merges task results into one dictionary per candidate. Duplicate keys raise `ValueError`; module names and aliases do not automatically namespace them.

Result dictionaries are aggregated, not injected into downstream calls. Each task receives the candidate's decoded design; inter-task data dependencies use the declared-file interface.

Returning results does not automatically create optimization-trial records. Recording returned and intermediate metrics is described in [Logging Metrics](metrics.md).

## Module with File I/O

Modules exchange data through files whose locations depend on where and how the workflow runs. Separate executions need separate files, and working directories may differ between execution environments. `Workdir.register_file()` creates a file reference instead of a fixed path, allowing the workflow to determine the appropriate location at runtime. The `Input` and `Output` wrappers declare whether the module reads or writes that file:

```python
import torch

from dsbo.modules import Module
from dsbo.utils.io import Output, Workdir
from dsbo.utils.io.storage import HDF5File


class ArraySource(Module):
    def __init__(self):
        super().__init__()
        self.output = Output(Workdir.register_file("data", "samples.h5"))

    def __call__(self, design, **kwargs):
        values = design["scale"] * torch.arange(4, dtype=torch.float32)
        HDF5File.write(self.output(), values)
```

Registering a file does not create it or prescribe its format. The module writes its data to the registered location using a format appropriate for the task. The example above uses `HDF5File.write()` to store a tensor. Other formats and file-handling utilities are described in [File Storage and Working Directories](io.md).

### Registered Paths

Calling a handle, as in `self.output()`, resolves it against the active execution context. Multi-candidate batches introduce candidate-specific subdirectories; replicas normally introduce filename suffixes. Resolve handles during execution, not construction: caching a resolved filename bypasses these scopes on subsequent calls. Options for shared replica files are described in [Resolving Paths](io.md#resolving-paths).

### Declaring Input and Output Dependencies

A consumer accepts the underlying handle rather than a filename resolved in the producer's current context:

```python
from dsbo.utils.io import Input


class MeanValue(Module):
    def __init__(self, input_path):
        super().__init__()
        self.input = Input(input_path)

    def __call__(self, design, **kwargs):
        values = HDF5File.open(self.input(), device="cpu")
        return {"mean": values.mean()}
```

This constructor expects `source.output.path`, which preserves the registration. In contrast, `source.output()` resolves the path prematurely and supplies a filename rather than the expected handle.

File declarations do not establish task order. The dependency must be expressed separately in the [workflow definition](workflows.md#defining-a-workflow); matching input and output paths alone do not prevent the consumer from running before its file is produced.

### Working Directory and File Lifetime

`Workdir.create()` requires a new directory by default and removes it on exit. Intermediates needed for a backward pass or subsequent task must be used before cleanup, or retained with `keep=True`.

Path registration does not stage files to workers. Distributed runs require a worker-accessible working directory; see [Parallelization with Dask](parallelization.md).

## Trainable Modules

`TrainableModule` extends `Module` with workflow-managed training stages for modules that learn from data, including surrogates. Subclasses implement `train` or `pretrain`, optionally `finetune`, as dynamic submodules, and define `state_dict()` and `from_state_dict()` to save and restore learned state. Trained models are persisted explicitly as [artifacts](artifacts.md), rather than relying on mutations to a module instance being shared between tasks. See [Writing Surrogates](surrogates.md) for a concrete implementation.

For reconstruction models, `ReconstructionBase` from `dsbo.modules.reconstruction` provides a predefined template whose forward call runs `train()` followed by `validate()`, returning the validation results. It inherits directly from `Module`, rather than `TrainableModule`, and requires subclasses to implement both methods. `ReconstructionContainerWrapper` extends this template with container execution support.
