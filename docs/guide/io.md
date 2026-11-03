# File Storage and Working Directories

Modules exchange data through registered files. A registered handle identifies a file by its directory and name; the active working directory, candidate, and replica determine its location when the module runs. The file backend determines how array data is stored and read.

Start with [Writing Modules](modules.md#module-with-file-io) for declaring `Input` and `Output` handles. This page explains how those handles map to files, how to read and write arrays, and how outputs from multiple tasks are combined.

## The Workdir

A workdir is the root directory for one execution's files. Open it explicitly around the workflow run:

```python
from dsbo.utils.io import Workdir

# With an existing workflow and candidate:
with Workdir.create("files/run_001"):
    result = workflow.run(candidate)
    # Use intermediate files and run any backward pass inside this context.
```

`Workdir.create()` requires a new directory by default and removes it on exit. Use `keep=True` to retain files for inspection, or `Workdir.load("files/run_001")` to reopen a retained workdir without deleting it on exit. Pass `error_if_exists=False` only when deliberately reusing a directory; it does not prevent cleanup when `keep=False`.

Without an active workdir, registered paths resolve relative to the current working directory and a warning is emitted. For distributed execution, the workdir must be accessible to every worker; see [Parallelization with Dask](parallelization.md).

### Organizing Files

`Workdir.register_file("simulation", "showers.h5")` registers a file under the `simulation` subdirectory. Registration does not write the file; the producing module does that during execution.

Use a custom directory such as `"analysis"`, or import `DefaultPaths` from `dsbo.utils.io` for the predefined `SIMULATION`, `RECONSTRUCTION`, `SURROGATE`, `LOGS`, and `TRAINING_RECORDS` directories.

The framework adds scopes where needed: multi-candidate batches use `batch_<i>` directories, batch-fused stages use `batch_reduce`, and surrogate training uses `surrogate_training`. You can add your own scope with `Workdir.create_subdir(...)`; registered files resolve below it until that context exits.

## Resolving Paths

Register a handle once, then call it when the file is needed. For example, given `out = Workdir.register_file("simulation", "out.h5")`, the active context determines the resolved path:

| Context | Result of `out()` |
|---|---|
| No candidate or replica scope | `<root>/simulation/out.h5` |
| Replica 2 | `<root>/simulation/out_2.h5` |
| Candidate 3 in a multi-candidate batch | `<root>/batch_3/simulation/out.h5` |
| Candidate 3, replica 2 | `<root>/batch_3/simulation/out_2.h5` |

Candidate and replica indices are zero-based. A single-candidate batch does not add a `batch_0` directory.

The workflow manages these scopes automatically. The equivalent manual scopes illustrate how directory names and filename suffixes compose:

```python
from dsbo.utils.io import DefaultPaths, Workdir

out = Workdir.register_file(DefaultPaths.SIMULATION, "out.h5")

with Workdir.create("files/path_example"):
    print(out())  # files/path_example/simulation/out.h5
    with Workdir.create_subdir("validation"):
        with Workdir.use_file_suffix("2"):
            print(out())  # files/path_example/validation/simulation/out_2.h5
```

Resolve handles inside `__call__` or `backward`, not in the module constructor. A cached filename would keep pointing to the original location when the same module runs for another candidate or replica.

By default, `unique_filename=True` adds replica suffixes. Register a file with `unique_filename=False` to suppress those suffixes, for example for an input shared by replicas. Candidate subdirectories still apply; this option does not share a file across candidates or coordinate concurrent writes.

### Checking Dependencies

File declarations describe what tasks read and write, but do not establish execution order. Connect producers and consumers in the workflow with `>>`; see [Declaring Input and Output Dependencies](modules.md#declaring-input-and-output-dependencies).

`validate_workflow_dependencies(wf, on_error="raise")`, imported from `dsbo.workflow`, checks non-root inputs against declared outputs encountered earlier in topological order. It is a declaration check, not a check that files exist or that all required dependency edges are present.

## File Backend

The active **file backend** supplies the format for framework-managed arrays, gradients, and replica reductions. A module remains backend-neutral only if it resolves that backend at runtime rather than calling a concrete format class directly.

`HDF5File` is the default backend. It stores tensors as HDF5 datasets, which can be inspected with standard HDF5 tools. To read and write a tensor directly:

```python
import torch

from dsbo.utils.io import Workdir
from dsbo.utils.io.storage import HDF5File

samples = Workdir.register_file("simulation", "samples.h5")

with Workdir.create("files/storage_example"), torch.no_grad():
    HDF5File.write(samples(), torch.arange(6, dtype=torch.float32).reshape(3, 2))
    values = HDF5File.open(samples(), device="cpu")
    print(values.shape)  # torch.Size([3, 2])
```

### Selecting a Backend

The active backend is selected through `dsbo.config`:

```python
from dsbo.config import config
from dsbo.utils.io.storage import HDF5File

with config.use(file_backend=HDF5File):
    ...   # framework-managed I/O uses this backend
```

Custom backends implement the storage interface described in [Backend Interface](#backend-interface) and can be selected the same way.

Configuration does **not** redirect explicit `HDF5File.open(...)` or `HDF5File.write(...)` calls. Those always use HDF5. Inside a backend-neutral module's `__call__` or `backward`, resolve the configured class alongside the registered paths:

```python
from dsbo.config import get_active_file_backend

# Inside a module with existing Input/Output declarations:
backend = get_active_file_backend()
values = backend.open(self.input(), device="cpu")
backend.write(self.output(), values.square())
```

Resolve the backend at execution time, just like the file handles, because configuration can change between runs. Reads currently produce CPU tensors; move the returned tensor explicitly with `.to(device)` when computation requires another device.

Registered filenames using a known backend extension are resolved with the active backend's default extension. With HDF5, for example, `"results.hdf5"` resolves to `results.h5`. Unrelated extensions remain unchanged. Registration determines a path, not how a module serializes its contents.

## Combining Replica Outputs

In a fan-in such as `simulation * 3 >> reconstruction`, each simulation replica writes its own output file. When `reconstruction` resolves a matching input, the framework combines those files into one input view. The consumer reads that view through its normal input handle; it does not need to enumerate the replica files.

For HDF5, this operation is **concatenation**, not a sum or average:

- Arrays with shapes `(100, 4)`, `(80, 4)`, and `(120, 4)` produce a view with shape `(300, 4)`.
- Scalar datasets become a vector with one element per source.
- Named-column datasets concatenate along each column's sample dimension.

Sources are ordered by replica index, and rows within each source retain their order. The HDF5 view uses virtual datasets that reference the original files instead of duplicating their data. Keep the source files available for as long as the combined input is needed, including any backward pass.

The view is built lazily when the consumer resolves its input handle. [Batch-Fused Workflow Stages](batch-fusion.md) extends this mechanism across candidates. Custom pipelines can also call `HDF5File.reduce(sources, destination)` directly, or use `get_active_file_backend().reduce(...)` for the configured backend.

## File-Based Autograd

Beyond the reverse-mode interface introduced in [Differentiable Workflows](autograd.md#file-based-autograd), `HDF5File.write()` accepts an explicit `(primal, tangent)` tuple of arrays. `open()` reconstructs a plain, reverse-mode, or forward-mode tensor according to the active context. Outside an active backward pass, `read_backward_gradient()` resolves the adjacent gradient; ordinary primal reads do not load that adjoint.

Tangents travel with primals; reverse-mode adjoints use separate contribution files and additive aggregation. Gradient file handling is described in [Computing a Gradient Contribution](gradient-flow.md#computing-a-gradient-contribution).

## Backend Interface

`HDF5File` combines `AutogradFileInterface` with `HDF5Writer`. The adapter delegates primal, tangent, and gradient I/O to its `writer`; reductions and surrogate dataset operations are also exposed on the backend class.

| Surface | Backend operations |
|---|---|
| Tensor I/O | `open(fpath, device, columns=None)`, `write(fpath, input, columns=None)` |
| Reductions and adjoints | `reduce`, `read_backward_gradient`, `aggregate_gradients` |
| Dataset access | `write_named_datasets`, `write_row_store`, `open_dataset`, `shape_from_path` |

`open_dataset()` returns a `DatasetView` with keys, shapes, indexed reads, partition sizes, and `close()`. Its optional `statistics()` result contains count, mean, and squared deviation. Subclassing `AutogradFileInterface` alone does not supply reductions or dataset views.

## Named Columns

!!! warning "Experimental"

    Named columns are an experimental feature; the API may still change.

Arrays can carry named columns: pass `columns=[...]` to `write()` and, optionally, to `open()` to select a subset. Unlike a conventional sample-first table, the column index is tensor axis 0. For example, a tensor with shape `(2, 100)` and `columns=["energy", "cost"]` stores two columns with 100 samples each.

Column names and order are stored with the data. Reads reconstruct that order, and backward propagation zero-fills gradients for columns that received no gradient.

There are three distinct layouts for named data:

| Method | Layout and intended use |
| --- | --- |
| `write(..., columns=[...])` | Named columns in the autograd-aware `primal` layout; read with `open()` |
| `write_named_datasets(fpath, datasets)` | A flat store of named arrays from a `dict[str, np.ndarray]` |
| `write_row_store(fpath, datasets)` | Equal-length scalar feature columns stored as rows for efficient surrogate training |

These layouts are not interchangeable. Dataset views expose samples first, whereas named-column tensor reads expose columns first.

## Training Datasets

The pre-built surrogates assemble training datasets from workflow files automatically. The wrappers in `dsbo.utils.io.datasets` implement `torch.utils.data.Dataset`:

- **Module surrogates** wrap the contract's data and conditioning files in a `ModuleSurrogateDataset`, which serves matched `(x, c)` pairs from the module's outputs and conditioning.
- **Workflow surrogates** wrap the row-store results file together with the design matrix in a `WorkflowSurrogateDataset`. The results file is recorded automatically while the training candidates run: each task's returned metrics are appended as rows.

See [Writing Surrogates](surrogates.md) for declaring the data and conditioning that these datasets use.
