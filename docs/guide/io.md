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

`validate_workflow_dependencies(wf, on_error="raise")`, imported from `dsbo.workflow`, checks input and output declarations. It does not check that files exist or replace the explicit dependencies between tasks.

## File Backend

The active **file backend** supplies the format for framework-managed arrays, gradients, and replica reductions. A module remains backend-neutral only if it resolves that backend at runtime rather than calling a concrete format class directly.

`HDF5File` is the default backend. It stores tensors as HDF5 datasets, which can be inspected with standard HDF5 tools.

For C++ installation requirements and standalone usage, see [Native C++ HDF5 I/O](native_io.md).

To read and write a tensor directly:

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
- Scalar flat datasets become a vector with one element per source.
- TensorDict stores concatenate each key along axis 0. A scalar TensorDict store (all leaves rank 0) is concatenated into a vector with one element per source; stores mixing scalar and non-scalar leaves are rejected.
- Reduction sources must share the same key set, per-key trailing shape, and dtype.

For example, two TensorDict stores with `energy` shapes `(3,)` and `(2,)` and `position` shapes `(3, 2)` and `(2, 2)` produce `energy` with shape `(5,)` and `position` with shape `(5, 2)`. Keys remain separate; reduction does not stack them into a common tensor or promote incompatible dtypes. Flat stores and TensorDict stores cannot be mixed in one reduction.

Sources are ordered by replica index, and rows within each source retain their order. Keep the source files available for as long as the combined input is needed, including any backward pass.

[Batch-Fused Workflow Stages](batch-fusion.md) describes how to combine inputs across candidates. Custom pipelines can also call `HDF5File.reduce(sources, destination)` directly, or use `get_active_file_backend().reduce(...)` for the configured backend.

## File-Based Autograd

Beyond the reverse-mode interface introduced in [Differentiable Workflows](autograd.md#file-based-autograd), `HDF5File.write()` accepts dual tensors or an explicit `(primal, tangent)` tuple. Keyed primals and tangents use matching field names; tangent keys can be a subset of the primal keys. Use `open()` under `torch.no_grad()` for ordinary reads, with gradient tracking enabled for reverse-mode differentiation, or inside a forward-AD level to read dual tensors. Keyed reads return a TensorDict. Use `read_backward_gradient()` to read reverse-mode adjoints; `open()` does not return them.

Unused TensorDict derivative keys are omitted, not zero-filled. See [TensorDict Adjoints](gradient-flow.md#tensordict-adjoints) and [Tangents Per Key](gradient-flow.md#tangents-per-key) for complete examples.

## Backend Interface

Custom backends must provide the operations used by their workflows and training code:

| Surface | Backend operations |
|---|---|
| Tensor I/O | `open(fpath, device, columns=None)`, `write(fpath, input)` |
| Reductions and adjoints | `reduce`, `read_backward_gradient`, `aggregate_gradients` |
| Dataset access | `write_named_datasets`, `write_row_store`, `open_dataset`, `shape_from_path` |

`open_dataset()` returns a `DatasetView` with keys, per-key shapes, indexed reads, partition sizes, and `close()`. Views also support context-manager cleanup. `statistics(keys)` returns count, mean, and squared-deviation summaries when available; TensorDict views return these per key.

## TensorDict Storage

Use a TensorDict store when one file carries several named quantities. Unlike a stacked tensor, the fields can have different trailing shapes and dtypes.

`write()` accepts a `torch.Tensor`, a `TensorDict`, or a flat `dict[str, torch.Tensor]`. For keyed values:

- Each key keeps its own dtype and shape.
- Reads preserve the key order unless you request a different selection.
- Keys must be flat, non-empty strings without `/` or NUL characters; the literal key `"."` is also invalid. Nested TensorDicts and tuple keys are not supported.

```python
import torch
from tensordict import TensorDict

from dsbo.utils.io import Workdir
from dsbo.utils.io.storage import HDF5File

with Workdir.create("files/tensordict_example"), torch.no_grad():
    path = Workdir.register_file("simulation", "events.h5")()
    events = TensorDict({
        "energy": torch.tensor([1.0, 2.0, 3.0]),
        "position": torch.arange(6, dtype=torch.float32).reshape(3, 2),
        "event_id": torch.arange(3, dtype=torch.int64),
    }, batch_size=[3])
    HDF5File.write(path, events)  # A plain dict[str, Tensor] works too.

    events = HDF5File.open(path, device="cpu")
    print(events.batch_size)          # torch.Size([3])
    print(events["position"].shape)   # torch.Size([3, 2])
    print(events["event_id"].dtype)   # torch.int64

    selected = HDF5File.open(path, device="cpu", columns=["position", "energy"])
    print(list(selected.keys()))     # ['position', 'energy']
    print(HDF5File.shape_from_path(path)["position"])  # (3, 2)
```

Batched fields must share their leading size: `(N,)`, `(N, 2)`, and `(N, 4, 4)` can coexist, but `(N,)` and `(M,)` cannot when `N != M`. All-scalar mappings have an empty batch shape; mixing a scalar field with batched fields is invalid. Repeat a per-file scalar across the samples if it belongs in a batched store.

Reads return `batch_size=(N,)` for batched fields; additional axes remain field dimensions, even if the original TensorDict declared more batch dimensions. `shape_from_path()` returns a dictionary of field shapes for keyed files. Use nonempty mappings for file round trips.

`open(fpath, device, columns=None)` returns a `TensorDict` for dict stores and a tensor for flat stores. For dict stores, `columns` selects an ordered subset of keys. Flat stores do not accept column selection.

Selection changes only the read, not the file's keys. Requesting a missing key raises `KeyError`; duplicate or invalid keys are rejected. During a module's backward pass, open the required keys once and reuse the returned TensorDict.

Tangents and adjoints use the same keys as their corresponding values. Keys that received no gradient are omitted, so querying them with `.get(key)` returns `None`, while zero gradients are retained.

### Migrating Named-Column Writes

The former `write(path, tensor, columns=[...])` interface is replaced by keyed values. For an existing column-first tensor, write each column under its name:

```python
# Previously: HDF5File.write(path, stacked, columns=["energy", "cost"])
# stacked has shape (2, N).
HDF5File.write(path, {"energy": stacked[0], "cost": stacked[1]})
values = HDF5File.open(path, device="cpu", columns=["cost"])
cost = values["cost"]  # shape (N,), not a stacked (1, N) tensor
```

Existing named-column files remain readable, but now return a TensorDict rather than a stacked tensor. The `columns=` argument remains available on reads, not writes. Flat tensor files still return tensors.

### Choosing a Write Method

Choose the write method according to how the data will be used:

| Method | Intended use |
| --- | --- |
| `write(fpath, TensorDict)` or `write(fpath, dict)` | Autograd-aware named values; read with `open()` |
| `write_named_datasets(fpath, datasets)` | Named NumPy arrays for indexed dataset reads |
| `write_row_store(fpath, datasets)` | Equal-length scalar feature columns for sample-first training batches |

These methods are not interchangeable. Dataset views expose samples first. Views of tensor, TensorDict, and named-array files return a TensorDict from `read_many()` and `read_indices()`. Files written with `write_row_store()` return tensor batches with features on the final axis. A single-key `read()` returns a tensor. Use `open()` for file-based differentiation; dataset views are intended for training and data inspection.

## Indexed Dataset Reads

Use a dataset view to fetch rows without loading every field in full. Index lists preserve their requested order and may contain repeated or negative indices:

```python
import torch

from dsbo.utils.io import Workdir
from dsbo.utils.io.storage import HDF5File

with Workdir.create("files/dataset_view"):
    path = Workdir.register_file("simulation", "events.h5")()
    HDF5File.write(path, {
        "energy": torch.tensor([1.0, 2.0, 3.0]),
        "position": torch.arange(6, dtype=torch.float32).reshape(3, 2),
    })
    with HDF5File.open_dataset(path) as view:
        rows = view.read_indices(["energy", "position"], [2, 0, 2])
        print(rows["energy"])          # tensor([3., 1., 3.])
        print(rows["position"].shape)  # torch.Size([3, 2])
        one_row = view.read_many(["energy", "position"], slice(0, 1))
        print(one_row.batch_size)      # torch.Size([1])
```

Integer indexing removes the sample axis. When reading scalar-per-sample and vector-valued fields together, retain that axis with a length-one slice or `read_indices(keys, [i])`; `read_many(keys, i)` does not support every combination of field shapes. Use `read(key, i)` when you want a single field without its sample axis. Close views before the workdir is cleaned up, and keep source files available while reading combined inputs.

## Training Datasets

The pre-built surrogates assemble training datasets from workflow files automatically. The wrappers in `dsbo.utils.io.datasets` implement `torch.utils.data.Dataset`:

- **Module surrogates** wrap the contract's data and conditioning files in a `ModuleSurrogateDataset`, which serves matched `(x, c)` pairs from the module's outputs and conditioning.
- **Workflow surrogates** wrap the row-store results file together with the design matrix in a `WorkflowSurrogateDataset`. The results file is recorded automatically while the training candidates run: each task's returned metrics are appended as rows.

`WorkflowSurrogateDataset` provides keyed scalar results as a target tensor with shape `(rows, number_of_keys)`; it does not treat vector-valued fields as scalar metrics. `ModuleSurrogateDataset` and `LazyShape` currently require files written from a single tensor, not a TensorDict. Writing named data does not automatically make it compatible with the pre-built module surrogates.

See [Writing Surrogates](surrogates.md) for declaring the data and conditioning that these datasets use.
