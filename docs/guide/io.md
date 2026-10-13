# I/O and Autograd

Modules in a DSBO workflow exchange data through files. Each module registers the files it reads and writes; the framework resolves these registrations to concrete paths inside the run directory, so one module's output file is the next module's input file.

There is a reason for this design. Modules run as short-lived, stateless tasks — on dask workers, in containers, or locally — so in-memory objects cannot cross task boundaries and the filesystem is the only channel every execution mode shares. Storing gradients in files as well keeps the task graph differentiable even when a backward pass runs on a different worker than the forward pass, and lets containerized modules take part in gradient-based optimization at all.

Surrogates benefit the same way: the real module's registered files gather the training data automatically, and a surrogate writes its predictions into the same files, so downstream modules cannot tell the difference.

## The Workdir

The `Workdir` manages where modules read and write. Register every file a module uses in the module's `__init__`, then call the returned handle at runtime to get the resolved path. Modules downstream of this one would register the same `RECONSTRUCTION` file to read it:

```python
from dsbo.modules import Module
from dsbo.utils.io import Workdir, DefaultPaths, Input, Output
from dsbo.utils.io.storage import HDF5File

class MyReconstruction(Module):
    def __init__(self):
        super().__init__()
        self.input_file = Input(Workdir.register_file(DefaultPaths.SIMULATION, "simulation.h5"))
        self.output_file = Output(Workdir.register_file(DefaultPaths.RECONSTRUCTION, "reco.h5"))

    def __call__(self, design, **kwargs):
        true = HDF5File.open(self.input_file(), device="cpu")
        reco = reconstruct(true, energy_threshold=design["threshold"])
        HDF5File.write(self.output_file(), reco)
        return {"mse": torch.mean((reco - true) ** 2)}
```

`DefaultPaths` provides common subdirectories: `SIMULATION`, `RECONSTRUCTION`, `SURROGATE`, `LOGS`, and `TRAINING_RECORDS`. A custom name like `"analysis"` can be used if additional separation is required.

Wrapping a registered file in `Input` or `Output` marks it as a declared input or output. The workflow validator uses these annotations to check that every input is produced by an upstream module, so misconfigured pipelines are caught before execution.

The `Workdir.create` context manager creates a workdir on disk and makes it active for the duration of the block:

```python
with Workdir.create("./run_001") as workdir:
    ...   # all registered files now resolve under ./run_001
```

The subdirectory of a registered file is part of the registration itself (see above), so all files of a stage resolve below the same directory.

!!! note "Subdirectories"

    The framework creates additional subdirectories inside the active workdir on its own, e.g.

    - batch execution scopes every candidate into its own `batch_<i>` directory,
    - surrogate training collects its training data under `surrogate_training`.

    While such a scope is active, registered files resolve below it. Depending on the implemented algorithms, additional subdirectories to scope and organize data might be beneficial — use `with Workdir.create_subdir(...)` for this.

!!! note "Running without a workdir"

    If no workdir is active, registered paths resolve relative to the current working directory and a warning is emitted. Open a workdir around your workflow runs.

## Resolving Paths

A registered file is not a string but a callable. Calling it returns the path for the current execution context, which the framework sets while the workflow runs:

| Context | Result of `out()` |
|---|---|
| Plain run | `<root>/simulation/out.npz` |
| Replica *i* active | `<root>/simulation/out_2.npz` — the replica index is appended to the filename |
| Batch subdirectory active | `<root>/batch_3/simulation/out.npz` — one subdirectory per candidate |

By default each replica gets its own copy of a file (`unique_filename=True`). Pass `unique_filename=False` to register a file that all replicas share, such as a common design input.

Scripts can set the context explicitly with the same context managers the framework uses:

```python
with Workdir.use_file_suffix("2"):
    out()   # <root>/simulation/out_2.npz — resolve as replica 2
```

## Storage Backends

A single **file backend** decides the format — how arrays, gradients, and replica reductions are written to and read from disk — keeping the whole workflow consistent without any module knowing about it.

Three backends are available, all implementing the same interface (`dsbo.utils.io.storage.AutogradFileInterface`):

- `HDF5File` — the default. Stores arrays as HDF5 datasets; widely supported and inspectable with standard tools.
- `ROOTFile` — stores arrays as ROOT TTrees via uproot; use it when results must be readable from ROOT-based analysis pipelines.

```python
import dsbo
from dsbo.utils.io.storage import ROOTFile

with dsbo.config.use(file_backend=ROOTFile):
    ...   # all reads and writes now use ROOT TTrees
```

The backend is also applied to *filenames*: registered files whose extension matches a known backend format (`.h5`, `.hdf5`, `.root`) are transparently rewritten to the active backend's default extension. Registering `"results.h5"` under the ROOT backend resolves to `results.root` — so modules stay backend-agnostic while files on disk are always consistent. Unrelated extensions (`.npz`, `.json`, ...) pass through unchanged.

## Combining Replica Outputs

Replicas write to per-replica files (the suffix resolution in [Resolving Paths](#resolving-paths)). Whenever a module's input is written by multiple upstream tasks — a fan-in, for instance — the framework reduces them into one file before the module reads it; the same happens for batch-fused stages. Custom pipelines can call `reduce_registered_file(sources, destination)` from `dsbo.utils.io.reduce` directly.

Reduction is **virtual**: the destination does not copy the source data. For HDF5 it contains virtual datasets (VDS) referencing the sources plus a manifest and per-partition row counts; for ROOT it stores a manifest that is resolved through uproot's multi-file concatenation on read. Reads of the reduced file therefore resolve lazily through the sources, so no data is duplicated on disk.

## File-Based Autograd

File contents participate in automatic differentiation. The storage backends share one interface, and `HDF5File` is the default choice:

```python
from dsbo.utils.io.storage import HDF5File

HDF5File.write(fpath, tensor)             # stores primal (and tangent, see below)
x = HDF5File.open(fpath, device="cpu")    # returns a torch.Tensor
```

`write()` accepts a `torch.Tensor` or an explicit `(primal, tangent)` tuple of numpy arrays — the latter lets you store tangents without a forward-AD context active. What `open()` returns depends on the surrounding AD context:

- **Plain evaluation** — the values as a tensor.
- **Reverse AD** (`torch.is_grad_enabled()`) — a tensor that tracks gradients; once backward pass accumulates a gradient into it, the gradient is written back to the file automatically.
- **Forward AD** (inside a dual-level context) — a dual tensor carrying primal and tangent, if the file was written from a dual tensor (or tuple with tangent).

The gradient round-trip in both directions:

```python
import torch

HDF5File.write(fpath, torch.tensor([1.0, 2.0]))
x = HDF5File.open(fpath, device="cpu")

loss = (x * torch.tensor([2.0, 3.0])).sum()
loss.backward()

HDF5File.read_backward_gradient(fpath)   # tensor([2., 3.])
```

Gradients are written to a sidecar file next to the primal: `results.h5` gets `results_gradient.h5`. Multiple `backward()` calls in the same pass **accumulate** into that sidecar, and the accumulated state is reset between passes. Reading a gradient for a file whose `open()` happened in a different pass raises a `RuntimeError` naming the file.

This is the mechanism behind "gradients flow through files": a differentiable module writes its outputs with `write()`, and its `backward()` reads the adjoint with `read_backward_gradient()` — no tensors need to be passed between tasks (see [Writing Modules](modules.md#differentiable-module-with-backward-pass)).

### Named Columns

!!! warning "Experimental"

    Named columns are an experimental feature; the API may still change.

Arrays can carry named columns: pass `columns=[...]` to `write()` (and to any `open()` to select a subset), or write a `dict[str, np.ndarray]` with `write_named_datasets()`. Columns are stored as an HDF5 group (or ROOT branches) with the column list persisted in metadata, so reads reconstruct the original column order and zero-fill gradients of columns that received no gradient. For tabular results with many rows and named features, `write_row_store(fpath, {"energy": e, "cost": c})` stores an interleaved row store that surrogate training reads efficiently.

## Training Datasets

The pre-built surrogates turn the files a workflow writes into training data automatically — no manual dataset code is required. Under the hood, `dsbo.utils.io.datasets` provides `torch.utils.data.Dataset` wrappers that the surrogate lifecycle feeds the recorded files into:

- **Module surrogates** wrap the contract's data and conditioning files in a `ModuleSurrogateDataset`, which serves matched `(x, c)` pairs from the module's outputs and conditioning.
- **Workflow surrogates** wrap the row-store results file together with the design matrix in a `WorkflowSurrogateDataset`. The results file is recorded automatically while the training candidates run: each task's returned metrics are appended as rows.
