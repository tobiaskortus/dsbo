# Containerized Modules

Containers are an optional execution environment for [stateless modules](modules.md). File handoffs remain explicit: configuration may live on the instance, but outputs needed by later tasks are saved rather than transferred through instance mutations.

Container wrappers combine the [module templates](module-templates-and-dynamic-submodules.md#pre-defined-module-structures) with container execution through a backend-agnostic integration that supports Docker and Singularity/Apptainer:

| Base class | Container wrapper | Methods to implement |
|---|---|---|
| `SimulationBase` | `SimulationContainerWrapperBase` | `simulate()` |
| `ReconstructionBase` | `ReconstructionContainerWrapper` | `train()`, `validate()` |

## Image Definitions

A typed image definition from `dsbo.container` is passed to the wrapper constructor:

| Definition class | Input | Compatible installation | Preparation |
|---|---|---|---|
| `DefinitionFile` | Dockerfile | Docker | Builds with `docker build`, using the file's parent directory as the build context; reuses an existing image with the selected tag. |
| `DefinitionFile` | Apptainer/Singularity `.def` | Apptainer or Singularity | Builds a `.sif` under `container.path` (or the current directory); reuses an existing output file. |
| `ImageRef` | Registry reference | Docker, Apptainer, or Singularity | Docker uses a local image or pulls it; Apptainer/Singularity reuses an existing `.sif` or pulls it to one under `container.path` (or the current directory). |
| `ImageFile` | Local `.sif` | Apptainer or Singularity | Resolves relative paths under `container.path` (or the current directory); absolute paths are used as given. Does not build or pull. Docker rejects this format. |

```python
from dsbo.container import DefinitionFile, ImageFile, ImageRef

dockerfile = DefinitionFile("./Dockerfile", container_name="simulation:latest")
apptainer_def = DefinitionFile("./simulation.def", container_name="simulation")
registry_image = ImageRef("python:3.11-slim")
local_image = ImageFile("./simulation.sif")
```

`DefinitionFile` treats paths containing `Dockerfile` as Docker builds and paths ending in `.def` as Apptainer/Singularity builds; all other paths default to Docker. `container_name` sets the Docker tag or the `.sif` base name, and `build_flags` adds build arguments. Backend selection prefers Apptainer, then Singularity, then Docker among compatible installations. Any definition can be passed to a wrapper constructor, such as `ContainerSimulation(dockerfile, workspace)` below.

!!! warning "Image and installation compatibility"

    Docker cannot run `ImageFile` (`.sif`). Dockerfiles are built only by Docker, and `.def` files are built only by Apptainer/Singularity. Incompatible combinations raise an error.

!!! note "Image preparation"

    On the first workflow call (`run()` or `run_batch()`), dsbo discovers all containerized modules in the graph and pulls or builds their images before execution. Later calls reuse the prepared images for that workflow instance.

## Image Storage

Docker manages images in its own image store; `container.path` does not affect Docker.

Apptainer/Singularity stores `.sif` files under `container.path` when set, otherwise under the current working directory. A pulled `ImageRef` is named after its normalized URI, for example `python:3.11-slim` becomes `python_3.11-slim.sif`. A built `.def` file uses `container_name` or its normalized filename, for example `simulation.def` becomes `simulation.sif`. A relative `ImageFile` path resolves under the same directory; an absolute path is used unchanged.

The path is set before the first container call:

```python
from dsbo.config import config

config.set("container.path", "/shared/containers")
```

For a scoped override:

```python
with config.use(**{"container.path": "/shared/containers"}):
    ...  # container preparation uses this directory
```

Pull and build operations create the directory when needed.

## Mounts and Working Directory

The container sees host paths only if they are mounted into it. `bind_mount((host_path, container_path))` maps a host directory to a container path; relative paths are resolved to absolute paths. Docker uses `-v host:container`; Apptainer/Singularity uses `--bind host:container`.

The active workdir is mounted at the same path so registered filenames remain valid inside the container, and the container's working directory is set to that path:

```python
workspace = Workdir.get_current_dir()
self.bind_mount((workspace, workspace))
self.set_workdir(workspace)
```

!!! warning "Execution-time path resolution"

    `Workdir.get_current_dir()` returns the directory active during execution. A path resolved in `__init__` uses the construction-time directory instead. Mounts and the working directory are therefore configured inside `__call__` or a dynamic submodule.

`set_workdir()` takes the path as seen inside the container. With a different container path, the mount and path translation are explicit:

`cuda_enabled=True` adds the backend's CUDA flags to the container command automatically. `ReconstructionContainerWrapper` enables CUDA by default; dsbo disables it when the assigned worker has no GPU.

Template wrappers accept extra command-line flags through `container_extra_flags`:

```python
super().__init__(image, container_extra_flags="--rm")
```

## Container Arguments

`call_container()` accepts a list of command arguments. Resolved file handles such as `self.output()` are passed as arguments instead of hardcoded paths: they resolve to the current replica or batch scope, so dsbo can merge and aggregate the resulting files correctly.

```python
import json

from dsbo.container import DefinitionFile, ImageFile, ImageRef
from dsbo.modules.simulation import SimulationContainerWrapperBase
from dsbo.utils.io import Output, Workdir

class ContainerSimulation(SimulationContainerWrapperBase):
    def __init__(self, image: ImageRef | ImageFile | DefinitionFile):
        super().__init__(image)
        self.output = Output(
            Workdir.register_file("simulation", "output.h5")
        )

    def simulate(self, design, **kwargs):
        workspace = Workdir.get_current_dir()
        self.bind_mount((workspace, workspace))
        self.set_workdir(workspace)

        self.call_container(
            [
                "python3",
                "/opt/module/run.py",
                json.dumps({"scale": float(design["scale"])}),
                self.output(),
            ],
            is_exec=True,
        )
```

`is_exec=True` runs the command instead of the image runscript. For Docker, both modes launch a new container and the image's entrypoint still applies.

!!! warning "Resolved file arguments"

    File paths passed to a container command should come from resolved handles, not hardcoded strings. Hardcoded paths bypass replica and batch scoping and can break dsbo's merge and aggregation operations.

!!! note "Gradient files"

    Gradient file definition is currently Python-only through `HDF5File`; a general library for file operations is in development. Containerized modules that write gradients directly must resolve paths from `self.output.gradient()` and `self.input.gradient()` so replica and batch scopes apply. See [Validation and Lifetime](gradient-flow.md#validation-and-lifetime) for the completion rules.

## Defining New Container Module Types

For a lifecycle that does not fit the simulation or reconstruction templates, `Module` is combined with `ContainerWrapperBase`:

```python
from dsbo.container import ContainerWrapperBase, DefinitionFile, ImageFile, ImageRef
from dsbo.modules import Module

class MyContainerModule(Module, ContainerWrapperBase):
    def __init__(self, image: ImageRef | ImageFile | DefinitionFile):
        Module.__init__(self)
        ContainerWrapperBase.__init__(self, image)
```

`run_container()` uses the image runscript; `exec_container()` runs an explicit command. Files are registered with `Input`/`Output` and resolved inside `__call__`, as for ordinary modules. Mounts and the working directory are configured before the first container call. `ContainerWrapperBase` raises an error if a subclass does not initialize the mixin.

Existing templates are extended by subclassing `SimulationContainerWrapperBase` or `ReconstructionContainerWrapper` and implementing `simulate()` or `train()`/`validate()`; they provide `call_container()`.
