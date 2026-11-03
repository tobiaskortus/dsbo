# Artifacts

Artifacts associate files and model state with a name, version, and metadata through `study.artifacts`. They provide persistent storage for checkpoints and other outputs beyond the lifetime of a workflow's working directory.

| Method | Stored Content | Loading Interface |
| --- | --- | --- |
| `add_file(path, name=...)` | A copy of an existing file, such as a model and optimizer checkpoint | `study.artifacts.load_file(ref, name)` returns the stored file path |
| `add_model(model)` | A PyTorch module's `state_dict()` | `get_model()` on a loaded artifact returns the state dictionary |

An artifact can contain multiple files and model entries.

## Storage and Retention

The SQLModel backend stores artifact records in the study database and artifact contents in filesystem directories. For a database named `<name>.db`, the default artifact root is `<name>_artifacts` beside the database:

```text
<artifact_root>/artifacts/<name>/v<version>/
```

Each version contains an `artifact.json` manifest, attached files under `files/`, and model state dictionaries as `<model_name>.pt`. Preserving an artifact requires both the study database and its artifact directory.

An explicit `artifact_root` can be supplied to `StudyStorage.create()` and `StudyStorage.open()`. Both calls require the same root selection for a study with a custom root.

!!! note "Working-directory cleanup"

    Files declared as module `Input` or `Output` are subject to [working-directory cleanup](io.md). Saving them as artifacts creates versioned copies outside the source directory. The artifact root must therefore be outside temporary working directories, and saving must complete before the source directory is removed. In distributed execution, the database and artifact root must be accessible to the processes saving and loading artifacts.

## Publishing a Checkpoint File

`add_file()` attaches an existing file to an artifact. For training checkpoints, the file can include model weights, optimizer state, and model configuration.

The following example assumes a trained PyTorch `model`, its `optimizer`, and a JSON-serializable `config` dictionary containing the model's constructor arguments:

```python
from datetime import datetime

import torch

from dsbo.storage.sqlmodel import StudyStorage
from dsbo.utils.io import Workdir

study_path = f"model_{datetime.now():%Y%m%d-%H%M%S-%f}.db"

with StudyStorage.create(study_path) as study:
    with Workdir.create("checkpoint_export"):
        checkpoint_path = Workdir.register_file("model", "checkpoint.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
            },
            checkpoint_path(),
        )

        artifact = study.artifacts.create_artifact("model", type="model")
        artifact.add_file(checkpoint_path(), name="model.pt")
        artifact.save(study.artifacts, stage="trained")
        file_ref = artifact.ref
```

`add_file()` records the source path; `save()` copies the file into artifact storage. The attachment name, `model.pt`, identifies the stored file and may differ from its source filename.

`load_file()` resolves the saved reference to a file path. Checkpoint restoration uses the model's original class, represented here by `Model`:

```python
with StudyStorage.open(study_path) as study:
    stored_path = study.artifacts.load_file(file_ref, "model.pt")
    state = torch.load(stored_path, weights_only=True, map_location="cpu")

    file_model = Model(**state["config"])
    file_model.load_state_dict(state["model"])
    file_optimizer = torch.optim.Adam(file_model.parameters(), lr=0.01)
    file_optimizer.load_state_dict(state["optimizer"])
```

## Publishing a Model Object

`add_model()` stores a PyTorch module's `state_dict()` when the artifact is saved, without an intermediate checkpoint file. Model constructor arguments can be stored in JSON-serializable metadata:

```python
object_study_path = f"model_objects_{datetime.now():%Y%m%d-%H%M%S-%f}.db"

with StudyStorage.create(object_study_path) as study:
    artifact = study.artifacts.create_artifact(
        "model", type="model", metadata={"config": config},
    )
    artifact.add_model(model)
    artifact.save(study.artifacts, stage="trained")
    object_ref = artifact.ref
```

On a loaded artifact, `get_model()` returns the state dictionary with tensors on the CPU. Restoring the model requires its original class and constructor arguments:

```python
with StudyStorage.open(object_study_path) as study:
    loaded = study.artifacts.load_artifact(
        object_ref.name, version=object_ref.version,
    )
    restored_model = Model(**loaded.metadata["config"])
    restored_model.load_state_dict(loaded.get_model())
```

Optimizer and scheduler state are not included by `add_model()`. A file checkpoint can contain these alongside the model state, as in the preceding example.

## Versions and References

After saving, `artifact.ref` provides an `ArtifactRef` containing the artifact name, type, assigned version, and optional stage. The reference identifies a specific version within the corresponding study storage.

Each save creates a new version for the artifact name within its storage root. The remaining fields describe the artifact:

- **`type`:** a category, such as `"model"`, used to filter artifact listings.
- **`stage`:** an optional label, such as `"trained"` or `"finetuned"`, used to filter versions when loading.
- **`step`:** an optional optimization or training step, supplied to `save(..., step=step)`.
- **`metadata`:** a JSON-serializable dictionary, supplied at creation, for configuration and other descriptive values.

`load_artifact(name)` returns the latest version. The `version` argument selects a specific version; `stage` restricts selection to versions with that label:

```python
with StudyStorage.open(study_path) as study:
    selected = study.artifacts.load_artifact("model", version=file_ref.version)
    latest = study.artifacts.load_artifact("model", stage="trained")
    versions = study.artifacts.list_artifacts(type="model")
    latest_version = study.artifacts.latest_version("model", stage="trained")
```

### Surrogate Checkpoints

Surrogate lifecycle tasks publish checkpoints as a `model.pt` file attachment. Runtime loading uses `load_file()` followed by the surrogate's `from_state_dict()`.

Automatic selection prioritizes the `finetuned`, `pretrained`, and `trained` stages, in that order, taking the latest version within the first available stage. Publication and reuse are described in [Training Lifecycle](surrogates.md#training-lifecycle).
