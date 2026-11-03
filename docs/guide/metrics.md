# Logging Metrics

Study storage manages trial records, intermediate metrics, workflow execution traces, and artifacts. Optimization metrics and training histories are recorded through two interfaces, depending on what a measurement describes:

| Store | Purpose | Recorded Data |
| --- | --- | --- |
| `study.trials` | An evaluated design, such as one optimization iteration | Design vector, objective and diagnostic metrics, optional constraints and feasibility |
| `study.intermediates` | A scalar history within a run, such as loss over training epochs | Tag, step, value, and an optional integer run ID |

Metrics are recorded through `study.trials.add_trial()` and `study.intermediates.add_scalar()`.

## Storage and Execution Context

!!! note "Creating versus opening a study"

    `StudyStorage.create(path)` requires a new database path and raises if it already exists. The timestamps in the examples below give each session a new path. `StudyStorage.open(path)` requires an existing database and opens trial and intermediate storage for reading; writes through these interfaces raise an error. It is not a way to resume metric logging.

Entering a `StudyStorage` context activates it for `get_current_storage()`. The built-in Dask backend reinstates the selected storage in worker tasks, provided the database is accessible to those workers. This propagates storage selection, not metric names or run identities.

!!! note "Default storage"

    When no study is active, `get_current_storage()` creates a temporary on-disk study and retains it as the session default. An explicit `StudyStorage` context determines the database used for logging.

Metric records survive run-directory cleanup when the study database is outside those directories. [Artifacts](artifacts.md) have separate versioned references and optional `step`, `stage`, and `metadata` values. These do not create links to trial rows or intermediate histories; the trial schema has no artifact-reference field.

Other recording mechanisms serve different purposes:

- **Workflow execution traces** record task status, timing, and text logs, not objective histories, as described in [Parallelization with Dask](parallelization.md).
- **Workflow-surrogate result collection** writes a file-backed training dataset, not trial or intermediate records, as described in [Writing Surrogates](surrogates.md).

## Optimization Metrics

Each call to `study.trials.add_trial()` records an evaluated design. The following loop extends [Gradient-Based Optimization](gradient-based-optimization.md): it assumes that `wf` is a differentiable workflow returning `"loss"` and that `candidate.tensor` has gradients enabled.

```python
from datetime import datetime

import torch

from dsbo.storage.sqlmodel import StudyStorage
from dsbo.utils.io import Workdir

study_path = f"metrics_{datetime.now():%Y%m%d-%H%M%S-%f}.db"
optimizer = torch.optim.Adam([candidate.tensor], lr=0.01)

with StudyStorage.create(study_path) as study:
    for step in range(3):
        with Workdir.create(f"evaluations/step={step}"):
            optimizer.zero_grad()
            result = wf.run(candidate)
            loss = result["loss"].sum()

            study.trials.add_trial(
                step=step,
                design_vec=candidate.tensor,
                objectives={"loss": loss},
            )

            loss.backward()
            optimizer.step()
```

Trial recording occurs **before `optimizer.step()`** so that the stored design vector matches the evaluation. Storage detaches and copies tensor values; recording does not consume the computation graph needed by `loss.backward()`.

- **Metric names:** the `objectives` and `constraints` dictionaries require consistent keys across trials.
- **Scalar values:** each metric must be a scalar. Scalar PyTorch tensors and NumPy scalars are converted automatically; multi-element tensors require reduction or selection of a single value.
- **Design coordinates:** the stored vector contains the numerical coordinates from `candidate.tensor`, without parameter names or decoded values from `candidate.values`. Interpreting the vector requires the original parameter-space definition.

### Constraints and Feasibility

Constraints and feasibility are optional and must be supplied explicitly. Storage does not infer feasibility from constraint values; `feasible` defaults to `True`. The following alternative trial call expresses a convention in which a residual must be non-positive:

```python
study.trials.add_trial(
    step=step,
    design_vec=candidate.tensor,
    objectives={"loss": loss},
    constraints={"g": residual},
    feasible=bool(residual <= 0),
)
```

Here, `residual` must be a scalar computed for the same evaluated design.

### Reading Trials

`get_trials()` returns records ordered by step. Each record exposes `step`, `objectives`, `constraints`, `feasible`, and the recorded coordinates through `design_vector`:

```python
with StudyStorage.open(study_path) as study:
    trials = study.trials.get_trials()
    loss_history = [(trial.step, trial.objectives["loss"]) for trial in trials]
    evaluated_vectors = [trial.design_vector for trial in trials]
```

Recorded design vectors are returned as NumPy arrays with `float32` coordinates.

## Intermediate Metrics

`study.intermediates.add_scalar(tag, step, value, run_id=None)` records measurements such as training and validation loss. Unlike a trial, an intermediate record does not contain a design vector.

- **`tag`** names the series, for example `"train/loss"` or `"validation/accuracy"`.
- **`step`** is the position within that series, usually an epoch or batch index.
- **`value`** is the scalar measurement.
- **`run_id`** is an optional integer identifying a training invocation. It is not assigned automatically and need not equal the outer optimization step.

In a training loop, losses are recorded after each epoch under separate training and validation tags. The following example assumes an initialized `model`, `optimizer`, and data loaders, with application-defined `train_one_epoch()` and `validate()` functions returning scalar losses:

```python
from datetime import datetime

from dsbo.storage.sqlmodel import StudyStorage

training_path = f"training_{datetime.now():%Y%m%d-%H%M%S-%f}.db"

with StudyStorage.create(training_path) as study:
    for epoch in range(20):
        train_loss = train_one_epoch(model, optimizer, train_loader)
        validation_loss = validate(model, validation_loader)

        study.intermediates.add_scalar(
            "train/loss", step=epoch, value=float(train_loss), run_id=0
        )
        study.intermediates.add_scalar(
            "validation/loss", step=epoch, value=float(validation_loss), run_id=0
        )
```

Both series belong to the same training invocation, identified by `run_id=0`. A subsequent training invocation uses a different run ID, allowing its epoch numbering to restart at zero without mixing histories.

Each call appends a record. Reusing a tag, step, and run ID does not overwrite the previous value.

### Logging Inside Modules

Inside a module, `get_current_storage()` provides access to the active study:

```python
from dsbo.storage import get_current_storage

storage = get_current_storage()
storage.intermediates.add_scalar(
    "train/loss", step=epoch, value=float(train_loss), run_id=run_id
)
```

### Reading Scalar Histories

```python
with StudyStorage.open(training_path) as study:
    tags = study.intermediates.get_tags()
    run_ids = study.intermediates.get_run_ids("train/loss")
    training_loss = study.intermediates.get_scalars("train/loss", run_id=0)
    run_metrics = study.intermediates.get_scalars(run_id=0)
```

With a tag, `get_scalars()` returns a value list; without one, it returns a dictionary keyed by tag.

Without a run-ID filter, matching runs are combined in step order. `get_run_ids()` lists only explicitly assigned IDs. Metrics logged without a run ID remain readable through `get_scalars()`, but invocations sharing a tag cannot be separated afterward by run ID.
