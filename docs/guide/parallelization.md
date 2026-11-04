# Parallelization with Dask

DSBO distributes workflow execution with [Dask](https://docs.dask.org) and [`dsbo-heterogeneous`](https://github.com/tobiaskortus/dsbo_heterogeneous), which extends Dask with clusters of heterogeneous worker types. Two worker types are currently provided: `LocalWorkerSpec` launches workers as local processes and `SLURMWorkerSpec` launches them as SLURM jobs (both described [below](#worker-specifications)). Each module can be annotated with a worker specification (see [Resource Annotation](workflows.md#resource-annotation)); at run time, every task is scheduled onto a matching worker. By default, tasks run sequentially in a single process, and a workflow without annotations runs entirely in-process. The following section describes how to distribute execution onto a Dask cluster.

!!! note "More job types are coming"
    Support for further jobqueue systems is planned; see the [`dask_jobqueue` documentation](https://jobqueue.dask.org/en/latest/) for examples.

## Running a Workflow on Dask

`HeterogeneousDaskWorker.from_workflow()` inspects the workflow, collects every distinct worker spec, and creates a Dask cluster with one worker pool per spec. Every spec must already have scaling bounds. Cluster lifetime and worker selection are separate contexts:

```python
from dsbo.worker.dask import HeterogeneousDaskWorker, WorkerContext

with HeterogeneousDaskWorker.from_workflow(wf, logdir="logs") as worker:
    with WorkerContext.use_worker(worker):
        result = wf.run(candidate)
        # If differentiating, call result["loss"].backward() here too.
```

Entering the worker alone does not select it: without `WorkerContext.use_worker(worker)`, workflow execution stays local. While selected, each task is scheduled onto a matching resource pool. Every module must therefore carry an annotation; scheduling an unannotated module raises an error. Batch execution is parallelized: `run_batch()` can evaluate candidates concurrently, subject to dependencies and available resources. Cluster logs are written to `logdir`.

`WorkerContext.use_worker()` checks for the jobqueue commands required by the specs. If SLURM commands are missing, it warns and selects local in-process execution instead. This is not a conversion of the SLURM allocation into an equivalent local cluster. `Problem.execution_context()` supplies both contexts for problems with worker annotations.

## Worker Specifications

A worker specification describes one pool of workers: which resources each provides and how the processes are launched. The resource parameters follow the [dask_jobqueue](https://jobqueue.dask.org/en/latest/) convention: `cores` and `memory` are the total resources of one job, split into `processes` Dask workers that each receive an equal share.

`SLURMWorkerSpec` launches workers as SLURM jobs:

```python
from dsbo.worker.spec import SLURMWorkerSpec

CPU_CONFIG = SLURMWorkerSpec(
    processes=5, cores=5, memory="48GB",
    walltime="48:00:00", partition="epyc-256",
).adapt(min_worker=1, max_worker=8)
GPU_CONFIG = SLURMWorkerSpec(
    processes=1, cores=4, memory="16GB",
    walltime="48:00:00", gpu_resources="gpu:1",
).scale(4)
```

| Parameter | Description |
|---|---|
| `cores`, `processes`, `memory` | CPU, process, and memory allocation of each SLURM job |
| `walltime` | Time limit per SLURM job |
| `partition` | SLURM partition to submit to |
| `account` | Optional SLURM account for project accounting |
| `gpu_resources` | GPU request, e.g. `"gpu:1"`, passed to SLURM as `--gres`. The spec-specific Dask resource token routes annotated tasks to this pool; it is not a generic GPU-availability probe |
| `min_worker`, `max_worker` | Bounds on job instances in this pool, set via `.adapt()` or `.scale()` |

`LocalWorkerSpec` launches workers as processes on the local machine. This runs the same Dask code path without an external batch scheduler:

```python
from dsbo.worker.spec import LocalWorkerSpec

LOCAL = LocalWorkerSpec(cores=4, processes=1, memory="16GB").scale(1)
```

It accepts `cores`, `processes`, and `memory`; `LocalWorkerSpec.default_local()` derives a specification from the machine's actual cores and memory, but still needs `.scale(...)` or `.adapt(...)` before cluster creation. Resource annotations constrain Dask scheduling; they do not configure native thread pools or select CUDA devices.

## Scaling

Two methods set the number of job instances per specification. Both return a new, immutable spec; use the returned value when annotating modules:

```python
CPU_CONFIG = CPU_CONFIG.adapt(min_worker=1, max_worker=8)  # elastic job count
GPU_CONFIG = GPU_CONFIG.scale(4)                          # four one-process jobs
```

`.scale()` fixes the job count; `.adapt()` grows and shrinks the pool between its bounds as tasks queue up and workers go idle. For example, `processes=5` with `.scale(2)` launches two jobs containing ten Dask worker processes in total. A workflow **task** is one module invocation, not a job submission; a job's processes serve many tasks over their lifetime.

Workers are reused across evaluations: a live worker serves tasks from every `run()` and `run_batch()` call, so queue wait and startup cost are paid once per worker, not per task or per run. How long a worker lives depends on the scaling mode: `.scale()` keeps it until the context closes, `.adapt()` retires and resubmits it with demand.

## Execution Context and Diagnostics

The Dask backend propagates the active configuration, study, workdir, file scopes, and differentiation mode to worker tasks. The database and files must be accessible there; they are not copied by worker selection. Keep these contexts active through any backward pass.

Distributed execution IDs are workdir paths, with `batch_<i>` appended for multi-candidate batches and `backward` for the reverse pass. Reusing an execution path within the same study raises `ExecutionConflictError`; separate runs can use `Workdir.create_subdir(...)`. These IDs are distinct from trial steps and intermediate-metric run IDs.

The workflow store records Dask task status, timing, worker names, and task-log events. The in-process runner does not register these traces. `StudyStorage.open()` opens an existing study for inspection:

```python
from dsbo.storage.sqlmodel import StudyStorage

with StudyStorage.open("study.db") as study:
    for execution in study.workflow.get_executions():
        traces = study.workflow.get_job_traces(execution.exec_id)
        for trace in traces:
            print(trace.name, trace.status, trace.start, trace.stop, trace.log)
```

`get_job_traces()` takes `Execution.exec_id`, not the database row's `id`. These are task traces, not SLURM job records or objective histories; see [Logging Metrics](metrics.md) for the latter.
