# Parallelization with Dask

DSBO distributes workflow execution with [Dask](https://docs.dask.org) and [`dsbo-heterogeneous`](https://github.com/tobiaskortus/dsbo_heterogeneous), which extends Dask with clusters of heterogeneous worker types. Two worker types are currently provided: `LocalWorkerSpec` launches workers as local processes and `SLURMWorkerSpec` launches them as SLURM jobs (both described [below](#worker-specifications)). Each module can be annotated with a worker specification (see [Resource Annotation](workflows.md#resource-annotation)); at run time, every task is scheduled onto a matching worker. By default, tasks run sequentially in a single process, and a workflow without annotations runs entirely in-process. The following section describes how to distribute execution onto a Dask cluster.

!!! note "More job types are coming"
    Support for further jobqueue systems is planned; see the [`dask_jobqueue` documentation](https://jobqueue.dask.org/en/latest/) for examples.

## Running a Workflow on Dask

`HeterogeneousDaskWorker.from_workflow()` inspects the workflow, collects every distinct worker spec, and launches a Dask cluster with one worker pool per spec. It is used as a context manager:

```python
from dsbo.worker.dask import HeterogeneousDaskWorker

with HeterogeneousDaskWorker.from_workflow(wf, logdir="logs") as worker:
    result = wf.run(candidate)
```

While the worker is active, each task is scheduled onto a worker pool that provides the resources its module was annotated with. Every module must therefore carry an annotation — scheduling an unannotated module raises an error. Batch execution is parallelized: `run_batch()` evaluates its candidates concurrently. Cluster logs are written to `logdir`.

## Worker Specifications

A worker specification describes one pool of workers: which resources each provides and how the processes are launched. The resource parameters follow the [dask_jobqueue](https://jobqueue.dask.org/en/latest/) convention: `cores` and `memory` are the total resources of one job, split into `processes` Dask workers that each receive an equal share.

`SLURMWorkerSpec` launches workers as SLURM jobs:

```python
from dsbo.worker.spec import SLURMWorkerSpec

CPU_CONFIG = SLURMWorkerSpec(
    processes=5, cores=5, memory="48GB",
    walltime="48:00:00", partition="epyc-256",
)
GPU_CONFIG = SLURMWorkerSpec(
    processes=1, cores=4, memory="16GB",
    walltime="48:00:00", gpu_resources="gpu:1",
)
```

| Parameter | Description |
|---|---|
| `cores`, `processes`, `memory` | CPU, process, and memory allocation of each SLURM job |
| `walltime` | Time limit per SLURM job |
| `partition` | SLURM partition to submit to |
| `account` | Optional SLURM account for project accounting |
| `gpu_resources` | GPU request, e.g. `"gpu:1"`. Passed to SLURM as `--gres` and registered as a Dask resource, restricting GPU-annotated tasks to these workers |
| `min_worker`, `max_worker` | Scaling bounds, set via `.adapt()` or `.scale()` |

`LocalWorkerSpec` launches workers as processes on the local machine. This runs the same distributed code path without a cluster:

```python
from dsbo.worker.spec import LocalWorkerSpec

LOCAL = LocalWorkerSpec(cores=4, processes=1, memory="16GB", min_worker=1, max_worker=2)
```

It accepts `cores`, `processes`, and `memory`; `LocalWorkerSpec.default_local()` derives a specification from the machine's actual cores and memory.

## Scaling

Two methods set the number of workers per specification:

```python
CPU_CONFIG = CPU_CONFIG.adapt(min_worker=1, max_worker=8)   # elastic: scale between bounds
GPU_CONFIG = GPU_CONFIG.scale(4)                            # static: exactly four workers
```

`.scale()` fixes the worker count; `.adapt()` grows and shrinks the pool between its bounds as tasks queue up and workers go idle. Scaling down only retires idle workers — a task in execution is never interrupted.

Workers are reused across evaluations: a live worker serves tasks from every `run()` and `run_batch()` call, so queue wait and startup cost are paid once per worker, not per task or per run. How long a worker lives depends on the scaling mode: `.scale()` keeps it until the context closes, `.adapt()` retires and resubmits it with demand.
