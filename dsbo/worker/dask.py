from __future__ import annotations
import os
import dask
import threading
import contextvars
from distributed import Future
from dask.distributed import Client
from contextlib import contextmanager
from dask_jobqueue import SLURMCluster
from dask_heterogeneous.slurm import HeterogeneousSlurmCluster
from typing import Any, Generator, Optional, Iterable, TYPE_CHECKING
from dsbo.utils.io import get_file_suffix, get_workdir

if TYPE_CHECKING:
    from dsbo.worker.config import SlurmWorkerConfig
    from dsbo.workflow import WorkflowTask, WorkflowTaskGraph

dask.config.set({
    "distributed.worker.memory.target": 0.85,
    "distributed.worker.memory.spill": 0.90,
    "distributed.worker.memory.pause": 0.95,
    "distributed.worker.memory.terminate": 0.98,
})

class DaskWorkerBase:
    def __init__(self):
        pass
    def submit(self, task: WorkflowTask, design: dict, priority: int = 0,
               dependencies: Optional[Iterable[Future]] = None, **client_kwargs) -> Future:
        raise NotImplementedError("submit method must be implemented by subclasses.")


class WorkerContext:
    _current_worker = contextvars.ContextVar("current_worker", default=None)

    @classmethod
    @contextmanager
    def use_worker(cls, worker: Optional[DaskWorkerBase]) -> Generator[None, Any, None]:
        assert worker == None or isinstance(worker, DaskWorkerBase), "Worker must be an instance of DaskWorkerBase"
        token = cls._current_worker.set(worker)
        try:
            yield
        finally:
            cls._current_worker.reset(token)

    @classmethod
    def get_current_worker(cls) -> Optional[DaskWorkerBase]:
        return cls._current_worker.get()


def get_num_gpus(worker_config: SlurmWorkerConfig) -> int:
    """Extract the number of GPUs from the gpu_resources SLURM string."""
    if worker_config.gpu_resources is None:
        return 0
    else:
        try:
            resource_str = worker_config.gpu_resources
            return int(resource_str.split(":")[-1])
        except Exception as e:
            raise ValueError(f"Invalid gpu_resources format: {worker_config.gpu_resources}") from e


def get_extra_args(worker_config: SlurmWorkerConfig) -> tuple[list[str], list[str]]:
    """Generate extra arguments for the Dask SlurmCluster based on the worker configuration."""
    resource_name = worker_config.resource_key
    if worker_config.gpu_resources is None:
        return ([f"--resources {resource_name}=1"], [])
    else:
        num_gpus = get_num_gpus(worker_config)
        return ([f"--resources {resource_name}={num_gpus}"],
                [f"--gres={worker_config.gpu_resources}"])
    

#class DaskSlurmWorker(DaskWorkerBase):
#    def __init__(self, n_worker: int, worker_config: SlurmWorkerConfig, adaptive: bool = False, logdir: str = "logs"):
#        worker_extra_args, job_extra_directives = get_extra_args(worker_config)
#        self.n_worker = n_worker
#        self.adaptive = adaptive
#
#        self.cluster = SLURMCluster(cores=1, queue=worker_config.partition, memory=worker_config.memory,
#                    walltime=worker_config.walltime, worker_extra_args=worker_extra_args,
#                    job_extra_directives=job_extra_directives, log_directory=logdir,
#                    processes=1, job_cpu=worker_config.job_cpu)
#
#        if not self.adaptive:
#            self.cluster.scale(n_worker)
#        else:
#            self.cluster.adapt(minimum=0, maximum=n_worker)
#
#        self.client = Client(self.cluster)
#
#    def submit(self, task: "WorkflowTask", design: dict, priority: int = 0,
#               dependencies: Optional[Iterable[Future]] = None, **client_kwargs) -> Future:
#        """Submit a task using the Dask client.
#
#        Predecessor futures passed in `dependencies` are forwarded as extra
#        positional arguments to `task.run` so that Dask builds the proper
#        dependency edges automatically.
#        """
#        workdir = get_workdir()
#
#        # Base kwargs that should always be forwarded to the remote invocation
#        submit_kwargs: dict = {
#            "workdir": workdir.workdir_root,
#            "suffix": get_file_suffix(),
#            "priority": priority,
#        }
#
#        # Merge any caller-specified client kwargs (e.g. resources)
#        submit_kwargs.update(client_kwargs or {})
#
#        # If the task has a worker_config with resources, prefer that unless
#        # caller explicitly provided `resources` in client_kwargs
#        if (
#            "resources" not in submit_kwargs
#            and getattr(task, "worker_config", None)
#            and getattr(task.worker_config, "resources", None)
#        ):
#            submit_kwargs["resources"] = task.worker_config.resources
#
#        deps = list(dependencies) if dependencies is not None else []
#
#        # Pass predecessor futures as positional args so Dask treats them as
#        # dependencies of this submission.
#        return self.client.submit(task.run, design, *deps, **submit_kwargs)
#
#    def __enter__(self):
#        return self
#
#    def __exit__(self, exc_type, exc_val, exc_tb):
#        self.client.close()
#        self.cluster.close()


class AdaptiveHeterogeneousDaskSlurmWorker(DaskWorkerBase):
    def __init__(self, job_specs: Iterable[SlurmWorkerConfig], logdir: str = "logs"):
        super().__init__()
        self.logdir = logdir

        spec_dict = {
            spec.resource_key: {
                "cores": 1,
                "queue": spec.partition,
                "memory": spec.memory,
                "walltime": spec.walltime,
                "worker_extra_args": get_extra_args(spec)[0],
                "job_extra_directives": get_extra_args(spec)[1],
                "processes": 1,
                "job_cpu": spec.job_cpu,
                "resources": spec.resources,
            }
            for spec in job_specs
        }

        self.cluster = HeterogeneousSlurmCluster(spec=spec_dict, log_directory=logdir)
        #TODO: change adapt call to specify min/max workers per spec_key based with reasonable values provided
        self.cluster.adapt({spec_key: (0, 10) for spec_key in spec_dict.keys()})

        self.client = Client(self.cluster)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()
        self.cluster.close()

    @classmethod
    def from_workflow(cls, workflow: WorkflowTaskGraph, logdir: str = "logs") -> "AdaptiveHeterogeneousDaskSlurmWorker":
        job_specs = workflow.get_unique_workers()
        return cls(job_specs=job_specs, logdir=logdir)

    def submit(self, task: WorkflowTask, design: dict, priority: int = 0,
               dependencies: Optional[Iterable[Future]] = None, **client_kwargs) -> Future:
        
        workdir = get_workdir()

        submit_kwargs: dict = {
            "workdir": workdir.workdir_root, 
            "suffix": get_file_suffix(), 
            "priority": priority,
            "resources": task.worker_config.resources}
        
        submit_kwargs.update(client_kwargs or {})
        deps = list(dependencies) if dependencies is not None else []
        return self.client.submit(task.run, design, *deps, **submit_kwargs)
