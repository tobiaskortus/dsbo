from __future__ import annotations
import dask
import contextvars
from abc import abstractmethod
from dask.delayed import Delayed, delayed
from dask.distributed import Client
from contextlib import contextmanager
from dask_jobqueue import SLURMCluster
from dsbo.utils.io import get_workdir, get_file_suffix
from dask_heterogeneous.slurm import HeterogeneousSlurmCluster
from typing import Any, Generator, Optional, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from dsbo.worker.config import SlurmWorkerConfig
    from dsbo.workflow import Workflow
    from dsbo.modules import Module

dask.config.set({
    "distributed.worker.memory.target": 0.85,
    "distributed.worker.memory.spill": 0.90,
    "distributed.worker.memory.pause": 0.95,
    "distributed.worker.memory.terminate": 0.98,
})

class DaskWorkerBase:
    def __init__(self):
        pass

    @abstractmethod
    def compute(self, objs, **kwargs):
        pass

    @abstractmethod
    def schedule(self, module: Module, design: dict, dependencies: Iterable = (), priority: int | None = None) -> Delayed:
        pass

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


class AdaptiveHeterogeneousDaskSlurmWorker(DaskWorkerBase):
    def __init__(self, job_specs: Iterable[SlurmWorkerConfig], logdir: str = "logs"):
        super().__init__()
        self.logdir = logdir

        spec_dict = {
            spec.resource_key: {
                "cores": spec.job_cpu,
                "queue": spec.partition,
                "memory": spec.memory,
                "walltime": spec.walltime,
                "job_extra_directives": get_extra_args(spec)[1],
                "processes": spec.processes,
                "job_cpu": spec.job_cpu,
                "resources": spec.resources,
            }
            for spec in job_specs
        }

        self.cluster = HeterogeneousSlurmCluster(spec=spec_dict, log_directory=logdir)
        self.cluster.adapt({spec_key: (0, 5) for spec_key in spec_dict.keys()})

        self.client = Client(self.cluster)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()
        self.cluster.close()

    def compute(self, objs, **kwargs):
        return self.client.compute(objs, **kwargs)

    def schedule(self, module: Module, design: dict, dependencies: Iterable = (), priority: int | None = None) -> Delayed:
        workdir = get_workdir().workdir_root
        suffix = get_file_suffix()

        resources = getattr(module.resources, "resources", None)
    
        def _module_wrapper(module, design_arg, *deps):
            return module.run(design=design_arg, workdir=workdir, suffix=suffix)

        with dask.annotate(resources=(resources or {}), priority=priority):
            delayed_obj = delayed(_module_wrapper, name=module.name)(module, design, *dependencies)

        return delayed_obj

    @classmethod
    def from_workflow(cls, workflow: Workflow, logdir: str = "logs") -> "AdaptiveHeterogeneousDaskSlurmWorker":
        job_specs = workflow.get_unique_workers()
        return cls(job_specs=job_specs, logdir=logdir)
