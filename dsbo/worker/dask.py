import os
import dask
import threading
import contextvars
from distributed import Future
from dask.distributed import Client
from contextlib import contextmanager
from dask_jobqueue import SLURMCluster
from typing import Any, Generator, Optional

from dsbo.modules.simulation import DaskSimulationWrapper
from dsbo.utils.io import get_file_suffix, get_workdir
from dsbo.workflow import WorkflowTask
from dsbo.worker.config import SlurmWorkerConfig


dask.config.set({
    "distributed.worker.memory.target": 0.85,
    "distributed.worker.memory.spill": 0.90,
    "distributed.worker.memory.pause": 0.95,
    "distributed.worker.memory.terminate": 0.98,
})

class DaskWorkerBase:
    def __init__(self):
        pass

    def submit(self, task: WorkflowTask, design: dict, priority: int = 0) -> Future:
        raise NotImplementedError("submit method must be implemented by subclasses.")


class WorkerContext:
    _current_worker = contextvars.ContextVar("current_worker", default=None)

    @classmethod
    @contextmanager
    def use_worker(cls, worker: DaskWorkerBase) -> Generator[None, Any, None]:
        assert isinstance(worker, DaskWorkerBase), "Worker must be an instance of DaskWorkerBase"
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
    if worker_config.gpu_resources is None:
        return ([], [])
    else:
        num_gpus = get_num_gpus(worker_config)
        return ([f"--resources GPU={num_gpus}"],
                [f"--gres={worker_config.gpu_resources}"])
    

class DaskSlurmWorker(DaskWorkerBase):
    def __init__(self, n_worker: int, worker_config: SlurmWorkerConfig, adaptive: bool = False, logdir: str = "logs"): 
        worker_extra_args, job_extra_directives = get_extra_args(worker_config)
        self.n_worker = n_worker
        self.adaptive = adaptive
        
        self.cluster = SLURMCluster(cores=1, queue=worker_config.partition, memory=worker_config.memory, 
                    walltime=worker_config.walltime, worker_extra_args=worker_extra_args,
                    job_extra_directives=job_extra_directives, log_directory=logdir, 
                    processes=1, job_cpu=worker_config.job_cpu)
        
        if not self.adaptive:
            self.cluster.scale(n_worker)
        else:
            self.cluster.adapt(minimum=0, maximum=n_worker)

        self.client = Client(self.cluster)

    def submit(self, task: WorkflowTask, design: dict, priority: int = 0) -> Future: 
        # TODO: fix with fused tasks
        #if isinstance(task.task, DaskSimulationWrapper):
        #    assert task.task.init_on_demand, "DaskSimulationWrapper inside DaskSlurmWorker must be initialized with init_on_demand=True"
        
        workdir = get_workdir()
        suffix = get_file_suffix()
        return self.client.submit(task.run, design, 
                                  workdir=workdir.workdir_root, suffix=suffix, 
                                  priority=priority)

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()
        self.cluster.close()

class AdaptiveHeterogenousDaskSlurmWorker(DaskWorkerBase):
    def __init__(self, max_workers: int, logdir: str = "logs"):
        self.logdir = logdir
        self.max_workers = max_workers
        self.workers: dict[SlurmWorkerConfig, DaskSlurmWorker] = {}
        self._lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with self._lock:
            for worker in self.workers.values():
                worker.client.close()
                worker.cluster.close()

    def submit(self, task: WorkflowTask, design: dict, priority: int = 0) -> Future:
        if task.worker_config is None:
            raise ValueError("Task does not have a worker_config defined.")
        
        with self._lock:
            if task.worker_config not in self.workers:
                self.worker_type = "cpu" if task.worker_config.gpu_resources is None else "gpu"
                self.workers[task.worker_config] = DaskSlurmWorker(
                    n_worker=self.max_workers, worker_config=task.worker_config, adaptive=True,
                    logdir=os.path.join(self.logdir, self.worker_type))
            worker = self.workers[task.worker_config]

        return worker.submit(task, design=design, priority=priority)
        