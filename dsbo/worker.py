import os
from dataclasses import dataclass
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from typing import Optional

from distributed import Future

from dsbo.simulation import DaskSimulationWrapper
from dsbo.workflow import Workflow

@dataclass
class SlurmWorkerConfig:
    walltime: str = "1:00:00"
    memory: str = "32GB"
    processes: int = 1
    job_cpu: int = 10
    gpu_resources: Optional[str] = None


DEFAULT_GPU_CONFIG = SlurmWorkerConfig(gpu_resources="gpu:1")
DEFAULT_CPU_CONFIG = SlurmWorkerConfig()

def get_num_gpus(worker_config: SlurmWorkerConfig) -> int:
    if worker_config.gpu_resources is None:
        return 0
    else:
        try:
            resource_str = worker_config.gpu_resources 
            return int(resource_str.split(":")[-1])
        except Exception as e:
            raise ValueError(f"Invalid gpu_resources format: {worker_config.gpu_resources}") from e


class DaskSlurmWorker:
    def __init__(self, n_worker: int, worker_config: SlurmWorkerConfig, logdir: str = "logs"): 
        worker_extra_args = [] if worker_config.gpu_resources is None\
                               else [f"--resources GPU={get_num_gpus(worker_config)}"]

        job_extra_directives = [] if worker_config.gpu_resources is None\
                               else [f"--gres={worker_config.gpu_resources}"]
        
        self.n_worker = n_worker
        
        self.cluster = SLURMCluster(cores=1, memory=worker_config.memory, 
                    walltime=worker_config.walltime, worker_extra_args=worker_extra_args,
                    job_extra_directives=job_extra_directives, log_directory=logdir, 
                    processes=1, job_cpu=8)
        
        self.cluster.scale(n_worker)
        self.client = Client(self.cluster)
        
        # Wait for at least one worker to be available
        self.client.wait_for_workers(n_workers=1)

    def submit(self, workflow: Workflow, design: dict, trial_id: Optional[int] = None) -> Future: 
        if isinstance(workflow.simulation, DaskSimulationWrapper):
            assert workflow.simulation.init_on_demand, "DaskSimulationWrapper inside DaskSlurmWorker must be initialized with init_on_demand=True"
            
        future = self.client.submit(workflow.run, design, trial_id)
        return future

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()
        self.cluster.close()
