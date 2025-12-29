from typing import NamedTuple, Optional

class SlurmWorkerConfig(NamedTuple):
    partition: Optional[str] = None
    walltime: str = "48:00:00"
    memory: str = "48GB"
    processes: int = 1
    job_cpu: int = 10
    gpu_resources: Optional[str] = None

DEFAULT_GPU_CONFIG = SlurmWorkerConfig(gpu_resources="gpu:1")
DEFAULT_CPU_CONFIG = SlurmWorkerConfig()