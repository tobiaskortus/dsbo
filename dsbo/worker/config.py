from typing import NamedTuple, Optional
import re

#TODO rename to SlurmJobSpec
class WorkerSpec(NamedTuple):
    partition: Optional[str] = None
    walltime: str = "48:00:00"
    memory: str = "48GB"
    processes: int = 1
    job_cpu: int = 10
    gpu_resources: Optional[str] = None

    @property
    def spec_key(self) -> str:
        name = "_".join([
            str(self.partition),
            self.walltime,
            self.memory,
            str(self.processes),
            str(self.job_cpu),
            str(self.gpu_resources)
        ])
        return name

    @property
    def resource_key(self) -> str:
        raw = self.spec_key
        tok = re.sub(r"[^A-Za-z0-9]", "_", raw)
        tok = re.sub(r"_+", "_", tok)
        tok = tok.strip("_")
        if not tok or not tok[0].isalpha():
            tok = f"r_{tok}"
        return tok
    
    @property
    def resources(self) -> dict:
        return { self.resource_key: 1 }
    

DEFAULT_GPU_CONFIG = WorkerSpec(gpu_resources="gpu:1")
DEFAULT_CPU_CONFIG = WorkerSpec()