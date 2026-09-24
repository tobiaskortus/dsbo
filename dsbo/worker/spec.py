from __future__ import annotations

import re
import multiprocessing as mp
from dataclasses import dataclass, replace
from abc import ABC, abstractmethod
from dask_jobqueue.local import LocalJob
from dask_jobqueue.slurm import SLURMJob

import psutil

def get_num_gpus(spec: WorkerSpec) -> int:
    """Extract the number of GPUs from the gpu_resources SLURM string."""
    if spec.gpu_resources is None:
        return 0
    else:
        try:
            resource_str = spec.gpu_resources
            return int(resource_str.split(":")[-1])
        except Exception as e:
            raise ValueError(f"Invalid gpu_resources format: {spec.gpu_resources}") from e


def get_extra_args(spec: WorkerSpec) -> tuple[list[str], list[str]]:
    """Generate extra arguments for the Dask SlurmCluster based on the worker configuration."""
    resource_name = spec.resource_key
    if spec.gpu_resources is None:
        return ([f"--resources {resource_name}=1"], [])
    else:
        num_gpus = get_num_gpus(spec)
        return ([f"--resources {resource_name}={num_gpus}"],
                [f"--gres={spec.gpu_resources}"])


def sanitize(name: str) -> str:
    tok = re.sub(r"[^A-Za-z0-9]", "_", name)
    tok = re.sub(r"_+", "_", tok).strip("_")
    if not tok or not tok[0].isalpha():
        tok = f"r_{tok}"
    return tok


class WorkerSpec(ABC):
    @property
    @abstractmethod
    def spec_key(self) -> str:
        pass

    @property
    def resource_key(self) -> str:
        return self.spec_key

    @property
    def resources(self) -> dict[str, int]:
        return {self.resource_key: 1}

    def _scope_suffix(self) -> str:
        """Disambiguate specs that share hardware but differ in scheduling scope.

        Two specs with the same cores/memory but different scaling bounds (or SLURM
        accounts that collide after sanitising) must not share a cluster spec key,
        otherwise their worker pools are silently merged.
        """
        suffix = ""
        if self.min_worker is not None or self.max_worker is not None:
            suffix += f"__workers_{self.min_worker}_{self.max_worker}"
        account = getattr(self, "account", None)
        if account is not None:
            suffix += f"__account_{account.encode().hex()}"
        return suffix

    @classmethod
    def default_local(cls) -> LocalWorkerSpec:
        def _format_memory(value: float) -> str:
            for unit in ("", "K", "M", "G"):
                if abs(value) < 1024:
                    return f"{value:.0f}{unit}B"
                value /= 1024
            return f"{value:.0f}TB"

        cores = mp.cpu_count()
        memory = _format_memory(psutil.virtual_memory().total)
        return LocalWorkerSpec(cores=cores, processes=1, memory=memory)
    
    def adapt(self, min_worker: int, max_worker: int):
        return replace(self, min_worker=min_worker, max_worker=max_worker)

    def scale(self, n_worker: int):
        return replace(self, min_worker=n_worker, max_worker=n_worker)


@dataclass(frozen=True)
class LocalWorkerSpec(WorkerSpec):
    cores: int
    processes: int
    memory: str

    min_worker: int | None = None
    max_worker: int | None = None

    @property
    def spec_key(self) -> str:
        return sanitize(f"local_{self.cores}c_{self.processes}p_{self.memory}") + self._scope_suffix()
    

    def get_spec_dict(self) -> dict[str, any]:
        return {self.spec_key: {
            "cls": LocalJob,
            "cores": self.cores,
            "processes": self.processes,
            "memory": self.memory,
            "resources": self.resources,
        }}


@dataclass(frozen=True)
class SLURMWorkerSpec(WorkerSpec):
    cores: int
    processes: int
    memory: str
    walltime: str
    account: str | None = None
    partition: str | None = None
    gpu_resources: str | None = None

    min_worker: int | None = None
    max_worker: int | None = None

    @property
    def spec_key(self) -> str:
        raw = "_".join([
            self.partition or "",
            self.walltime,
            self.memory,
            str(self.processes),
            str(self.cores),
            str(self.gpu_resources or ""),
        ])
        return sanitize(raw) + self._scope_suffix()
    
    def get_spec_dict(self) -> dict[str, any]:
        return {self.spec_key: {
            "cls": SLURMJob,
            "cores": self.cores,
            "job_cpu": self.cores,
            "processes": self.processes,
            "memory": self.memory,
            "walltime": self.walltime,
            "account": self.account,
            "queue": self.partition,
            "job_extra_directives": get_extra_args(self)[1],
            "resources": self.resources,
        }}

