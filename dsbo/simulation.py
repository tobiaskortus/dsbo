import os
import shutil
from uuid import uuid4
import dask.distributed
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import List, Optional
from dask.distributed import Client, LocalCluster

from dsbo.utils import ContainerWrapperBase
from dsbo.transforms import SimulationTransform


class SimulationBase(ABC):
    def __init__(self, workdir: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._workdir = workdir
        self._simulation_dir = f"{workdir}/output"

        if not os.path.exists(self._workdir):
            os.makedirs(self._workdir, exist_ok=False)
        if not os.path.exists(self._simulation_dir):
            os.makedirs(self._simulation_dir, exist_ok=False)

    @abstractmethod
    def simulate(self, design: dict, suffix: Optional[str] = None) -> None:
        pass

    def remove_simulation_files(self) -> None:
        if os.path.exists(self._simulation_dir):
            shutil.rmtree(self._simulation_dir, ignore_errors=True)

    @property
    def simulation_dir(self) -> str:
        return self._simulation_dir
    
    @property
    def workdir(self) -> str:
        return self._workdir


class SimulationContainerWrapperBase(SimulationBase, ContainerWrapperBase):
    def __init__(self, workdir: str, container_path: str, container_extra_flags: Optional[str] = ""):
        super().__init__(workdir, container_path=container_path, container_extra_flags=container_extra_flags)

    def call_container(self, arguments: list, is_exec: bool = False) -> None:
        if is_exec: super().exec_container(arguments)
        else: super().run_container(arguments)


class DaskSimulationWrapper(SimulationBase):
    def __init__(self, base_simulation: SimulationBase, n_simulations, n_workers: int, 
                threads_per_worker: int, merge_function: SimulationTransform) -> None:
        
        super().__init__(workdir=base_simulation.workdir)

        assert n_simulations > 1, "n_simulations must be greater than 1"
        assert n_workers >= 1, "n_workers must be at least 1"
        assert threads_per_worker >= 1, "threads_per_worker must be at least 1"
        assert not isinstance(base_simulation, DaskSimulationWrapper), "Nested DaskSimulationWrapper are not allowed."

        self.base_simulation = base_simulation
        self.merge_function = merge_function
        self.n_simulations = n_simulations

        self.cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker,
                                    dashboard_address=None, processes=True)
        self.client = Client(self.cluster)

    def simulate(self, design_parameters: dict) -> None:
        futures: List[dask.distributed.Future] = []
        for i in range(self.n_simulations):
            future = self.client.submit(self.base_simulation.simulate, design_parameters, str(i))
            futures.append(future)
            
        self.client.gather(futures)
        self.merge_function(self.base_simulation.simulation_dir)
    
    def close(self):
        if hasattr(self, 'client') and self.client:
            self.client.close()
        if hasattr(self, 'cluster') and self.cluster:
            self.cluster.close()
    
    def __del__(self):
        try:
            self.close()
        except:
            pass