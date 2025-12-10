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
        self._trial_id = None

    def __init_subclass__(cls, **kwargs):
        """Automatically wrap simulate() to handle trial_id and directory setup."""
        super().__init_subclass__(**kwargs)
        
        # Store original implementation
        original_simulate = cls.simulate
        cls._simulate_impl = original_simulate
        
        # Create a new simulate method that wraps the original
        def simulate_wrapper(self, design: dict, suffix: Optional[str] = None, trial_id: Optional[int] = None) -> None:
            self._trial_id = trial_id
            if not os.path.exists(self.simulation_dir):
                os.makedirs(self.simulation_dir, exist_ok=True)
            # Call the stored implementation
            return type(self)._simulate_impl(self, design, suffix, trial_id)
        
        # Set proper metadata for serialization
        simulate_wrapper.__name__ = 'simulate'
        simulate_wrapper.__qualname__ = f'{cls.__qualname__}.simulate'
        cls.simulate = simulate_wrapper
    
    @abstractmethod
    def simulate(self, design: dict, suffix: Optional[str] = None, trial_id: Optional[int] = None) -> None:
        pass

    def remove_simulation_files(self) -> None:
        if os.path.exists(self.simulation_dir):
            shutil.rmtree(self.simulation_dir, ignore_errors=True)

    @property
    def simulation_dir(self) -> str:
        return os.path.join(self.workdir, "output")

    @property
    def workdir(self) -> str:
        if self._trial_id is None:
            return self._workdir

        return os.path.join(self._workdir, f"trial_{self._trial_id}")


class SimulationContainerWrapperBase(SimulationBase, ContainerWrapperBase):
    def __init__(self, workdir: str, container_path: str, container_extra_flags: Optional[str] = ""):
        super().__init__(workdir, container_path=container_path, container_extra_flags=container_extra_flags)

    def call_container(self, arguments: list, is_exec: bool = False) -> None:
        if is_exec: super().exec_container(arguments)
        else: super().run_container(arguments)


class DaskSimulationWrapper(SimulationBase):
    def __init__(self, base_simulation: SimulationBase, n_simulations, n_workers: int, 
                threads_per_worker: int, merge_function: SimulationTransform, init_on_demand: bool = False) -> None:
        
        super().__init__(workdir=base_simulation.workdir)

        assert n_simulations > 1, "n_simulations must be greater than 1"
        assert n_workers >= 1, "n_workers must be at least 1"
        assert threads_per_worker >= 1, "threads_per_worker must be at least 1"
        assert not isinstance(base_simulation, DaskSimulationWrapper), "Nested DaskSimulationWrapper are not allowed."

        self.base_simulation = base_simulation
        self.merge_function = merge_function
        self.n_simulations = n_simulations
        self.init_on_demand = init_on_demand

        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker

        if not init_on_demand:
            self.initialize_local_cluster(self.n_workers, self.threads_per_worker)

    def initialize_local_cluster(self, n_workers: int, threads_per_worker: int) -> None:
        assert (not hasattr(self, 'cluster') or self.cluster is None and 
                not hasattr(self, 'client') or self.client is None), "Dask local cluster is already initialized."
        
        self.cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker,
                                    dashboard_address=None, processes=False)
        self.client = Client(self.cluster)

    def simulate(self, design: dict, suffix: Optional[str] = None, trial_id: Optional[int] = None) -> None:
        if self.init_on_demand:
            self.initialize_local_cluster(self.n_workers, self.threads_per_worker)

        futures: List[dask.distributed.Future] = []
        for i in range(self.n_simulations):
            future = self.client.submit(self.base_simulation.simulate, design, str(i), trial_id)
            futures.append(future)
            
        self.client.gather(futures)
        self.merge_function(self.simulation_dir)

        if self.init_on_demand:
            self.close()
    
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