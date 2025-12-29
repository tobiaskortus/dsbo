
import dask.distributed
from contextlib import nullcontext
from abc import ABC, abstractmethod
from typing import List, Optional, Union
from dask.distributed import Client, LocalCluster
from sqlalchemy import null

from dsbo.modules import Module
from dsbo.container import ContainerWrapperBase
from dsbo.transforms import SimulationTransform
from dsbo.utils.io import Workdir, get_file_suffix, get_workdir, with_remote_workdir


class SimulationBase(Module):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @with_remote_workdir
    def __call__(self, design: dict, **kwargs) -> None:
        return self.simulate(design, **kwargs)

    @abstractmethod
    def simulate(self, design: dict) -> None:
        pass


class SimulationContainerWrapperBase(SimulationBase, ContainerWrapperBase):
    def __init__(self, container_path: str, container_extra_flags: Optional[str] = ""):
        SimulationBase.__init__(self)
        ContainerWrapperBase.__init__(self, container_path, container_extra_flags)

    def call_container(self, arguments: list, is_exec: bool = False) -> None:
        if is_exec: super().exec_container(arguments)
        else: super().run_container(arguments)


class DaskSimulationWrapper(SimulationBase):
    def __init__(self, base_simulation: SimulationBase, n_simulations, n_workers: int, 
                threads_per_worker: int, merge_function: SimulationTransform, init_on_demand: bool = False) -> None:
        
        super().__init__()

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
    
    def simulate(self, design: dict) -> None:
        if self.init_on_demand:
            self.initialize_local_cluster(self.n_workers, self.threads_per_worker)

        #Initialize input/output dirs before submitting jobs
        #self.base_simulation.input_dir()
        #self.base_simulation.output_dir()
        
        futures: List[dask.distributed.Future] = []
        for i in range(self.n_simulations):
            with Workdir.use_file_suffix(str(i)):
                print(f"Submitting simulation {i+1}/{self.n_simulations}")
                workdir = get_workdir()
                suffix = get_file_suffix()
                future = self.client.submit(self.base_simulation, design, workdir=workdir.workdir_root, suffix=suffix)
                futures.append(future)
            
        self.client.gather(futures)
        self.merge_function(design)

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