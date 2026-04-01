
import dask.distributed
from contextlib import nullcontext
from abc import ABC, abstractmethod
from typing import List, Optional, Union
from dask.distributed import Client, LocalCluster

from dsbo.modules import Module
from dsbo.container import ContainerWrapperBase
from dsbo.transforms import SimulationTransform
from dsbo.utils.io import Workdir, get_file_suffix, get_subdir, get_workdir, with_remote_workdir


class SimulationBase(Module):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __call__(self, design: dict, **kwargs) -> dict:
        self.simulate(design, **kwargs)

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
