from collections.abc import Sequence
import os
import shutil
from abc import ABC, abstractmethod
from typing import List, Optional

from dsbo.utils import ContainerWrapperBase

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
    def simulate(self, design: dict) -> None:
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
