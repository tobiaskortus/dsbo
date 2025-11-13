import os
import shutil
from typing import Optional
from abc import ABC, abstractmethod
from collections.abc import Sequence

from dsbo.utils import ContainerWrapperBase

class ReconstructionBase(ABC):
    """Abstract base class for reconstruction modules."""
    def __init__(self, workdir, **kwargs) -> None:
        super().__init__(**kwargs)
        self._workdir = workdir
        self._reconstruction_dir = f"{workdir}/reconstruction"

        if not os.path.exists(self._workdir):
            os.makedirs(self._workdir, exist_ok=False)
        if not os.path.exists(self._reconstruction_dir):
            os.makedirs(self._reconstruction_dir, exist_ok=False)

    @abstractmethod
    def train(self, simulation_dir: str) -> None:
        pass
    
    @abstractmethod
    def validate(self, simulation_dir: str) -> Sequence[float]:
        pass

    def remove_reconstruction_files(self) -> None:
        if os.path.exists(self._reconstruction_dir):
            shutil.rmtree(self._reconstruction_dir, ignore_errors=True)


class ReconstructionContainerWrapper(ReconstructionBase, ContainerWrapperBase):
    def __init__(self, workdir: str, container_path: str, container_extra_flags: Optional[str]) -> None:
        super().__init__(workdir=workdir, container_path=container_path, container_extra_flags=container_extra_flags)

    def call_container(self, arguments: list, is_exec: bool = False) -> None:
        if is_exec: super().exec_container(arguments)
        else: super().run_container(arguments)



    