import os
import shutil
from typing import Optional
from abc import ABC, abstractmethod
from collections.abc import Sequence

from dsbo.container import ContainerWrapperBase
from dsbo.utils.io import with_remote_workdir

class ReconstructionBase(ABC):
    """Abstract base class for reconstruction modules."""
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @with_remote_workdir
    def __call__(self, *args, **kwds):
        self.train()
        return self.validate()
    
    @abstractmethod
    def train(self) -> None:
        pass
    
    @abstractmethod
    def validate(self) -> Sequence[float]:
        pass


class ReconstructionContainerWrapper(ReconstructionBase, ContainerWrapperBase):
    def __init__(self, container_path: str, container_extra_flags: Optional[str] = None, cuda_enabled: bool = True) -> None:
        super().__init__(container_path=container_path, 
                         container_extra_flags=container_extra_flags, cuda_enabled=cuda_enabled)

    def call_container(self, arguments: list, is_exec: bool = False) -> None:
        if is_exec: super().exec_container(arguments)
        else: super().run_container(arguments)



    