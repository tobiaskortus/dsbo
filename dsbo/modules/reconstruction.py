import os
import shutil
from typing import Optional
from abc import ABC, abstractmethod
from collections.abc import Sequence

from dsbo.modules import Module
from dsbo.container import ContainerWrapperBase

class ReconstructionBase(Module):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, design: dict, **kwargs) -> Optional[dict[str, float]]:
        self.train()
        return self.validate()
    
    @abstractmethod
    def train(self) -> None:
        pass
    
    @abstractmethod
    def validate(self) -> Optional[dict[str, float]]:
        pass


class ReconstructionContainerWrapper(ReconstructionBase, ContainerWrapperBase):
    def __init__(self, container_path: str, container_extra_flags: Optional[str] = None, cuda_enabled: bool = True) -> None:
        ReconstructionBase.__init__(self)
        ContainerWrapperBase.__init__(self, container_path=container_path,
                                      container_extra_flags=container_extra_flags,
                                      cuda_enabled=cuda_enabled)

    def call_container(self, arguments: list, is_exec: bool = False) -> None:
        if is_exec: super().exec_container(arguments)
        else: super().run_container(arguments)



    