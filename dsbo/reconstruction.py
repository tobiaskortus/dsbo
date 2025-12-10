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
        self._trial_id = None
        self._workdir = workdir
    
    def __init_subclass__(cls, **kwargs):
        """Automatically wrap simulate() to handle trial_id and directory setup."""
        super().__init_subclass__(**kwargs)
        
        # Store original implementation
        cls._train_impl = cls.train
        cls._validate_impl = cls.validate

        def train_wrapper(self, simulation_dir: str, trial_id: Optional[int] = None) -> None:
            self._trial_id = trial_id
            if not os.path.exists(self.reconstruction_dir):
                os.makedirs(self.reconstruction_dir, exist_ok=True)
            return type(self)._train_impl(self, simulation_dir)
        
        def validate_wrapper(self, simulation_dir: str, trial_id: Optional[int] = None) -> Sequence[float]:
            self._trial_id = trial_id
            if not os.path.exists(self.reconstruction_dir):
                os.makedirs(self.reconstruction_dir, exist_ok=True)
            return type(self)._validate_impl(self, simulation_dir)
        

        train_wrapper.__name__ = 'simulate'
        train_wrapper.__qualname__ = f'{cls.__qualname__}.simulate'
        cls.train = train_wrapper

        validate_wrapper.__name__ = 'validate'
        validate_wrapper.__qualname__ = f'{cls.__qualname__}.validate'
        cls.validate = validate_wrapper

    @abstractmethod
    def train(self, simulation_dir: str, trial_id: Optional[int] = None) -> None:
        pass
    
    @abstractmethod
    def validate(self, simulation_dir: str, trial_id: Optional[int] = None) -> Sequence[float]:
        pass

    def remove_reconstruction_files(self) -> None:
        if os.path.exists(self.reconstruction_dir):
            shutil.rmtree(self.reconstruction_dir, ignore_errors=True)

    @property
    def reconstruction_dir(self) -> str:
        print(self._trial_id)
        return os.path.join(self.workdir, "reconstruction")
    
    @property
    def workdir(self) -> str:
        if self._trial_id is None:
            return self._workdir

        return os.path.join(self._workdir, f"trial_{self._trial_id}")


class ReconstructionContainerWrapper(ReconstructionBase, ContainerWrapperBase):
    def __init__(self, workdir: str, container_path: str, container_extra_flags: Optional[str] = None, cuda_enabled: bool = True) -> None:
        super().__init__(workdir=workdir, container_path=container_path, 
                         container_extra_flags=container_extra_flags, cuda_enabled=cuda_enabled)

    def call_container(self, arguments: list, is_exec: bool = False) -> None:
        if is_exec: super().exec_container(arguments)
        else: super().run_container(arguments)



    