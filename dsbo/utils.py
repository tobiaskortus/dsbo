from enum import Enum
import logging
import subprocess
from typing import List, Optional

def which_python():
    import shutil
    for cmd in ("python3", "python"):
        if shutil.which(cmd):
            return cmd
    raise Exception("No python interpreter found.")

def python_script_args(script_path: str, args: List[str]):
    return [which_python(), script_path] + args

class ContainerType(Enum):
    SINGULARITY = "singularity"
    APPTAINER = "apptainer"
    DOCKER = "docker"

class ContainerWrapperBase:
    def __init__(self, container_path, container_extra_flags: Optional[str], 
                 cuda_enabled: bool = False, verbose: bool = False, **kwargs) -> None:
        
        super().__init__(**kwargs)
        self._container_path = container_path
        self._container_extra_flags = container_extra_flags
        self._container_type = ContainerType.SINGULARITY
        self._logger = logging.getLogger(__name__)
        self._verbose = verbose

        if cuda_enabled:
            if self._container_type in {ContainerType.SINGULARITY, ContainerType.APPTAINER}:
                extra_flags = "--nv"
                if self._container_extra_flags:
                    self._container_extra_flags += f" {extra_flags}"
                else:
                    self._container_extra_flags = extra_flags
            else:
                raise NotImplementedError("CUDA support for Docker containers is not implemented yet.")

    def __construct_cmd(self, args, is_exec: bool = False) -> List[str]:
        if self._container_type == ContainerType.DOCKER:
            raise NotImplementedError("Docker container support is not implemented yet.")
        
        cmd = ["singularity" if self._container_type == ContainerType.SINGULARITY else "apptainer", 
               "exec" if is_exec else "run"]

        cmd += self.container_extra_flags.split() if self.container_extra_flags else []
        cmd += [self.container_path]
        cmd += [*args]
        return cmd

    def run_container(self, args: List[str]) -> None:
        try:
            cmd = self.__construct_cmd(args, is_exec=False)
            logging.debug(f"Running container with command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=not self._verbose, text=True)
        except subprocess.CalledProcessError as e:
            if not self._verbose:
                print("STDOUT:", e.stdout)
                print("STDERR:", e.stderr)
            raise

    def exec_container(self, args: List[str]) -> None:
        try:
            cmd = self.__construct_cmd(args, is_exec=True)
            logging.debug(f"Running container with command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=not self._verbose, text=True)
        except subprocess.CalledProcessError as e:
            if not self._verbose:
                print("STDOUT:", e.stdout)
                print("STDERR:", e.stderr)
            raise

    @property
    def container_path(self) -> str:
        return self._container_path
    
    @property
    def container_extra_flags(self) -> Optional[str]:
        return self._container_extra_flags

        