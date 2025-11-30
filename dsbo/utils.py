import logging
import subprocess
from typing import List, Optional

def python_script_args(script_path: str, args: List[str]):
    return ["python3", script_path] + args

class ContainerWrapperBase:
    def __init__(self, container_path, container_extra_flags: Optional[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self._container_path = container_path
        self._container_extra_flags = container_extra_flags
        self._logger = logging.getLogger(__name__)

    def __construct_cmd(self, args, is_exec: bool = False) -> List[str]:
        cmd = ["singularity", "exec" if is_exec else "run"]
        cmd += self.container_extra_flags.split() if self.container_extra_flags else []
        cmd += [self.container_path]
        cmd += [*args]
        return cmd

    def run_container(self, args: List[str]) -> None:
        cmd = self.__construct_cmd(args, is_exec=False)
        logging.debug(f"Running container with command: {' '.join(cmd)}")
        subprocess.run(cmd,  check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) #
    def exec_container(self, args: List[str]) -> None:
        cmd = self.__construct_cmd(args, is_exec=True)
        logging.debug(f"Running container with command: {' '.join(cmd)}")
        subprocess.run(cmd,  check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) #

    @property
    def container_path(self) -> str:
        return self._container_path
    
    @property
    def container_extra_flags(self) -> Optional[str]:
        return self._container_extra_flags

        