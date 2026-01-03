import logging
import subprocess
from enum import Enum
from typing import Callable, List, Optional
from dsbo.logger import WorkflowTaskLogger

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


def subprocess_run(cmd: List[str], 
                   stdout_callback: Callable[[str], None], 
                   stderr_callback: Callable[[str], None]) -> int:
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    from threading import Thread

    def stream(pipe, callback):
        for line in iter(pipe.readline, ''):
            callback(line)
        pipe.close()

    threads = [
        Thread(target=stream, args=(process.stdout, stdout_callback)),
        Thread(target=stream, args=(process.stderr, stderr_callback)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return process.wait()


class ContainerWrapperBase:
    def __init__(self, container_path, container_extra_flags: Optional[str], 
                 cuda_enabled: bool = False, verbose: bool = True, **kwargs) -> None:
        
        super().__init__(**kwargs)
        self._container_path = container_path
        self._container_extra_flags = container_extra_flags
        self._container_type = ContainerType.SINGULARITY
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
        cmd = self.__construct_cmd(args, is_exec=False)
        
        logger = WorkflowTaskLogger.get_current()
        
        if logger is not None:
            logger.log(f"Running container: {self.container_path} with args: {' '.join(cmd)}")
            stdout = logger.get_log_callback(logging.INFO)
            stderr = logger.get_log_callback(logging.ERROR)
        else:
            stdout = lambda x: print(x)
            stderr = lambda x: print(x)
        
        process_returncode = subprocess_run(cmd, stdout, stderr)
        if process_returncode != 0:
            logger.log(f"Container process failed with return code {process_returncode}", level=logging.ERROR)

    def exec_container(self, args: List[str]) -> None:
        cmd = self.__construct_cmd(args, is_exec=True)

        logger = WorkflowTaskLogger.get_current()
        
        if logger is not None:
            logger.log(f"Running container: {self.container_path} with args: {' '.join(cmd)}")
            stdout = logger.get_log_callback(logging.INFO)
            stderr = logger.get_log_callback(logging.ERROR)
        else:
            stdout = lambda x: print(x)
            stderr = lambda x: print(x)
        
        process_returncode = subprocess_run(cmd, stdout, stderr)
        if process_returncode != 0:
            logger.log(f"Container process failed with return code {process_returncode}", log_type=logging.ERROR)

    @property
    def container_path(self) -> str:
        return self._container_path
    
    @property
    def container_extra_flags(self) -> Optional[str]:
        return self._container_extra_flags

        