import os
import shutil
import contextvars
from enum import Enum
from contextlib import contextmanager
from typing import Callable, Generator, Generator, List, Optional, Set, Union


class DefaultPaths(Enum):
    SIMULATION = "simulation"
    RECONSTRUCTION = "reconstruction"
    LOGS = "logs"

class Workdir(dict):
    _current_workdir = contextvars.ContextVar("current_workdir", default=None)
    _registered_paths: Set[str] = set()

    def __init__(self, workdir_root: str, stages: Optional[List[str]] = None,
                 timestamp: bool = False, error_if_exists: bool = True):
        
        super().__init__()
        self.workdir_root = workdir_root
        self.error_if_exists = error_if_exists

        if stages is not None:
            for stage in stages:
                stage_dir = os.path.join(self.workdir_root, stage)
                self[stage] = stage_dir

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    @classmethod
    @contextmanager
    def create(cls, workdir_root: str, stages: Optional[List[str]] = None, 
                    error_if_exists: bool = True, keep: bool = False) -> Generator["Workdir", None, None]:
        
        if stages is None:
            stages = cls.get_registered_paths()

        registry = cls(workdir_root, stages, error_if_exists)
        token = cls._current_workdir.set(registry)
        
        try:
            for path in registry.values():
                if error_if_exists and os.path.exists(path):
                    raise FileExistsError(f"Directory {path} already exists.")
                os.makedirs(path, exist_ok=True)

            yield
        finally:
            if not keep: registry.cleanup()
            cls._current_workdir.reset(token)
        
        return registry
    

    @classmethod
    @contextmanager
    def load(cls, workdir_root: str) -> Generator["Workdir", None, None]:
        if not os.path.exists(workdir_root):
            raise FileNotFoundError(f"Workdir {workdir_root} does not exist.")
        
        registry = cls(workdir_root)
        token = cls._current_workdir.set(registry)
        try:
            for stage in os.listdir(workdir_root):
                path = os.path.join(workdir_root, stage)
                if os.path.isdir(path):
                    registry[stage] = path
            yield
        finally:
            cls._current_workdir.reset(token)
        

    def cleanup(self, stages: Optional[List[str]] = None):
        if stages:
            for stage in stages:
                path = self.get(stage)
                if path and os.path.exists(path):
                    shutil.rmtree(path)
        elif os.path.exists(self.workdir_root):
            shutil.rmtree(self.workdir_root)

    @classmethod
    def register(cls, path: Union[str, DefaultPaths]) -> Callable[[], str]:
        class WorkdirPath:
            def __call__(self, *args, **kwds):
                workdir = cls._current_workdir.get()
                if workdir is None:
                    print("Warning: No current workdir set. Using cwd instead.")
                    return os.path.join(os.getcwd())
                
                workdir_path = os.path.join(workdir.root, path)
                if not os.path.exists(workdir_path):
                    print(f"Path {workdir_path} has not been registered. Creating it once.")
                    os.makedirs(workdir_path, exist_ok=True)

                return workdir_path

            @property
            def name(self) -> str:
                return path
            
        if isinstance(path, DefaultPaths):
            path = path.value
            
        cls._registered_paths.add(path)
        return WorkdirPath()
    
    @classmethod
    def get_registered_paths(cls) -> Set[str]:
        return cls._registered_paths

    @property
    def stages(self) -> List[str]:
        return list(self.keys())
    
    @property
    def root(self) -> str:
        return self.workdir_root
    

def get_workdir():
    workdir = Workdir._current_workdir.get()
    if workdir is None:
        raise RuntimeError("No current workdir set.")
    return workdir

def get_workdir_path(stage: str) -> str:
    workdir = get_workdir()       
    path =  workdir.get(stage, workdir.root)
    if path is workdir.root:
        print(f"Warning: Stage '{stage}' not found in workdir registry. Using root path instead.")
    return path