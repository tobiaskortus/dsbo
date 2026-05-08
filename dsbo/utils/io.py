import functools
import os
from pathlib import Path
import shutil
import contextvars
from enum import Enum
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Generator, Generator, List, Optional, Set, Union

def _is_dry_run():
    from dsbo.modules.compile import is_dry_run
    return is_dry_run()


class DefaultPaths(str, Enum):
    SIMULATION = "simulation"
    RECONSTRUCTION = "reconstruction"
    LOGS = "logs"


class AbstractPath:
    def __init__(self, path: str):
        self._path = path

    def __call__(self):
        if _is_dry_run():
            return None
        
        root = Workdir._current_workdir.get()
        subdir_parts = Workdir._current_subdir.get()

        if root is None:
            print("Warning: No workdir root is set. Using cwd instead.")
            base = os.path.join(os.getcwd(), *subdir_parts) if subdir_parts else os.getcwd()
        else:
            base = os.path.join(root.root, *subdir_parts) if subdir_parts else root.root

        workdir_path = os.path.join(base, self._path) if self._path else base
        if not os.path.exists(workdir_path):
            os.makedirs(workdir_path, exist_ok=True)
        
        return workdir_path
    
    def __eq__(self, other : "AbstractPath"):
        if not isinstance(other, type(self)):
            return False
        
        return self._path == other._path

    @property
    def pathname(self) -> str:
        return self._path
    

class AbstractFilePath:
    def __init__(self, path: AbstractPath, filename: str, unique: bool = True):
        self._filename = filename
        self._unique = unique
        self._path = path

    def __call__(self):
        parts = Workdir._current_file_context.get()
        if parts and self._unique:
            name, ext = os.path.splitext(self._filename)
            filename = f"{name}_{'_'.join(parts)}{ext}"
        else:
            filename = self._filename
        return os.path.join(self._path(), filename)
    

    def __eq__(self, other: "AbstractFilePath"):
        if not isinstance(other, type(self)):
            return False

        return (self._path == other._path and 
                self._filename == other._filename and 
                self._unique == other._unique)

    @property
    def filename(self) -> str:
        return self._filename
    
    @property
    def pathname(self) -> str:
        return self._path.pathname


class AbstractFilePatternPath:
    def __init__(self, path: AbstractPath, filename_pattern: str):
        self._filename_pattern = filename_pattern
        self._path = path

    def __call__(self):
        return Path(self._path()).glob(self._filename_pattern)

    @property
    def filename_pattern(self) -> str:
        return self._filename_pattern
    
    @property
    def pathname(self) -> str:
        return self._path.pathname

class Workdir(dict):
    _current_workdir = contextvars.ContextVar("current_workdir", default=None)
    _current_file_context = contextvars.ContextVar("current_file_context", default=())
    _current_subdir = contextvars.ContextVar("current_subdir", default=())
    _registered_paths: Set[str] = set()

    def __init__(self, workdir_root: str, paths: Optional[List[str]] = None,
                 error_if_exists: bool = True):
        
        super().__init__()
        self.workdir_root = workdir_root
        self.error_if_exists = error_if_exists

        if paths is not None:
            for path in paths:
                path_dir = os.path.join(self.workdir_root, path)
                self[path] = path_dir

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)
        
    #def __enter__(self):
    #    return self
    
    #def __exit__(self, exc_type, exc_val, exc_tb):
    #    self.cleanup()
    
    @classmethod
    @contextmanager
    def create(cls, workdir_root: str, paths: Optional[List[str]] = None, 
                    error_if_exists: bool = True, keep: bool = False) -> Generator["Workdir", None, None]:
        
        if paths is None:
            paths = cls.get_registered_paths()

        if error_if_exists and os.path.exists(workdir_root):
            raise FileExistsError(f"Directory {workdir_root} already exists.")

        registry = cls(workdir_root, paths, error_if_exists)
        token = cls._current_workdir.set(registry)

        try:
            os.makedirs(workdir_root, exist_ok=True)
            for path in registry.values():
                if error_if_exists and os.path.exists(path):
                    raise FileExistsError(f"Directory {path} already exists.")
                os.makedirs(path, exist_ok=True)

            yield registry
        finally:
            cls._current_workdir.reset(token)
            if not keep:
                registry.cleanup()
    
    @classmethod
    @contextmanager
    def create_subdir(cls, subdir: str) -> Generator[None, None, None]:
        current = cls._current_subdir.get()
        token = cls._current_subdir.set(current + (subdir,))
        try:
            yield
        finally:
            cls._current_subdir.reset(token)

    @classmethod
    @contextmanager
    def use_file_suffix(cls, suffix: str) -> Generator[None, None, None]:
        current = cls._current_file_context.get()
        token = cls._current_file_context.set(current + (suffix,))
        try:
            yield
        finally:
            cls._current_file_context.reset(token)
    

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
    def register(cls, path: Optional[Union[str, DefaultPaths]]) -> Callable[[], str]:
        if isinstance(path, DefaultPaths):
            path = path.value

        if path is not None:
            cls._registered_paths.add(path)

        return AbstractPath(path)
    
    @classmethod
    def register_file(cls, path: Optional[Union[str, DefaultPaths]], filename: str, unique_filename: bool = True) -> AbstractFilePath:
        workdir_path = Workdir.register(path)
        return AbstractFilePath(workdir_path, filename, unique_filename)
    

    @classmethod
    def register_file_pattern(cls, path: str, filename_pattern: str) -> str:
        workdir_path = Workdir.register(path)
        return AbstractFilePatternPath(workdir_path, filename_pattern)
    
    @classmethod
    def get_registered_paths(cls) -> Set[str]:
        return cls._registered_paths

    @property
    def stages(self) -> List[str]:
        return list(self.keys())
    
    @property
    def root(self) -> str:
        return self.workdir_root
    

def get_workdir() -> Workdir:
    workdir = Workdir._current_workdir.get()
    #if workdir is None:
    #    raise RuntimeError("No current workdir set.")
    return workdir


def get_subdir() -> Optional[str]:
    parts = Workdir._current_subdir.get()
    return os.path.join(*parts) if parts else None


def get_file_suffix() -> Optional[str]:
    parts = Workdir._current_file_context.get()
    return "_".join(parts) if parts else None


def with_remote_workdir(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs) -> Any:
        if _is_dry_run():
            return func(self, *args, **kwargs)
        
        workdir = kwargs.pop("workdir", None)
        suffix = kwargs.pop("suffix", None)
        subdir = kwargs.pop("subdir", None)
        with Workdir.load(workdir) if workdir else nullcontext():
            with Workdir.create_subdir(subdir) if subdir else nullcontext():
                with Workdir.use_file_suffix(suffix) if suffix else nullcontext():
                    return func(self, *args, **kwargs)
    return wrapper