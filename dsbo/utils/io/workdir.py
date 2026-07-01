from __future__ import annotations

import os
import shutil
import functools
import contextvars
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Generator, Iterator, List, Optional, Set, Union


def _is_dry_run():
    from dsbo.utils.compile import is_dry_run
    return is_dry_run()


class DefaultPaths(str, Enum):
    SIMULATION = "simulation"
    RECONSTRUCTION = "reconstruction"
    LOGS = "logs"

@dataclass(frozen=True)
class PathInfo:
    root: Optional[str]
    subdirs: tuple[str, ...]
    rel_path: Optional[str]
    suffix_parts: tuple[str, ...]

    @property
    def subdir_path(self) -> Optional[str]:
        return os.path.join(*self.subdirs) if self.subdirs else None

    @property
    def suffix(self) -> Optional[str]:
        return "_".join(self.suffix_parts) if self.suffix_parts else None

class AbstractPath:
    def __init__(self, path: str):
        self._path = path

    def __call__(self) -> Optional[str]:
        if _is_dry_run():
            return None

        workdir = get_workdir()
        subdir_parts = get_subdir_parts()
        base = os.path.join(workdir.root, *subdir_parts) if subdir_parts else workdir.root

        workdir_path = os.path.join(base, self._path) if self._path else base
        if not os.path.exists(workdir_path):
            os.makedirs(workdir_path, exist_ok=True)

        info = PathInfo(root=workdir.root, subdirs=subdir_parts, rel_path=self._path, suffix_parts=())
        Workdir._register_path(workdir_path, info)
        return workdir_path

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self._path == other._path

    def __hash__(self) -> int:
        return hash(self._path)

    @property
    def pathname(self) -> str:
        return self._path

class AbstractFilePath:
    def __init__(self, path: AbstractPath, filename: str, unique: bool = True):
        self._filename = filename
        self._unique = unique
        self._path = path

    def __call__(self) -> Optional[str]:
        parts = Workdir._current_file_context.get()
        if parts and self._unique:
            name, ext = os.path.splitext(self._filename)
            filename = f"{name}_{'_'.join(parts)}{ext}"
        else:
            filename = self._filename

        dir_path = self._path()
        if dir_path is None:
            return None
        full_path = os.path.join(dir_path, filename)

        dir_info = Workdir.get_path_info(dir_path)
        info = PathInfo(
            root=dir_info.root if dir_info else None,
            subdirs=dir_info.subdirs if dir_info else (),
            rel_path=self._path.pathname,
            suffix_parts=parts,
        )
        Workdir._register_path(full_path, info)
        return full_path

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        return (self._path == other._path and 
                self._filename == other._filename and 
                self._unique == other._unique)

    def __hash__(self) -> int:
        return hash((self._path, self._filename, self._unique))

    def __repr__(self) -> str:
        name, ext = os.path.splitext(self._filename)
        suffix = "_<suffix>" if self._unique else ""
        return f"{type(self).__name__}({self._path.pathname}/{name}{suffix}{ext})"

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def pathname(self) -> str:
        return self._path.pathname



class Workdir(dict):
    _current_workdir = contextvars.ContextVar("current_workdir", default=None)
    _current_file_context = contextvars.ContextVar("current_file_context", default=())
    _current_subdir = contextvars.ContextVar("current_subdir", default=())
    _registered_paths: Set[str] = set()
    _path_registry: dict[str, PathInfo] = {}
    _registry_by_root: dict[str, list[str]] = {}  # root → [path, …]
    _cwd_workdir: Workdir | None = None

    def __init__(self, workdir_root: str, paths: Optional[List[str]] = None,
                 error_if_exists: bool = True):

        super().__init__()
        self.workdir_root = workdir_root
        self.error_if_exists = error_if_exists

        if paths is not None:
            for path in paths:
                path_dir = os.path.join(self.workdir_root, path)
                self[path] = path_dir

    def __getattr__(self, item: str):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item) from None

    @classmethod
    def _register_path(cls, path: str, info: PathInfo) -> None:
        cls._path_registry[path] = info
        root = info.root or ""
        cls._registry_by_root.setdefault(root, []).append(path)

    @classmethod
    def _clear_registry_for_root(cls, root: str) -> None:
        paths = cls._registry_by_root.pop(root, ())
        for p in paths:
            cls._path_registry.pop(p, None)

    @classmethod
    @contextmanager
    def create(cls, workdir_root: str, paths: Optional[List[str]] = None,
               error_if_exists: bool = True, keep: bool = False) -> Generator[Workdir, None, None]:

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
                cls._clear_registry_for_root(workdir_root)
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
    def load(cls, workdir_root: str) -> Generator[Workdir, None, None]:
        if not os.path.exists(workdir_root):
            raise FileNotFoundError(f"Workdir {workdir_root} does not exist.")

        registry = cls(workdir_root)
        token = cls._current_workdir.set(registry)
        try:
            for stage in os.listdir(workdir_root):
                path = os.path.join(workdir_root, stage)
                if os.path.isdir(path):
                    registry[stage] = path
            yield registry
        finally:
            cls._current_workdir.reset(token)
            cls._clear_registry_for_root(workdir_root)

    def cleanup(self, stages: Optional[List[str]] = None):
        if stages:
            for stage in stages:
                path = self.get(stage)
                if path and os.path.exists(path):
                    shutil.rmtree(path)
        elif os.path.exists(self.workdir_root):
            shutil.rmtree(self.workdir_root)

    @classmethod
    def register(cls, path: Optional[Union[str, DefaultPaths]]) -> AbstractPath:
        if isinstance(path, DefaultPaths):
            path = path.value
        if path is not None:
            cls._registered_paths.add(path)
        return AbstractPath(path) if path is not None else AbstractPath("")

    @classmethod
    def register_file(cls, path: Optional[Union[str, DefaultPaths]], filename: str,
                      unique_filename: bool = True) -> AbstractFilePath:
        workdir_path = Workdir.register(path)
        return AbstractFilePath(workdir_path, filename, unique_filename)

    @classmethod
    def get_registered_paths(cls) -> Set[str]:
        return cls._registered_paths

    @classmethod
    def get_path_info(cls, path: str) -> Optional[PathInfo]:
        return cls._path_registry.get(path)

    @property
    def stages(self) -> List[str]:
        return list(self.keys())

    @property
    def root(self) -> str:
        return self.workdir_root
    
    @classmethod
    def _singleton_cwd(cls) -> Workdir:
        if cls._cwd_workdir is None:
            cls._cwd_workdir = cls(os.getcwd())
        return cls._cwd_workdir


def get_workdir() -> Workdir:
    wd = Workdir._current_workdir.get()
    if wd is None:
        import warnings
        warnings.warn("No Workdir context active — using cwd as workdir root.")
        wd = Workdir._singleton_cwd()
        Workdir._current_workdir.set(wd)
    return wd


def get_subdir_parts() -> str:
    return Workdir._current_subdir.get()


def get_subdir() -> str:
    parts = get_subdir_parts()
    return os.path.join(*parts) if parts else ""


def get_file_suffix() -> str:
    parts = Workdir._current_file_context.get()
    return "_".join(parts) if parts else ""


def with_remote_workdir(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if _is_dry_run():
            return func(*args, **kwargs)

        workdir = kwargs.pop("workdir", None)
        suffix = kwargs.pop("suffix", None)
        subdir = kwargs.pop("subdir", None)

        with Workdir.load(workdir) if workdir else nullcontext():
            with Workdir.create_subdir(subdir) if subdir else nullcontext():
                with Workdir.use_file_suffix(suffix) if suffix else nullcontext():
                    return func(*args, **kwargs)
                
    return wrapper


@dataclass(frozen=True)
class PathAnnotation:
    path: AbstractFilePath

    def __call__(self) -> Optional[str]:
        return self.path()

class Input(PathAnnotation):
    pass

class Output(PathAnnotation):
    pass