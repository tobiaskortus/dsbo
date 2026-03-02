from __future__ import annotations

from abc import ABC
from contextlib import nullcontext
from typing import Optional
from functools import wraps
from dsbo.modules.compile import dryrunnable, is_dry_run
from dsbo.modules.dsl import ModuleDslMixin  # noqa: F401
from dsbo.logger import WorkflowTaskLogger
from dsbo.utils.io import AbstractFilePath, with_remote_workdir
from dsbo.worker.config import SlurmWorkerConfig


class Module(ModuleDslMixin, ABC):
    def __init__(self):
        self._registered_files = {}
        self.i: int | None = None
        self.resources = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        raw = cls.__dict__.get("__call__")
        if raw is None or getattr(raw, "_dsbo_dry_wrapped", False): 
            return

        def ensure_dict_return(fn):
            @wraps(fn)
            def _inner(*args, **kwargs):
                res = fn(*args, **kwargs)
                return {} if res is None else res
            return _inner

        wrapped = with_remote_workdir(raw)
        wrapped = dryrunnable(wrapped)
        wrapped = ensure_dict_return(wrapped)
        setattr(wrapped, "_dsbo_dry_wrapped", True)
        setattr(cls, "__call__", wrapped)

    @dryrunnable
    @with_remote_workdir
    def __call__(self, *args, **kwargs) -> Optional[dict[str, float]]:
        pass

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        if isinstance(value, AbstractFilePath):
            self._registered_files[name] = value

    def __repr__(self):
        return self.name

    @property
    def name(self) -> str:
        suffix = f"[{self.i}]" if self.i is not None else ""
        return f"{type(self).__name__}{suffix}"

    @with_remote_workdir
    def run(self, *args, **kwargs):
        from dsbo.utils.io import Workdir
        with WorkflowTaskLogger(self) if not is_dry_run() else nullcontext():
            with Workdir.use_file_suffix(str(self.i)) if self.i is not None else nullcontext():
                return self(*args, **kwargs)

    def to(self, resource: SlurmWorkerConfig):
        self.resources = resource
        return self
                
    @property
    def registered_files(self) -> dict[str, AbstractFilePath]:
        return self._registered_files
