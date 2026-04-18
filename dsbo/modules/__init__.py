from __future__ import annotations

from abc import ABC
from contextlib import nullcontext
from typing import Optional
from functools import wraps
import weakref
from dsbo.utils.io import Workdir, get_workdir
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
        if raw is None or getattr(raw, "_dsbo_wrapped", False):
            return

        def ensure_dict_return(fn):
            @wraps(fn)
            def _inner(*args, **kwargs):
                res = fn(*args, **kwargs)
                return {} if res is None else res
            return _inner

        wrapped = dryrunnable(raw)
        wrapped = ensure_dict_return(wrapped)
        setattr(wrapped, "_dsbo_wrapped", True)
        setattr(cls, "__call__", wrapped)

    @dryrunnable
    def __call__(self, design: dict, **kwargs) -> Optional[dict[str, float]]:
        pass

    def __setattr__(self, name, value):
        if not hasattr(self, "_registered_files") and name != "_registered_files":
            raise RuntimeError(
                "Subclasses of Module must call super().__init__() in their __init__ before setting attributes."
            )

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
    def run(self, design: dict, **kwargs):
        with nullcontext() if (is_dry_run() or get_workdir() is None) else WorkflowTaskLogger(self):
            with Workdir.use_file_suffix(str(self.i)) if self.i is not None else nullcontext():
                return self(design=design)

    def to(self, resource: SlurmWorkerConfig):
        self.resources = resource
        return self

    @property
    def registered_files(self) -> dict[str, AbstractFilePath]:
        return self._registered_files


class dynamic_submodule:
    def __init__(self, wrapped_fn):
        self._wrapped_fn = wrapped_fn
        self._submodule_cache = weakref.WeakKeyDictionary()

    def __get__(self, obj, *args, **kwargs):
        if obj is None:
            return self._wrapped_fn
        
        if not isinstance(obj, Module):
            raise TypeError(
                f"@dynamic_submodule must be accessed on a `Module` instance; got {type(obj).__name__}."
            )

        class Submodule(Module):
            def __init__(self, parent, fn):
                object.__setattr__(self, "_parent", parent)
                object.__setattr__(self, "fn", fn)

            def __setattr__(self, name, value):
                if name in ("_parent", "fn") or name.startswith("_"):
                    object.__setattr__(self, name, value)
                else:
                    setattr(self._parent, name, value)

            def __call__(self, design, **kwargs):
                try:
                    return self.fn(self._parent, design, **kwargs)
                except Exception as e:
                    tb = e.__traceback__
                    if tb is not None:
                        tb = tb.tb_next
                    raise e.with_traceback(tb)

            def __getattr__(self, name):
                return getattr(self._parent, name)
            
            @property
            def name(self) -> str:
                suffix = f"[{self.i}]" if self.i is not None else ""
                return f"{type(self._parent).__name__}.{self.fn.__name__}{suffix}"

        submodule = self._submodule_cache.get(obj)
        if submodule is not None:
            return submodule

        submodule = Submodule(obj, self._wrapped_fn)
        self._submodule_cache[obj] = submodule

        return submodule