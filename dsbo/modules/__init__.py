from abc import ABC
from typing import Optional
from functools import wraps
from dsbo.modules.compile import dryrunnable
from dsbo.utils.io import AbstractFilePath, with_remote_workdir

class Module(ABC):
    def __init__(self):
        self._registered_files = {}

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
                
    @property
    def registered_files(self) -> dict[str, AbstractFilePath]:
        return self._registered_files

