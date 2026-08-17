from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Generator

from dsbo.utils.io.storage.autograd import AutogradFileInterface
from dsbo.utils.io.storage.hdf5 import HDF5File

_config_var: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "dsbo_config", default={}
)


class Config:
    def get(self, key: str, default: Any = None) -> Any:
        return _config_var.get().get(key, default)

    def set(self, key: str, value: Any) -> None:
        cfg = _config_var.get()
        new = {**cfg, key: value}
        _config_var.set(new)

    @contextmanager
    def use(self, **overrides: Any) -> Generator[None, None, None]:
        parent = _config_var.get()
        child = {**parent, **overrides}
        token = _config_var.set(child)
        try:
            yield
        finally:
            _config_var.reset(token)

    def reset(self, key: str) -> None:
        cfg = _config_var.get()
        if key in cfg:
            new = {k: v for k, v in cfg.items() if k != key}
            _config_var.set(new)

    def state(self) -> dict[str, Any]:
        return _config_var.get()

    def _set(self, state: dict[str, Any]) -> None:
        _config_var.set(state)



config = Config()


def get_active_file_backend() -> type[AutogradFileInterface]:
    backend = config.get("file_backend")
    if backend is not None:
        return backend

    return HDF5File
