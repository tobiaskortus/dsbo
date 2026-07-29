from __future__ import annotations

from collections import OrderedDict

_MODULE_TYPES: OrderedDict[str, tuple[type, str]] = OrderedDict()


def _default_import(cls: type) -> str:
    module = cls.__module__
    qualname = cls.__qualname__
    top_name = qualname.split(".", 1)[0] if "." in qualname else qualname
    return f"from {module} import {top_name}"


def register_cli_module(name: str, import_path: str | None = None):
    def decorator(cls):
        imp = import_path if import_path is not None else _default_import(cls)
        _MODULE_TYPES[name] = (cls, imp)
        return cls

    return decorator


def get_registered_types() -> dict[str, tuple[type, str]]:
    return dict(_MODULE_TYPES)


def get_type_import(name: str) -> str | None:
    entry = _MODULE_TYPES.get(name)
    return entry[1] if entry else None
