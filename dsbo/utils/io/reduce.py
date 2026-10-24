from __future__ import annotations

import hashlib
import os
import uuid
from filelock import FileLock
from abc import ABC, abstractmethod
from typing import TypeAlias, Union

ResolvableSource: TypeAlias = Union[str, "LazyReductionResolver"]

def resolve_value(value: ResolvableSource) -> str:
    #str is passthrough for single files.
    return value if isinstance(value, str) else value.resolve()


def reduction_sources(value: ResolvableSource | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        #str is passthrough for single files.
        return [value]
    return list(value.sources)


def make_reduction_resolver(
    sources: list[str],
    destination: str,
    resolver_cls: LazyReductionResolver,
    generation: str | uuid.UUID,
    context: str | None = None,
):
    if resolver_cls.passthrough_single and len(sources) == 1:
        return sources[0]

    resolve_repr = repr((resolver_cls.__name__, generation, sources, destination)).encode("utf-8")
    resolve_id = "reduce:" + hashlib.sha256(resolve_repr).hexdigest()
    return resolver_cls(sources, destination, context, resolve_id=resolve_id)


class LazyReductionResolver(ABC):
    passthrough_single = False

    def __init__(self, sources: list[str], destination: str, context: str | None = None, *, resolve_id: str | None = None):
        self._sources = list(sources)
        self._destination = destination
        self._context = context
        self._resolve_id = resolve_id or uuid.uuid4().hex

    @property
    def sources(self) -> list[str]:
        return list(self._sources)

    def resolve(self) -> str:
        lock_path = self._destination + ".lock"
        with FileLock(lock_path):
            if self._is_current():
                return self._destination
            try:
                # An old plan's marker must never authenticate a newly published file.
                try:
                    os.unlink(self._marker_path)
                except FileNotFoundError:
                    pass
                self._materialize()
                self._write_marker()
            except Exception as exc:
                if self._context:
                    raise type(exc)(f"{self._context}: {exc}") from exc
                raise
        return self._destination

    @abstractmethod
    def _materialize(self) -> None:
        """Write the reduced result to ``self._destination``."""

    def _is_current(self) -> bool:
        if not os.path.isfile(self._destination):
            return False
        return self._read_marker() == self._resolve_id

    @property
    def _marker_path(self) -> str:
        return self._destination + ".resolve_id"

    def _read_marker(self) -> str | None:
        try:
            with open(self._marker_path) as f:
                return f.read().strip()
        except (FileNotFoundError, OSError):
            return None

    def _write_marker(self) -> None:
        with open(self._marker_path, "w") as f:
            f.write(self._resolve_id)


class FileConcatenationResolver(LazyReductionResolver):
    """Forward path: concatenate registered files into one destination view."""
    passthrough_single = True
    def _materialize(self) -> None:
        from dsbo.config import get_active_file_backend

        get_active_file_backend().reduce(self._sources, self._destination)


class GradientAggregationResolver(LazyReductionResolver):
    """Backward path: aggregate write-once contribution files."""
    def _materialize(self) -> None:
        from dsbo.config import get_active_file_backend

        get_active_file_backend().aggregate_gradients(self._sources, self._destination)
