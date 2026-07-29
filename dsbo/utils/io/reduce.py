from __future__ import annotations

import os
import uuid
from filelock import FileLock

from dsbo.config import get_active_file_backend


def reduce_registered_file(sources: list[str], destination: str) -> None:
    get_active_file_backend().reduce(sources, destination)


class LazyReductionResolver:
    def __init__(self, sources: list[str], destination: str, context: str | None = None):
        self._sources = list(sources)
        self._destination = destination
        self._context = context
        self._resolve_id = uuid.uuid4().hex

    def resolve(self) -> str:
        lock_path = self._destination + ".lock"
        with FileLock(lock_path):
            if self._is_current():
                return self._destination
            try:
                reduce_registered_file(self._sources, self._destination)
                self._write_marker()
            except Exception as exc:
                if self._context:
                    raise type(exc)(f"{self._context}: {exc}") from exc
                raise
        return self._destination

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
