from __future__ import annotations

from dsbo.utils.io.workdir import AbstractFilePath


class LazyShape(tuple):
    """A shape that resolves lazily from a produced file.

    Dimensions like module-surrogate output/conditional shapes are unknown
    until the module's training batch writes its output files, so they are
    deferred here: the value behaves like a tuple but only resolves the file's
    shape on first access, and can also be pinned explicitly via :meth:`set`
    (e.g. when restoring from a saved model state).
    """
    _fpath: AbstractFilePath
    _resolved: tuple[int, ...] | None

    def __new__(cls, fpath: AbstractFilePath):
        instance = super().__new__(cls)
        instance._fpath = fpath
        instance._resolved = None
        return instance

    def resolve(self, path: str):
        if self._resolved is None:
            from dsbo.config import get_active_file_backend
            self._resolved = get_active_file_backend().shape_from_path(path)[1:]

    def set(self, shape: tuple[int, ...]):
        self._resolved = tuple(shape)

    def _resolve(self):
        if self._resolved is not None:
            return
        raise RuntimeError(
            f"Cannot resolve shape for '{self._fpath.filename}' before the "
            "module training batch has produced its output files."
        )

    def __getitem__(self, idx):  # type: ignore[override]
        self._resolve()
        return self._resolved[idx]

    def __len__(self):
        self._resolve()
        return len(self._resolved)

    def __iter__(self):
        self._resolve()
        return iter(self._resolved)

    def __repr__(self):
        if self._resolved is None:
            return "LazyShape(unresolved)"
        return repr(self._resolved)
