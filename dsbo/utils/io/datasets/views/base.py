from __future__ import annotations

import numpy as np
from typing import Callable
from abc import ABC, abstractmethod


STORE_FLAT = "flat"
STORE_COLUMN = "column"
STORE_ROW = "row"


def _leading_count(shape: tuple[int, ...]) -> int:
    return 1 if shape == () else shape[0]


def _stack_statistics(values: list[tuple[int, object, object] | None]) -> tuple[int, np.ndarray, np.ndarray] | None:
    if not values or any(value is None for value in values):
        return None
    counts, means, squared_deviations = zip(*values, strict=True)
    if len(set(counts)) != 1:
        return None
    return counts[0], np.asarray(means), np.asarray(squared_deviations)


class RowSource:
    def __init__(self, num_rows: int, read: Callable[[int | slice], "np.ndarray"]):
        self._num_rows = num_rows
        self._read = read

    @property
    def num_rows(self) -> int:
        return self._num_rows

    def __getitem__(self, index: int | slice) -> "np.ndarray":
        return self._read(index)


class DatasetView(ABC):
    @property
    def keys(self) -> tuple[str, ...]:
        return self._keys

    @abstractmethod
    def shape(self, key: str) -> tuple[int, ...]:
        pass

    @abstractmethod
    def read(self, key: str, index: int | slice):
        pass

    def read_many(self, keys: list[str], index: int | slice) -> np.ndarray:
        values = [self.read(key, index) for key in keys]
        return np.asarray(values) if isinstance(index, int) else np.stack(values, axis=-1)

    def read_indices(self, keys: list[str], indices: list[int]) -> np.ndarray:
        return np.stack([self.read_many(keys, index) for index in indices], axis=0)

    def statistics(self, keys: list[str]) -> tuple[int, np.ndarray, np.ndarray] | None:
        return None

    @abstractmethod
    def partition_sizes(self, key: str) -> tuple[int, ...]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    def __enter__(self) -> "DatasetView":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class ColumnStoreDatasetView(DatasetView):
    def __init__(self, group_name: str = "primal", keys: tuple[str, ...] = ()):
        self._group_name = group_name
        self._keys = tuple(keys)

    def read_many(self, keys: list[str], index: int | slice) -> dict[str, np.ndarray]:
        return {key: np.asarray(self.read(key, index)) for key in keys}

    def read_indices(self, keys: list[str], indices: list[int]) -> dict[str, np.ndarray]:
        return {key: self._read_indices(key, indices) for key in keys}

    def statistics(self, keys: list[str]) -> dict[str, tuple[int, np.ndarray, np.ndarray] | None]:
        return {key: self._statistics(key) for key in keys}

    @abstractmethod
    def _read_indices(self, key: str, indices: list[int]) -> np.ndarray:
        ...

    @abstractmethod
    def _statistics(self, key: str) -> tuple[int, np.ndarray, np.ndarray] | None:
        ...


class RowStoreDatasetView(DatasetView):
    def __init__(self, keys: tuple[str, ...], num_rows: int):
        self._keys = keys
        self._column_indices = {key: index for index, key in enumerate(keys)}
        self.num_rows = num_rows

    def _columns(self, keys: list[str]) -> list[int]:
        return [self._column_indices[key] for key in keys]

    def shape(self, key: str) -> tuple[int, ...]:
        self._column_indices[key]
        return (self.num_rows,)

    def read(self, key: str, index: int | slice):
        return self.read_many([key], index)[..., 0]

    @abstractmethod
    def read_many(self, keys: list[str], index: int | slice) -> np.ndarray:
        ...
