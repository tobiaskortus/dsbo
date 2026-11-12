from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dsbo.utils.io.storage import _hdf5
from dsbo.utils.io.datasets.views.base import (
    ColumnStoreDatasetView,
    DatasetView,
    RowStoreDatasetView,
    _stack_statistics,
)


def _reader(fpath, file):
    # The storage facade imports these adapters; defer importing its extension
    # until construction to avoid a package-initialization cycle.
    from dsbo.utils.io.storage import _hdf5

    return file if file is not None else _hdf5.Reader(os.fsdecode(fpath))


class HDF5DatasetView(DatasetView):
    def __init__(self, fpath: str, file: _hdf5.Reader | None = None):
        self._file = _reader(fpath, file)
        try:
            self._keys = tuple(self._file.keys())
        except Exception:
            self._file.close()
            raise

    def shape(self, key: str) -> tuple[int, ...]:
        return tuple(self._file.shape(key))

    def read(self, key: str, index: int | slice):
        value = self._file.read(key, index)
        return value[()] if not isinstance(index, slice) and value.ndim == 0 else value

    def read_indices(self, keys: list[str], indices: list[int]) -> np.ndarray:
        return np.stack([self._file.read_indices(key, indices) for key in keys], axis=1)

    def partition_sizes(self, key: str) -> tuple[int, ...]:
        return tuple(self._file.partition_sizes(key))

    def statistics(self, keys: list[str]) -> tuple[int, np.ndarray, np.ndarray] | None:
        return _stack_statistics([self._file.statistics(key) for key in keys])

    def close(self) -> None:
        self._file.close()


class HDF5RowStoreDatasetView(RowStoreDatasetView):
    def __init__(self, fpath: str, file: _hdf5.Reader | None = None):
        from dsbo.utils.io.storage import _hdf5

        self._file = _reader(fpath, file)
        self._dataset_name = _hdf5.ROW_STORE_DATASET
        try:
            keys = tuple(self._file.row_keys())
            super().__init__(keys, self._file.shape(self._dataset_name)[0])
        except Exception:
            self._file.close()
            raise

    def read_many(self, keys: list[str], index: int | slice) -> np.ndarray:
        columns = self._columns(keys)
        return self._file.read(self._dataset_name, index)[..., columns]

    def read_indices(self, keys: list[str], indices: list[int]) -> np.ndarray:
        columns = self._columns(keys)
        return self._file.read_indices(self._dataset_name, indices)[..., columns]

    def partition_sizes(self, key: str) -> tuple[int, ...]:
        self._columns([key])
        return tuple(self._file.partition_sizes(self._dataset_name))

    def statistics(self, keys: list[str]) -> tuple[int, np.ndarray, np.ndarray] | None:
        statistics = self._file.statistics(self._dataset_name)
        if statistics is None:
            return None
        count, means, squared_deviations = statistics
        columns = self._columns(keys)
        return count, means[columns], squared_deviations[columns]

    def close(self) -> None:
        self._file.close()


class HDF5ColumnStoreDatasetView(ColumnStoreDatasetView):
    def __init__(self, fpath: str, group_name: str = "primal", file: _hdf5.Reader | None = None):
        self._file = _reader(fpath, file)
        try:
            columns = tuple(self._file.columns(group_name))
            if not columns:
                raise ValueError(f"HDF5 group '{group_name}' contains no columns.")
            first = f"{group_name}/{columns[0]}"
            shape = tuple(self._file.shape(first))
            super().__init__(
                group_name=group_name,
                sample_shape=shape[1:] if shape else (),
                partitions=tuple(self._file.partition_sizes(first)),
            )
            self.columns = columns
        except Exception:
            self._file.close()
            raise

    def _read_column(self, column: str, index: int | slice):
        return self._file.read(f"{self._group_name}/{column}", index)

    def read_indices(self, keys: list[str], indices: list[int]) -> np.ndarray:
        # Match DatasetView's placement of the key axis (axis 1), even when a
        # sample itself has additional dimensions.
        values = np.stack(
            [self._file.read_indices(f"{self._group_name}/{column}", indices) for column in self.columns],
            axis=-1,
        )
        return np.stack([values for _ in keys], axis=1)

    def close(self) -> None:
        self._file.close()
