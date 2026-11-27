from __future__ import annotations

import os
import numpy as np
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

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
        if not isinstance(index, slice) and value.ndim == 0:
            return torch.as_tensor(value[()])
        return torch.from_numpy(value)

    def read_many(self, keys: list[str], index: int | slice) -> TensorDict:
        from dsbo.utils.io.storage.utils import to_tensordict
        return to_tensordict({key: self.read(key, index) for key in keys})

    def read_indices(self, keys: list[str], indices: list[int]) -> TensorDict:
        from dsbo.utils.io.storage.utils import to_tensordict
        if len(keys) == 1:
            arrays = {keys[0]: self._file.read_indices(keys[0], indices)}
        else:
            arrays = self._file.read_indices_many(keys, indices)
        return to_tensordict(arrays)

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

    def read_many(self, keys: list[str], index: int | slice) -> torch.Tensor:
        columns = self._columns(keys)
        return torch.from_numpy(self._file.read(self._dataset_name, index)[..., columns])

    def read_indices(self, keys: list[str], indices: list[int]) -> torch.Tensor:
        columns = self._columns(keys)
        return torch.from_numpy(self._file.read_indices(self._dataset_name, indices)[..., columns])

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
            keys = tuple(self._file.columns(group_name))
            if not keys:
                raise ValueError(f"HDF5 group '{group_name}' contains no keys.")
            super().__init__(group_name=group_name, keys=keys)
        except Exception:
            self._file.close()
            raise

    def shape(self, key: str) -> tuple[int, ...]:
        return tuple(self._file.shape(f"{self._group_name}/{key}"))

    def read(self, key: str, index: int | slice):
        value = self._file.read(f"{self._group_name}/{key}", index)
        if not isinstance(index, slice) and value.ndim == 0:
            return torch.as_tensor(value[()])
        return torch.from_numpy(value)

    def read_many(self, keys: list[str], index: int | slice) -> TensorDict:
        from dsbo.utils.io.storage.utils import to_tensordict
        return to_tensordict({key: self.read(key, index) for key in keys})

    def read_indices(self, keys: list[str], indices: list[int]) -> TensorDict:
        from dsbo.utils.io.storage.utils import to_tensordict
        names = [f"{self._group_name}/{key}" for key in keys]
        if len(keys) == 1:
            arrays = {keys[0]: self._file.read_indices(names[0], indices)}
        else:
            raw = self._file.read_indices_many(names, indices)
            arrays = {key: raw[name] for key, name in zip(keys, names)}
        return to_tensordict(arrays)

    def _read_indices(self, key: str, indices: list[int]) -> np.ndarray:
        return self._file.read_indices(f"{self._group_name}/{key}", indices)

    def _statistics(self, key: str) -> tuple[int, np.ndarray, np.ndarray] | None:
        return self._file.statistics(f"{self._group_name}/{key}")

    def partition_sizes(self, key: str) -> tuple[int, ...]:
        return tuple(self._file.partition_sizes(f"{self._group_name}/{key}"))

    def close(self) -> None:
        self._file.close()
