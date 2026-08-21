from __future__ import annotations

import h5py
import json
import numpy as np

from dsbo.utils.io.datasets.views.base import (
    ColumnStoreDatasetView,
    DatasetView,
    RowStoreDatasetView,
    _leading_count,
    _stack_statistics,
)


def _hdf5_partition_sizes(dataset: h5py.Dataset) -> tuple[int, ...]:
    from dsbo.utils.io.storage.hdf5 import HDF5File

    stored = dataset.attrs.get(HDF5File.PARTITION_SIZES_ATTRIBUTE)
    if stored is not None:
        return tuple(json.loads(stored))
    if dataset.is_virtual:
        return tuple(
            end[0] - start[0] + 1
            for mapping in dataset.virtual_sources()
            for start, end in [mapping.vspace.get_select_bounds()]
        )
    return (_leading_count(dataset.shape),)


class HDF5DatasetView(DatasetView):
    def __init__(self, fpath: str, file: h5py.File | None = None):
        self._file = file if file is not None else h5py.File(fpath, "r")
        try:
            from dsbo.utils.io.storage.hdf5 import HDF5File

            if file is None:
                HDF5File._validate_vds_sources(self._file)
            root = self._file
            if not isinstance(root, h5py.Group):
                raise ValueError("HDF5 dataset root is not a group.")
            datasets: dict[str, h5py.Dataset] = {}
            root.visititems(
                lambda name, value: datasets.update([(name, value)])
                if isinstance(value, h5py.Dataset) else None
            )
            self._datasets = datasets
            self._keys = tuple(sorted(datasets))
        except Exception:
            self._file.close()
            raise

    def shape(self, key: str) -> tuple[int, ...]:
        return self._datasets[key].shape

    def read(self, key: str, index: int | slice):
        return self._datasets[key][index]

    def partition_sizes(self, key: str) -> tuple[int, ...]:
        return _hdf5_partition_sizes(self._datasets[key])

    def statistics(self, keys: list[str]) -> tuple[int, np.ndarray, np.ndarray] | None:
        from dsbo.utils.io.storage.hdf5 import HDF5File

        return _stack_statistics(
            [HDF5File._read_statistics(self._datasets[key]) for key in keys]
        )

    def close(self) -> None:
        self._file.close()


class HDF5RowStoreDatasetView(RowStoreDatasetView):
    def __init__(self, file: h5py.File):
        from dsbo.utils.io.storage.hdf5 import HDF5File

        self._file = file
        try:
            self._dataset = file[HDF5File.ROW_STORE_DATASET]
            keys = tuple(json.loads(self._dataset.attrs[HDF5File.ROW_STORE_KEYS_ATTRIBUTE]))
        except Exception:
            file.close()
            raise
        super().__init__(keys, self._dataset.shape[0])

    def read_many(self, keys: list[str], index: int | slice) -> np.ndarray:
        columns = self._columns(keys)
        values = np.asarray(self._dataset[index, :])
        return values[..., columns]

    def read_indices(self, keys: list[str], indices: list[int]) -> np.ndarray:
        columns = self._columns(keys)
        requested = np.asarray(indices, dtype=np.int64)
        if requested.size == 0:
            return np.empty((0, len(columns)), dtype=self._dataset.dtype)
        requested = np.where(requested < 0, requested + self._dataset.shape[0], requested)
        if np.any((requested < 0) | (requested >= self._dataset.shape[0])):
            raise IndexError("HDF5 dataset index is out of range.")
        unique, inverse = np.unique(requested, return_inverse=True)
        values = np.asarray(self._dataset[unique, :])
        return values[inverse][..., columns]

    def partition_sizes(self, key: str) -> tuple[int, ...]:
        self._columns([key])
        return _hdf5_partition_sizes(self._dataset)

    def statistics(self, keys: list[str]) -> tuple[int, np.ndarray, np.ndarray] | None:
        from dsbo.utils.io.storage.hdf5 import HDF5File

        statistics = HDF5File._read_statistics(self._dataset)
        if statistics is None:
            return None
        count, means, squared_deviations = statistics
        columns = self._columns(keys)
        return count, np.asarray(means)[columns], np.asarray(squared_deviations)[columns]

    def close(self) -> None:
        self._file.close()


class HDF5ColumnStoreDatasetView(ColumnStoreDatasetView):
    def __init__(
        self,
        fpath: str,
        group_name: str = "primal",
        file: h5py.File | None = None,
    ):
        from dsbo.utils.io.storage.hdf5 import HDF5File
        self._file = file if file is not None else h5py.File(fpath, "r")
        try:
            group = self._file[group_name]
            if not isinstance(group, h5py.Group):
                raise ValueError(f"HDF5 dataset '{group_name}' is not a group of named columns.")
            stored = group.attrs.get(HDF5File.COLUMNS_ATTRIBUTE)
            columns = tuple(json.loads(stored)) if stored else tuple(group.keys())
            if not columns:
                raise ValueError(f"HDF5 group '{group_name}' contains no columns.")
            template = group[columns[0]]
            sample_shape = () if template.shape == () else template.shape[1:]
            partitions = _hdf5_partition_sizes(template)
        except Exception:
            self._file.close()
            raise

        super().__init__(
            group_name=group_name,
            sample_shape=sample_shape,
            partitions=partitions,
        )
        self.columns = columns
        self._group = group

    def _read_column(self, column: str, index: int | slice):
        dataset = self._group[column]
        if dataset.shape == ():
            return np.asarray(dataset[()]).reshape(1)[index]
        return np.asarray(dataset[index])

    def read_indices(self, keys: list[str], indices: list[int]) -> np.ndarray:
        requested = np.asarray(indices, dtype=np.int64)
        requested = np.where(requested < 0, requested + self.num_samples, requested)
        if np.any((requested < 0) | (requested >= self.num_samples)):
            raise IndexError("HDF5 dataset index is out of range.")
        unique, inverse = np.unique(requested, return_inverse=True)
        rows = self.read(self._group_name, unique)[inverse]
        return np.repeat(rows[:, np.newaxis], len(keys), axis=1)

    def close(self) -> None:
        self._file.close()
