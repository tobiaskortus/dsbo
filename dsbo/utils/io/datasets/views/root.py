from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from dsbo.utils.io.datasets.roottree import PartitionedROOTTTreeReader
from dsbo.utils.io.datasets.views.base import (
    ColumnStoreDatasetView,
    DatasetView,
    RowStoreDatasetView,
    _stack_statistics,
)


@dataclass(frozen=True)
class DatasetStatistics:
    count: int
    mean: np.ndarray
    squared_deviation: np.ndarray

    @classmethod
    def from_dict(cls, value: dict) -> "DatasetStatistics":
        return cls(
            count=value["count"],
            mean=np.asarray(value["mean"]),
            squared_deviation=np.asarray(value["squared_deviation"]),
        )


@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    shape: tuple[int, ...]
    dtype: str
    branches: tuple[str, ...]
    columns: tuple[str, ...] | None
    keys: tuple[str, ...] | None
    partitions: tuple[int, ...] | None
    statistics: DatasetStatistics | Mapping[str, DatasetStatistics] | None

    @classmethod
    def from_dict(cls, name: str, value: dict) -> "DatasetMetadata":
        statistics = value.get("statistics")
        if statistics is None:
            statistics = None
        elif isinstance(statistics, dict) and "count" in statistics:
            statistics = DatasetStatistics.from_dict(statistics)
        else:
            statistics = {
                key: DatasetStatistics.from_dict(item) for key, item in statistics.items()
            }
        return cls(
            name=name,
            shape=tuple(value["shape"]),
            dtype=value["dtype"],
            branches=tuple(value["branches"]),
            columns=None if value.get("columns") is None else tuple(value["columns"]),
            keys=None if value.get("keys") is None else tuple(value["keys"]),
            partitions=None if value.get("partitions") is None else tuple(value["partitions"]),
            statistics=statistics,
        )


class ROOTDatasetView(DatasetView):
    def __init__(self, fpath: str):
        self._reader = PartitionedROOTTTreeReader(fpath)
        self._datasets = {
            name: DatasetMetadata.from_dict(name, item)
            for name, item in self._reader.metadata["datasets"].items()
        }
        self._keys = tuple(sorted(self._datasets))
        self._partitions = {}
        for key, meta in self._datasets.items():
            row_axis = 0 if meta.columns is None else 1
            total_size = 1 if len(meta.shape) <= row_axis else meta.shape[row_axis]
            self._partitions[key] = self._reader.partition_sizes(
                key, meta.partitions, total_size, row_axis
            )

    def shape(self, key: str) -> tuple[int, ...]:
        return self._datasets[key].shape

    def read(self, key: str, index: int | slice):
        item = self._datasets[key]
        return self._reader.read_values(
            self._partitions[key], key, item.branches[0], index,
            item.shape[1:], np.dtype(item.dtype),
        )

    def read_indices(self, keys: list[str], indices: list[int]) -> np.ndarray:
        if not keys:
            return np.empty((len(indices), 0))
        values = [self.read(key, indices) for key in keys]
        return np.stack(values, axis=1)

    def partition_sizes(self, key: str) -> tuple[int, ...]:
        return self._partitions[key]

    def statistics(self, keys: list[str]) -> tuple[int, np.ndarray, np.ndarray] | None:
        return _stack_statistics(
            [
                None
                if (value := self._datasets[key].statistics) is None
                else (
                    value.count,
                    value.mean,
                    value.squared_deviation,
                )
                for key in keys
            ]
        )

    def close(self) -> None:
        self._reader.close()


class ROOTRowStoreDatasetView(RowStoreDatasetView):
    def __init__(self, fpath: str, dataset_name: str):
        from dsbo.utils.io.storage.root import ROOTFile

        self._reader = PartitionedROOTTTreeReader(fpath)
        self._dataset_name = dataset_name
        meta = DatasetMetadata.from_dict(
            dataset_name, self._reader.metadata["datasets"][dataset_name]
        )
        self._dtype = np.dtype(meta.dtype)
        self._statistics = meta.statistics
        self._value_branch = ROOTFile.VALUE_BRANCH

        self._single_branch = meta.keys is not None
        keys = meta.keys if self._single_branch else meta.columns
        self._branches = (
            {} if self._single_branch
            else dict(zip(keys, meta.branches, strict=True))
        )

        row_axis = 0 if self._single_branch else 1
        shape = meta.shape
        self._partitions = self._reader.partition_sizes(
            dataset_name, meta.partitions, shape[row_axis], row_axis
        )
        super().__init__(keys, sum(self._partitions))

    def read_many(self, keys: list[str], index: int | slice) -> np.ndarray:
        return self._read_rows(keys, index)

    def read_indices(self, keys: list[str], indices: list[int]) -> np.ndarray:
        return self._read_rows(keys, indices)

    def _read_rows(self, keys: list[str], index) -> np.ndarray:
        selected = tuple(keys)
        column_indices = self._columns(keys)
        branches = [] if self._single_branch else [self._branches[key] for key in selected]
        return self._reader.read_keyed(
            self._partitions, self._dataset_name, branches, self._value_branch,
            self._single_branch, column_indices, len(selected), index, self._dtype,
        )

    def partition_sizes(self, key: str) -> tuple[int, ...]:
        self._columns([key])
        return self._partitions

    def statistics(self, keys: list[str]) -> tuple[int, np.ndarray, np.ndarray] | None:
        if not isinstance(self._statistics, Mapping) or any(
            key not in self._statistics for key in keys
        ):
            return None
        return _stack_statistics(
            [
                (
                    self._statistics[key].count,
                    self._statistics[key].mean,
                    self._statistics[key].squared_deviation,
                )
                for key in keys
            ]
        )

    def close(self) -> None:
        self._reader.close()


class ROOTColumnStoreDatasetView(ColumnStoreDatasetView):
    def __init__(self, fpath: str, group_name: str = "primal"):
        self._reader = PartitionedROOTTTreeReader(fpath)
        dataset = self._reader.metadata["datasets"].get(group_name)
        if dataset is None or dataset["columns"] is None:
            raise ValueError(f"ROOT dataset '{group_name}' is not a group of named columns.")
        meta = DatasetMetadata.from_dict(group_name, dataset)
        columns = meta.columns
        if not columns:
            raise ValueError(f"ROOT dataset '{group_name}' contains no columns.")

        shape = meta.shape
        num_samples = 1 if len(shape) == 1 else shape[1]
        super().__init__(
            group_name=group_name,
            sample_shape=shape[2:],
            partitions=self._reader.partition_sizes(
                group_name, meta.partitions, num_samples, 1
            ),
        )
        self.columns = columns
        self._branches = dict(zip(columns, meta.branches, strict=True))
        self._dtype = np.dtype(meta.dtype)

    def read(self, key: str, index: int | slice):
        return self._read_columns(self.columns, index)

    def read_indices(self, keys: list[str], indices: list[int]) -> np.ndarray:
        if any(key != self._group_name for key in keys):
            missing = next(key for key in keys if key != self._group_name)
            raise KeyError(missing)
        result = self._read_columns(self.columns, indices)
        if not keys:
            return np.empty((result.shape[0], 0), dtype=self._dtype)
        return np.stack([result] * len(keys), axis=1)

    def _read_columns(self, columns: tuple[str, ...], index) -> np.ndarray:
        branches = [self._branches[column] for column in columns]
        return self._reader.read_columns(
            self._partitions, self._group_name, branches, index,
            (*self.sample_shape, len(columns)), self._dtype,
        )

    def _read_column(self, column: str, index: int | slice):
        return self._read_columns((column,), index)[..., 0]

    def close(self) -> None:
        self._reader.close()
