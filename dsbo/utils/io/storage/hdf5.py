import os
from collections.abc import Iterator
from typing import TypeAlias

import numpy as np
from tensordict import TensorDict

from dsbo.utils.io.datasets.views import (
    STORE_COLUMN,
    STORE_FLAT,
    STORE_ROW,
    DatasetView,
    HDF5ColumnStoreDatasetView,
    HDF5DatasetView,
    HDF5RowStoreDatasetView,
)
from dsbo.utils.io.storage import _hdf5
from dsbo.utils.io.storage.autograd import AutogradFileInterface
from dsbo.utils.io.storage.utils import to_dict, to_tensordict

ArrayValues: TypeAlias = np.ndarray | TensorDict
GradientSource: TypeAlias = str | np.ndarray | TensorDict


class HDF5Statistics:
    STATS_COUNT_ATTRIBUTE = _hdf5.STATS_COUNT_ATTRIBUTE
    STATS_MEAN_ATTRIBUTE = _hdf5.STATS_MEAN_ATTRIBUTE
    STATS_SQUARED_DEVIATION_ATTRIBUTE = _hdf5.STATS_SQUARED_DEVIATION_ATTRIBUTE

    @classmethod
    def _statistics(cls, value: np.ndarray) -> tuple[int, np.ndarray, np.ndarray] | None:
        return _hdf5.statistics(value)

    @staticmethod
    def _combine_statistics(values: list[tuple[int, np.ndarray, np.ndarray] | None]) -> tuple[int, np.ndarray, np.ndarray] | None:
        return _hdf5.combine_statistics(values)


class HDF5Writer(HDF5Statistics):
    VDS_SOURCES_ATTRIBUTE = _hdf5.VDS_SOURCES_ATTRIBUTE
    PARTITION_SIZES_ATTRIBUTE = _hdf5.PARTITION_SIZES_ATTRIBUTE
    ROW_STORE_DATASET = _hdf5.ROW_STORE_DATASET
    ROW_STORE_KEYS_ATTRIBUTE = _hdf5.ROW_STORE_KEYS_ATTRIBUTE
    STORE_ATTRIBUTE = _hdf5.STORE_ATTRIBUTE
    COLUMNS_ATTRIBUTE = _hdf5.COLUMNS_ATTRIBUTE

    @classmethod
    def reducible_extensions(cls) -> frozenset[str]:
        return frozenset({".h5", ".hdf5"})

    @classmethod
    def open_primal(cls, fpath: str, columns: list[str] | None = None) -> ArrayValues:
        value = _hdf5.open_primal(os.fsdecode(fpath), columns)
        return to_tensordict(value) if isinstance(value, dict) else value

    @classmethod
    def open_primal_tangent(cls, fpath: str, columns: list[str] | None = None) -> tuple[ArrayValues, ArrayValues | None]:
        primal, tangent = _hdf5.open_primal_tangent(os.fsdecode(fpath), columns)
        if isinstance(primal, dict):
            primal = to_tensordict(primal)
        if isinstance(tangent, dict):
            tangent = to_tensordict(tangent)
        return primal, tangent

    @classmethod
    def write_primal(cls, fpath: str, primal: ArrayValues):
        if isinstance(primal, TensorDict):
            primal = to_dict(primal, numpy=True)
        _hdf5.write_primal(os.fsdecode(fpath), primal)

    @classmethod
    def write_primal_tangent(cls, fpath: str, primal: ArrayValues, tangent: ArrayValues) -> None:
        if isinstance(primal, TensorDict):
            primal = to_dict(primal, numpy=True)
        if isinstance(tangent, TensorDict):
            tangent = to_dict(tangent, numpy=True)
        _hdf5.write_primal_tangent(os.fsdecode(fpath), primal, tangent)

    @classmethod
    def write_named_datasets(cls, fpath: str, datasets: dict[str, np.ndarray]) -> None:
        _hdf5.write_named_datasets(os.fsdecode(fpath), datasets)

    @classmethod
    def write_row_store(cls, fpath: str, datasets: dict[str, np.ndarray]) -> None:
        _hdf5.write_row_store(os.fsdecode(fpath), datasets)

    @classmethod
    def open_dataset(cls, fpath: str) -> DatasetView:
        path = os.fsdecode(fpath)
        reader = _hdf5.Reader(path)
        try:
            views = {
                STORE_ROW: HDF5RowStoreDatasetView,
                STORE_COLUMN: HDF5ColumnStoreDatasetView,
                STORE_FLAT: HDF5DatasetView,
            }
            view = views.get(reader.store)
            if view is None:
                raise ValueError(f"HDF5 file '{path}' has no '{cls.STORE_ATTRIBUTE}' store-type tag.")
            return view(path, file=reader)
        except Exception:
            reader.close()
            raise

    @classmethod
    def columns(cls, fpath: str) -> list[str]:
        return _hdf5.columns(os.fsdecode(fpath))

    @classmethod
    def shape_from_path(cls, path: str) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
        value = _hdf5.shape_from_path(os.fsdecode(path))
        if isinstance(value, dict):
            return {key: tuple(shape) for key, shape in value.items()}
        return tuple(value)

    @classmethod
    def load_gradient(cls, grad_path: str, columns: list[str] | None = None) -> ArrayValues:
        value = _hdf5.load_gradient(os.fsdecode(grad_path), columns)
        return to_tensordict(value) if isinstance(value, dict) else value

    @classmethod
    def load_backward_gradient(cls, fpath: str, grad_path: str) -> ArrayValues:
        value = _hdf5.load_backward_gradient(os.fsdecode(fpath), os.fsdecode(grad_path))
        return to_tensordict(value) if isinstance(value, dict) else value

    @classmethod
    def write_gradient(cls, fpath: str, gradient: ArrayValues, *, merge: bool = False) -> None:
        if isinstance(gradient, TensorDict):
            gradient = to_dict(gradient, numpy=True)
        _hdf5.write_gradient(os.fsdecode(fpath), gradient, merge=merge)

    @classmethod
    def aggregate_gradients(cls, sources: list[str], destination: str) -> None:
        _hdf5.aggregate_gradients([os.fsdecode(source) for source in sources], os.fsdecode(destination))

    @classmethod
    def gradient_file_contributions(cls, fpath: str, gradient_path: str) -> Iterator[tuple[str, GradientSource, list[str] | None]]:
        for target, source, keys in _hdf5.gradient_file_contributions(os.fsdecode(fpath), os.fsdecode(gradient_path)):
            if isinstance(source, dict):
                source = to_tensordict(source)
            yield target, source, keys or None

    @classmethod
    def reduce(cls, sources: list[str], destination: str) -> None:
        _hdf5.reduce([os.fsdecode(source) for source in sources], os.fsdecode(destination))


class HDF5File(AutogradFileInterface, HDF5Writer):
    """DSBO autograd file channel backed by the native HDF5 writer."""

    writer = HDF5Writer
