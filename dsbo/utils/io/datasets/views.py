from __future__ import annotations

from typing import Callable
from abc import ABC, abstractmethod

import h5py
import torch


class RowSource:
    """Uniform chunkable row-block for per-feature statistics over the sample axis.

    Wraps any indexable row source (a ``DatasetView`` key, an in-memory tensor,
    or several scalar keys stacked as columns) behind a single
    ``(num_rows, __getitem__)`` interface so statistics can be computed in
    bounded chunks.
    """
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
    @abstractmethod
    def keys(self) -> tuple[str, ...]:
        pass

    @abstractmethod
    def shape(self, key: str) -> tuple[int, ...]:
        pass

    @abstractmethod
    def read(self, key: str, index: int | slice):
        pass

    @abstractmethod
    def partition_sizes(self, key: str) -> tuple[int, ...]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    def __enter__(self) -> "DatasetView":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class HDF5DatasetView(DatasetView):
    def __init__(self, fpath: str, prefix: str | None = None):
        self._file = h5py.File(fpath, "r")
        try:
            if prefix is not None and prefix not in self._file:
                self._datasets = {}
                self._keys = ()
                return
            root = self._file if prefix is None else self._file[prefix]
            if not isinstance(root, h5py.Group):
                raise ValueError(f"HDF5 dataset prefix '{prefix}' is not a group.")
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

    @property
    def keys(self) -> tuple[str, ...]:
        return self._keys

    def shape(self, key: str) -> tuple[int, ...]:
        return self._datasets[key].shape

    def read(self, key: str, index: int | slice):
        return self._datasets[key][index]

    def partition_sizes(self, key: str) -> tuple[int, ...]:
        dataset = self._datasets[key]
        if dataset.is_virtual:
            counts = []
            for mapping in dataset.virtual_sources():
                start, end = mapping.vspace.get_select_bounds()
                counts.append(end[0] - start[0] + 1)
            return tuple(counts)
        return (1 if dataset.shape == () else dataset.shape[0],)

    def close(self) -> None:
        self._file.close()


class InMemoryDatasetView(DatasetView):
    def __init__(self, datasets: dict[str, torch.Tensor], prefix: str | None = None):
        prefix = f"{prefix}/" if prefix else ""
        self._datasets = {
            key.removeprefix(prefix): value
            for key, value in datasets.items()
            if key.startswith(prefix)
        }
        self._keys = tuple(sorted(self._datasets))

    @property
    def keys(self) -> tuple[str, ...]:
        return self._keys

    def shape(self, key: str) -> tuple[int, ...]:
        return tuple(self._datasets[key].shape)

    def read(self, key: str, index: int | slice):
        return self._datasets[key][index]

    def partition_sizes(self, key: str) -> tuple[int, ...]:
        shape = self.shape(key)
        return (1 if shape == () else shape[0],)

    def close(self) -> None:
        pass
