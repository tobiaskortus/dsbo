from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import glob
import os
import numpy as np
from pathlib import Path
from typing import Optional

import h5py
import torch
from torch.utils.data import Dataset, DataLoader

from dsbo.utils.io.autograd import AutogradFileStorage
from dsbo.utils.io.workdir import AbstractFilePath, Workdir


@dataclass
class WelfordAggregate:
    mean_x: Optional[torch.Tensor] = None
    M2_x: Optional[torch.Tensor] = None
    mean_c: Optional[torch.Tensor] = None
    M2_c: Optional[torch.Tensor] = None
    n: int = 0


class ShardedDataset(Dataset):
    def __init__(
        self,
        data_paths: list[str],
        cond_paths: list[str],
    ):
        self._data_paths = sorted(data_paths)
        self._cond_paths = sorted(cond_paths)
        assert len(self._data_paths) == len(self._cond_paths), (
            f"Mismatched shard count: {len(self._data_paths)} data vs "
            f"{len(self._cond_paths)} cond"
        )

        self._welford_agg = WelfordAggregate()

        self._mean_x: Optional[torch.Tensor] = None
        self._std_x: Optional[torch.Tensor] = None
        self._mean_c: Optional[torch.Tensor] = None
        self._std_c: Optional[torch.Tensor] = None

        self._len: Optional[int] = None
        self._shard_lengths: list[int] = []

    def _welford_combine(self, mean_a, M2_a, n_a, mean_b, M2_b, n_b):
        delta = mean_b - mean_a
        n = n_a + n_b
        mean = mean_a + delta * n_b / n
        M2 = M2_a + M2_b + delta ** 2 * n_a * n_b / n
        return mean, M2, n

    def _welford_update(self, x: torch.Tensor, c: torch.Tensor):
        n_s = x.shape[0]
        mean_x_s = x.mean(dim=0)
        mean_c_s = c.mean(dim=0)
        M2_x_s = ((x - mean_x_s) ** 2).sum(dim=0)
        M2_c_s = ((c - mean_c_s) ** 2).sum(dim=0)

        if self._welford_agg.n == 0:
            self._welford_agg.mean_x, self._welford_agg.M2_x, self._welford_agg.n = mean_x_s, M2_x_s, n_s
            self._welford_agg.mean_c, self._welford_agg.M2_c = mean_c_s, M2_c_s
        else:
            self._welford_agg.mean_x, self._welford_agg.M2_x, self._welford_agg.n = self._welford_combine(
                self._welford_agg.mean_x, self._welford_agg.M2_x, self._welford_agg.n, mean_x_s, M2_x_s, n_s
            )
            self._welford_agg.mean_c, self._welford_agg.M2_c, _ = self._welford_combine(
                self._welford_agg.mean_c, self._welford_agg.M2_c, self._welford_agg.n - n_s, mean_c_s, M2_c_s, n_s
            )

    def _finalize(self):
        self._mean_x = self._welford_agg.mean_x
        self._mean_c = self._welford_agg.mean_c

        if self._welford_agg.n < 2:
            self._std_x = torch.ones_like(self._welford_agg.mean_x)
            self._std_c = torch.ones_like(self._welford_agg.mean_c)
        else:
            self._std_x = torch.sqrt(self._welford_agg.M2_x / (self._welford_agg.n - 1)) + 1e-8
            self._std_c = torch.sqrt(self._welford_agg.M2_c / (self._welford_agg.n - 1)) + 1e-8


    def _ensure_loaded(self):
        if hasattr(self, "_data_arrays"):
            return
        self._data_arrays = [_load_single_h5(p) for p in self._data_paths]
        self._cond_arrays = [_load_single_h5(p) for p in self._cond_paths]
        self._shard_lengths = [a.shape[0] for a in self._data_arrays]
        self._len = sum(self._shard_lengths)


    def _map_index(self, flat_idx: int) -> tuple[int, int]:
        self._ensure_loaded()
        cum = 0
        for sidx, arr_len in enumerate(self._shard_lengths):
            if flat_idx < cum + arr_len:
                return sidx, flat_idx - cum
            cum += arr_len
        raise IndexError(f"Index {flat_idx} out of range [0, {self._len})")


    @property
    def output_dim(self) -> int:
        if self._data_paths:
            arr = _load_single_h5(self._data_paths[0])
            return arr.shape[1] if arr.ndim > 1 else 1
        raise RuntimeError("No data shards available")

    @property
    def cond_dim(self) -> int:
        if self._cond_paths:
            arr = _load_single_h5(self._cond_paths[0])
            return arr.shape[1] if arr.ndim > 1 else 1
        raise RuntimeError("No cond shards available")

    @property
    def mean_x(self) -> Optional[torch.Tensor]:
        return self._mean_x

    @property
    def std_x(self) -> Optional[torch.Tensor]:
        return self._std_x

    @property
    def mean_c(self) -> Optional[torch.Tensor]:
        return self._mean_c

    @property
    def std_c(self) -> Optional[torch.Tensor]:
        return self._std_c

    def compute_stats(self):
        self._ensure_loaded()
        if len(self) == 0:
            return

        for sidx in range(len(self._data_arrays)):
            x = torch.from_numpy(self._data_arrays[sidx].copy()).float()
            c = torch.from_numpy(self._cond_arrays[sidx].copy()).float()
            self._welford_update(x, c)

        self._finalize()


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_loaded()
        sidx, local = self._map_index(idx)
        x = torch.from_numpy(self._data_arrays[sidx][local].copy()).float()
        c = torch.from_numpy(self._cond_arrays[sidx][local].copy()).float()

        if self._mean_x is None:
            self.compute_stats()

        x = (x - self._mean_x) / self._std_x
        c = (c - self._mean_c) / self._std_c
        return x, c

    def __len__(self) -> int:
        self._ensure_loaded()
        return self._len


class ShardedStorageBase(AutogradFileStorage, ABC):
    max_rows_per_shard: int = 10_000

    @classmethod
    @abstractmethod
    def _storage_ext(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def _load_array(cls, path: str) -> np.ndarray:
        pass

    @classmethod
    @abstractmethod
    def _save_array(cls, path: str, arr: np.ndarray):
        pass

    @classmethod
    def _storage_path(cls, fpath: str) -> str:
        base, _ = os.path.splitext(fpath)
        return base + cls._storage_ext()

    @classmethod
    def _discover_shards(cls, stem: str) -> list[str]:
        pattern = f"{stem}.*{cls._storage_ext()}"
        return sorted(glob.glob(pattern))

    @classmethod
    def _discover_shards_from_abstract(cls, file_path: AbstractFilePath) -> list[str]:
        from dsbo.utils.io.workdir import PathAnnotation
        if isinstance(file_path, PathAnnotation):
            file_path = file_path.path

        wd = Workdir._current_workdir.get()
        if wd is None:
            return []

        search_root = Path(wd.root)
        name = file_path.filename
        stem, _ = os.path.splitext(name)
        rel = file_path.pathname
        ext = cls._storage_ext()

        #TODO: this is a bit hacky as it includes additional 
        # files in the workdir that matches the pattern. 
        patterns = [
            f"**/{rel}/{stem}.*{ext}",
            f"**/{rel}/{stem}{ext}",
            f"**/batch_*/{rel}/{stem}.*{ext}",
            f"**/batch_*/{rel}/{stem}{ext}",
        ]

        matches: set[str] = set()
        for pat in patterns:
            for p in search_root.glob(pat):
                if p.is_file() and "_gradient" not in str(p):
                    matches.add(str(p))

        return sorted(matches)



    @classmethod
    def open_primal(cls, fpath: str, columns: list[str] | None = None) -> np.ndarray:
        return cls._load_array(cls._storage_path(fpath))

    @classmethod
    def open_primal_tangent(cls, fpath: str, columns: list[str] | None = None) -> tuple[np.ndarray, Optional[np.ndarray]]:
        storage_path = cls._storage_path(fpath)
        tangent_path = cls._tangent_path(storage_path)
        primal = cls._load_array(storage_path)
        try:
            tangent = cls._load_array(tangent_path)
        except FileNotFoundError:
            tangent = None
        return primal, tangent

    @classmethod
    def write_primal(cls, fpath: str, primal: np.ndarray, columns: list[str] | None = None):
        cls._save_array(cls._storage_path(fpath), primal)

    @classmethod
    def write_primal_tangent(cls, fpath: str, primal: np.ndarray, tangent: np.ndarray, columns: list[str] | None = None):
        storage_path = cls._storage_path(fpath)
        cls._save_array(storage_path, primal)
        cls._save_array(cls._tangent_path(storage_path), tangent)

    @classmethod
    def _save_gradient(cls, grad_path: str, gradient: np.ndarray, columns: list[str] | None = None):
        cls._save_array(cls._storage_path(grad_path), gradient)

    @classmethod
    def _load_gradient(cls, grad_path: str, columns: list[str] | None = None) -> np.ndarray:
        return cls._load_array(cls._storage_path(grad_path))

    @classmethod
    def read_backward_gradient(cls, fpath: str) -> torch.Tensor:
        return torch.from_numpy(cls._load_array(cls._storage_path(cls._gradient_path(fpath)))).float()

    @classmethod
    def reset(cls):
        cls.reset_gradient_state()

    @classmethod
    def _tangent_path(cls, storage_path: str) -> str:
        ext = cls._storage_ext()
        return storage_path.replace(ext, f"_tangent{ext}")

    @classmethod
    def shape_from_path(cls, path: str) -> tuple[int, ...]:
        return cls._load_array(path).shape

    @classmethod
    def to_data_loader(
        cls,
        data_fpath_abstract: AbstractFilePath,
        cond_fpath_abstract: AbstractFilePath,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        data_shards = cls._discover_shards_from_abstract(data_fpath_abstract)
        cond_shards = cls._discover_shards_from_abstract(cond_fpath_abstract)

        if not data_shards or not cond_shards:
            raise FileNotFoundError(
                f"Cannot create DataLoader: shards missing. "
                f"data_shards={len(data_shards)}, cond_shards={len(cond_shards)}"
            )

        dataset = ShardedDataset(data_shards, cond_shards)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def _load_single_h5(path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f["primal"][:]


class ShardedHDF5Storage(ShardedStorageBase):
    @classmethod
    def _storage_ext(cls) -> str:
        return ".h5"

    @classmethod
    def _load_array(cls, path: str) -> np.ndarray:
        if os.path.exists(path):
            with h5py.File(path, "r") as f:
                return f["primal"][:]
        stem, _ = os.path.splitext(path)
        shards = cls._discover_shards(stem)
        if not shards:
            raise FileNotFoundError(f"No shard files found for {path}")
        return np.concatenate(
            [_load_single_h5(s) for s in shards], axis=0
        )

    @classmethod
    def _save_array(cls, path: str, arr: np.ndarray):
        stem, _ = os.path.splitext(path)
        n = arr.shape[0]
        if n <= cls.max_rows_per_shard:
            with h5py.File(path, "w") as f:
                f.create_dataset("primal", data=arr, compression="gzip")
        else:
            n_shards = (n + cls.max_rows_per_shard - 1) // cls.max_rows_per_shard
            for i in range(n_shards):
                start = i * cls.max_rows_per_shard
                end = min(start + cls.max_rows_per_shard, n)
                shard_path = f"{stem}.{i}.h5"
                with h5py.File(shard_path, "w") as f:
                    f.create_dataset("primal", data=arr[start:end], compression="gzip")

    @classmethod
    def reduce(cls, sources: list[str], destination: str) -> None:
        from dsbo.utils.io.hdf5 import HDF5File
        HDF5File.reduce(sources, destination)
