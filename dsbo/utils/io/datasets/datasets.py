from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING
from torch.utils.data import Dataset

from dsbo.utils.io.datasets.views import RowSource

if TYPE_CHECKING:
    from dsbo.utils.io.storage.autograd import AutogradFileInterface
    from dsbo.utils.io.datasets.views import DatasetView


class RowSubset(RowSource):
    def __init__(self, view, keys, indices):
        self._view = view
        self._keys = keys
        self._indices = np.asarray(indices, dtype=np.int64)
        super().__init__(len(self._indices), self._read)

    def _read(self, index):
        sel = self._indices[index]
        if np.ndim(sel) == 0:
            return self._view.read_many(self._keys, int(sel))
        return self._view.read_indices(self._keys, sel.tolist())


class SurrogateDataset(Dataset):
    _CHUNK = 1024

    def __init__(self, backend: type[AutogradFileInterface]):
        self._backend = backend
        self._views: list[DatasetView] = []
        self._closed = False
        self._loaded = False
        self._mean: dict[str, torch.Tensor] = {}
        self._std: dict[str, torch.Tensor] = {}

    def _open_view(self, path: str) -> DatasetView:
        self._ensure_open()
        view = self._backend.open_dataset(path)
        self._views.append(view)
        return view

    def _release_views(self) -> None:
        for view in self._views:
            view.close()
        self._views = []

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} has been closed.")

    def _ensure_loaded(self) -> None:
        self._ensure_open()
        if not self._loaded:
            self._open()
            self._loaded = True

    def close(self) -> None:
        self._release_views()
        self._closed = True

    def __enter__(self) -> "SurrogateDataset":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_views"] = []
        state["_loaded"] = False
        for attribute in ("_data", "_cond", "_records"):
            if attribute in state:
                state[attribute] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__ = dict(state)
        self._views = []

    def __del__(self) -> None:
        self._release_views()

    def fit_normalization(self, sources: dict[str, RowSource]) -> None:
        for name, source in sources.items():
            self._mean[name], self._std[name] = self._fit_stats(source)

    def set_normalization_stats(
        self,
        name: str,
        count: int,
        mean: np.ndarray,
        squared_deviation: np.ndarray,
    ) -> None:
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32)
        self._mean[name] = mean_tensor
        if count < 2:
            self._std[name] = torch.ones_like(mean_tensor)
            return
        variance = torch.as_tensor(squared_deviation, dtype=torch.float64) / (count - 1)
        self._std[name] = torch.sqrt(variance).float() + 1e-8

    @classmethod
    def _fit_stats(cls, source: RowSource) -> tuple[torch.Tensor, torch.Tensor]:
        count = 0
        mean: torch.Tensor | None = None
        squared_deviation: torch.Tensor | None = None

        for start in range(0, source.num_rows, cls._CHUNK):
            values = torch.as_tensor(source[start:start + cls._CHUNK], dtype=torch.float64)
            batch_count = values.shape[0]
            batch_mean = values.mean(dim=0)
            batch_sq_dev = ((values - batch_mean) ** 2).sum(dim=0)

            if mean is None:
                mean, squared_deviation, count = batch_mean, batch_sq_dev, batch_count
                continue

            total = count + batch_count
            delta = batch_mean - mean
            mean = mean + delta * (batch_count / total)
            squared_deviation = (
                squared_deviation
                + batch_sq_dev
                + delta.square() * count * batch_count / total
            )
            count = total

        assert mean is not None and squared_deviation is not None
        mean = mean.float()
        if count < 2:
            return mean, torch.ones_like(mean)
        std = torch.sqrt(squared_deviation / (count - 1)).float() + 1e-8
        return mean, std

    def normalize(self, name: str, value: torch.Tensor) -> torch.Tensor:
        return (value - self._mean[name]) / self._std[name]

    def _set_normalization_from_view(self, name: str, view: "DatasetView", key: str) -> bool:
        stats = view.statistics([key])
        if stats is None:
            return False
        count, mean, squared_deviation = stats
        if mean.shape[0] == 1:
            mean = mean[0]
            squared_deviation = squared_deviation[0]
        self.set_normalization_stats(name, count, mean, squared_deviation)
        return True

    def _ensure_normalized(self) -> None:
        if not self._mean:
            self.compute_stats()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_loaded()
        return self.__getitems__([idx])[0]

    @property
    def mean_x(self) -> torch.Tensor | None:
        return self._mean.get("x")

    @property
    def std_x(self) -> torch.Tensor | None:
        return self._std.get("x")

    @staticmethod
    def _is_contiguous(indices: list[int]) -> bool:
        return all(right == left + 1 for left, right in zip(indices, indices[1:]))

class ModuleSurrogateDataset(SurrogateDataset):
    def __init__(
        self,
        data_path: str,
        cond_path: str,
        backend: type[AutogradFileInterface],
    ):
        super().__init__(backend)
        self._data_path = data_path
        self._cond_path = cond_path
        self._data: DatasetView | None = None
        self._cond: DatasetView | None = None
        self._open()
        self._loaded = True

    def _open(self) -> None:
        data = self._open_view(self._data_path)
        try:
            cond = self._open_view(self._cond_path)
        except Exception:
            self._release_views()
            raise

        try:
            if "primal" not in data.keys or "primal" not in cond.keys:
                raise ValueError("Module surrogate data must contain a 'primal' dataset.")
            data_shape = data.shape("primal")
            cond_shape = cond.shape("primal")
            if not data_shape or not cond_shape:
                raise ValueError("Module surrogate data must have a sample dimension.")
            if data_shape[0] != cond_shape[0]:
                raise ValueError(
                    "Module surrogate data and conditioning must contain the same "
                    "number of samples."
                )
            if data_shape[0] == 0:
                raise ValueError("Module surrogate data must contain at least one sample.")
        except Exception:
            self._release_views()
            raise

        self._data = data
        self._cond = cond

    @property
    def mean_c(self) -> torch.Tensor | None:
        return self._mean.get("c")

    @property
    def std_c(self) -> torch.Tensor | None:
        return self._std.get("c")

    def compute_stats(self) -> None:
        self._ensure_loaded()
        self.fit_normalization(
            {
                "x": RowSource(self._data.shape("primal")[0], lambda i: self._data.read("primal", i)),
                "c": RowSource(self._cond.shape("primal")[0], lambda i: self._cond.read("primal", i)),
            }
        )

    def __getitems__(self, indices: list[int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        self._ensure_loaded()
        if not indices:
            return []
        self._ensure_normalized()
        if self._is_contiguous(indices):
            selection = slice(indices[0], indices[-1] + 1)
            x_values = self._data.read("primal", selection)
            c_values = self._cond.read("primal", selection)
        else:
            x_values = self._data.read_indices(["primal"], indices)[:, 0]
            c_values = self._cond.read_indices(["primal"], indices)[:, 0]
        x = torch.as_tensor(x_values, dtype=torch.float32)
        c = torch.as_tensor(c_values, dtype=torch.float32)
        return list(zip(self.normalize("x", x), self.normalize("c", c), strict=True))

    def __len__(self) -> int:
        self._ensure_loaded()
        return self._data.shape("primal")[0]


class WorkflowSurrogateDataset(SurrogateDataset):
    def __init__(
        self,
        records_path: str,
        design: torch.Tensor,
        backend: type[AutogradFileInterface],
    ):
        super().__init__(backend)
        self._records_path = records_path
        self._design = torch.as_tensor(design).detach().cpu().float()
        self._records: DatasetView | None = None
        self._keys: list[str] = []
        self._n = 0
        self._length = 0
        try:
            self._open()
            self._loaded = True
        except Exception:
            self.close()
            raise

    def _open(self) -> None:
        records = self._open_view(self._records_path)

        try:
            keys = list(records.keys)
            if not keys:
                raise ValueError("No results recorded for workflow surrogate training.")
            if any(len(records.shape(key)) != 1 for key in keys):
                raise ValueError("Workflow surrogate results must contain one scalar per sample.")
            if self._design.ndim == 0 or self._design.shape[0] == 0:
                raise ValueError("Workflow surrogate training requires at least one candidate.")

            partition_sizes = [records.partition_sizes(key) for key in keys]
            expected = partition_sizes[0]
            if not expected or any(sizes != expected for sizes in partition_sizes[1:]):
                raise ValueError("Workflow surrogate results must all have the same row count.")
            if len(expected) != self._design.shape[0]:
                raise ValueError(
                    "Workflow surrogate results and design must contain the same number of candidates."
                )
            if len(set(expected)) != 1:
                raise ValueError("Every workflow surrogate candidate must have the same row count.")
        except Exception:
            self._release_views()
            raise

        self._records = records
        self._keys = keys
        self._n = expected[0]
        self._length = self._n * len(expected)

    @property
    def mean_y(self) -> torch.Tensor | None:
        return self._mean.get("y")

    @property
    def std_y(self) -> torch.Tensor | None:
        return self._std.get("y")

    def compute_stats(self, row_indices: list[int] | None = None) -> None:
        self._ensure_loaded()
        if row_indices is None:
            candidates = list(range(self._design.shape[0]))
        else:
            candidates = sorted({int(r) // self._n for r in row_indices})
        self.fit_normalization(
            {"x": RowSource(len(candidates), lambda i: self._design[candidates[i]].numpy())}
        )
        if row_indices is not None:
            # Stream only the selected training rows: the persisted whole-file
            # statistics span the full dataset (including validation).
            self.fit_normalization({"y": RowSubset(self._records, self._keys, row_indices)})
            return
        stored = self._records.statistics(self._keys)
        if stored is not None:
            self.set_normalization_stats("y", *stored)
            return
        raise RuntimeError("Workflow surrogate records do not contain persisted statistics.")

    @property
    def keys(self) -> list[str]:
        self._ensure_loaded()
        return self._keys

    @property
    def n(self) -> int:
        self._ensure_loaded()
        return self._n

    def __len__(self) -> int:
        self._ensure_loaded()
        return self._length

    def __getitems__(self, indices: list[int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        self._ensure_loaded()
        if not indices:
            return []
        self._ensure_normalized()
        values = (
            self._records.read_many(self._keys, slice(indices[0], indices[-1] + 1))
            if self._is_contiguous(indices)
            else self._records.read_indices(self._keys, indices)
        )
        y = torch.as_tensor(
            values,
            dtype=torch.float32,
        )
        design_indices = torch.as_tensor(indices) // self._n
        x = self._design[design_indices]
        normalized_x = self.normalize("x", x)
        normalized_y = self.normalize("y", y)
        return list(zip(normalized_x, normalized_y, strict=True))


def design_split(n_candidates: int, rows_per_design: int, val_split: float, seed: int) -> tuple[list[int], list[int]]:
    """Split candidates by *design* into (train, val) row indices."""
    k = rows_per_design

    if val_split <= 0.0 or n_candidates < 2:
        return list(range(n_candidates * k)), []

    def rows_of(designs: list[int]) -> list[int]:
        return [r for c in designs for r in range(c * k, (c + 1) * k)]

    perm = torch.randperm(n_candidates, generator=torch.Generator().manual_seed(seed)).tolist()
    n_val = min(max(1, int(n_candidates * val_split)), n_candidates - 1)
    val_designs, train_designs = perm[:n_val], perm[n_val:]
    return rows_of(train_designs), rows_of(val_designs)
