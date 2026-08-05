from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from dsbo.utils.io.autograd import AutogradFileInterface
    from dsbo.utils.io.datasets.views import DatasetView, RowSource


class SurrogateDataset(Dataset):
    _CHUNK = 1024

    def __init__(self, backend: type[AutogradFileInterface]):
        self._backend = backend
        self._views: list[DatasetView] = []
        self._closed = False
        self._mean: dict[str, torch.Tensor] = {}
        self._std: dict[str, torch.Tensor] = {}

    def _open_view(self, path: str, prefix: str | None = None) -> DatasetView:
        self._ensure_open()
        view = self._backend.open_dataset(path, prefix=prefix)
        self._views.append(view)
        return view

    def _release_views(self) -> None:
        for view in self._views:
            view.close()
        self._views = []

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} has been closed.")

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
        state["_closed"] = False
        return state

    def __del__(self) -> None:
        self._release_views()

    def fit_normalization(self, sources: dict[str, RowSource]) -> None:
        for name, source in sources.items():
            self._mean[name], self._std[name] = self._fit_stats(source)

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
    def mean_x(self) -> torch.Tensor | None:
        return self._mean.get("x")

    @property
    def std_x(self) -> torch.Tensor | None:
        return self._std.get("x")

    @property
    def mean_c(self) -> torch.Tensor | None:
        return self._mean.get("c")

    @property
    def std_c(self) -> torch.Tensor | None:
        return self._std.get("c")

    def compute_stats(self) -> None:
        self._ensure_open()
        self.fit_normalization(
            {
                #Read from default Storage.write() generated files
                "x": RowSource(self._data.shape("primal")[0], lambda i: self._data.read("primal", i)),
                "c": RowSource(self._cond.shape("primal")[0], lambda i: self._cond.read("primal", i)),
            }
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_open()
        if len(self._mean) == 0:
            self.compute_stats()

        #Read from default Storage.write() generated files
        x = torch.as_tensor(self._data.read("primal", idx), dtype=torch.float32)
        c = torch.as_tensor(self._cond.read("primal", idx), dtype=torch.float32)
        return self.normalize("x", x), self.normalize("c", c)

    def __len__(self) -> int:
        self._ensure_open()
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
        self._design = torch.as_tensor(design).float()
        self._records: DatasetView | None = None
        self._keys: list[str] = []
        self._n = 0
        self._length = 0
        try:
            self._open()
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
    def mean_x(self) -> torch.Tensor | None:
        return self._mean.get("x")

    @property
    def std_x(self) -> torch.Tensor | None:
        return self._std.get("x")

    @property
    def mean_y(self) -> torch.Tensor | None:
        return self._mean.get("y")

    @property
    def std_y(self) -> torch.Tensor | None:
        return self._std.get("y")

    def compute_stats(self) -> None:
        self._ensure_open()
        self.fit_normalization(
            {
                #Use bound design and record files
                "x": RowSource(self._design.shape[0], lambda i: self._design[i].numpy()),
                "y": RowSource(
                    self._records.shape(self._keys[0])[0],
                    lambda i: np.stack(
                        [self._records.read(k, i) for k in self._keys], axis=-1
                    ),
                ),
            }
        )

    @property
    def keys(self) -> list[str]:
        self._ensure_open()
        return self._keys

    @property
    def n(self) -> int:
        self._ensure_open()
        return self._n

    def __len__(self) -> int:
        self._ensure_open()
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_open()
        if "y" not in self._mean :
            self.compute_stats()

        #Use bound design and record files
        x = self._design[idx // self._n]
        y = torch.tensor(
            [self._records.read(key, idx) for key in self._keys],
            dtype=torch.float32,
        )
        return self.normalize("x", x), self.normalize("y", y)
