import os
import re
import glob
import torch
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable
import torch.autograd.forward_ad as fw
from torch.utils.hooks import RemovableHandle

from dsbo.utils.io.datasets.views import DatasetView, InMemoryDatasetView


@dataclass
class GradientBlock:
    values: np.ndarray
    columns: list[str] | None = None

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values)


@dataclass
class GradientChild:
    path: str
    select: Callable[[GradientBlock], GradientBlock]


def merge_gradients(existing: GradientBlock | None, contribution: GradientBlock) -> GradientBlock:
    if existing is None:
        return contribution

    if existing.columns is None and contribution.columns is None:
        return GradientBlock(existing.values + contribution.values, None)
    if existing.columns is None or contribution.columns is None:
        raise ValueError("Cannot combine named and unnamed gradient contributions.")

    columns = list(existing.columns)
    for column in contribution.columns:
        if column not in columns:
            columns.append(column)

    existing_by_column = {column: existing.values[i] for i, column in enumerate(existing.columns)}
    contribution_by_column = {column: contribution.values[i] for i, column in enumerate(contribution.columns)}

    values = []
    for column in columns:
        left = existing_by_column.get(column)
        right = contribution_by_column.get(column)
        if left is None:
            left = np.zeros_like(right)
        if right is None:
            right = np.zeros_like(left)
        values.append(left + right)
    return GradientBlock(np.stack(values, axis=0), columns)


class AutogradFileInterface(ABC):
    handles: dict[str, RemovableHandle] = {}
    _registered_extensions: set[str] = set()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        AutogradFileInterface._registered_extensions.update(cls.reducible_extensions())

    @classmethod
    def reducible_extensions(cls) -> frozenset[str]:
        return frozenset()

    @classmethod
    def default_extension(cls) -> str:
        extensions = cls.reducible_extensions()
        return min(extensions, key=lambda extension: (len(extension), extension)) if extensions else ""

    @classmethod
    def registered_extensions(cls) -> frozenset[str]:
        return frozenset(AutogradFileInterface._registered_extensions)

    @classmethod
    def reset_gradient_state(cls):
        """Call at the start of each backward pass to clear per-pass gradient tracking."""
        for handle in cls.handles.values():
            handle.remove()
        cls.handles.clear()

    @classmethod
    def open(cls, fpath: str, device: str, columns: list[str] | None = None) -> torch.Tensor:
        if fw._current_level != -1:
            primal, tangent = cls.open_primal_tangent(fpath, columns=columns)
            if tangent is not None:
                primal, tangent = torch.from_numpy(primal), torch.from_numpy(tangent)
                return fw.make_dual(primal, tangent)
            else:
                return torch.from_numpy(primal)
        else:
            primal = cls.open_primal(fpath, columns=columns)
            x = torch.from_numpy(primal)

            if torch.is_grad_enabled():
                x.requires_grad_(True)
                handle = x.register_post_accumulate_grad_hook(
                    lambda x: cls.write_backward_gradient(x.grad.detach().cpu().numpy(), fpath, columns=columns)
                )
                cls.handles[fpath] = handle

            return x

    @classmethod
    @abstractmethod
    def open_primal(cls, fpath: str, columns: list[str] | None = None) -> np.ndarray:
        pass

    @classmethod
    @abstractmethod
    def open_primal_tangent(cls, fpath: str, columns: list[str] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        pass

    @classmethod
    def write(
        cls,
        fpath: str,
        input: torch.Tensor | tuple[np.ndarray | torch.Tensor, ...],
        columns: list[str] | None = None,
    ):
        primal, tangent = cls._get_primal_tangent(input)
        if tangent is None:
            cls.write_primal(fpath, primal, columns=columns)
        else:
            cls.write_primal_tangent(fpath, primal, tangent, columns=columns)

    @classmethod
    @abstractmethod
    def write_primal(cls, fpath: str, primal: np.ndarray, columns: list[str] | None = None):
        pass

    @classmethod
    @abstractmethod
    def write_primal_tangent(cls, fpath: str, primal: np.ndarray, tangent: np.ndarray, columns: list[str] | None = None):
        pass

    @classmethod
    def columns(cls, fpath: str) -> list[str]:
        return []

    @classmethod
    @abstractmethod
    def write_named_datasets(cls, fpath: str, datasets: dict[str, np.ndarray]) -> None:
        pass

    @classmethod
    def write_row_store(cls, fpath: str, datasets: dict[str, np.ndarray]) -> None:
        cls.write_named_datasets(fpath, datasets)

    @classmethod
    @abstractmethod
    def open_dataset(cls, fpath: str) -> DatasetView:
        pass

    @classmethod
    def write_backward_gradient(cls, gradient: np.ndarray, fpath: str, columns: list[str] | None = None):
        if fpath not in cls.handles:
            raise RuntimeError(
                f"Trying to write backward gradient for '{fpath}', but no autograd hook is "
                "registered. open() was never called for this file in the current forward pass."
            )
        cls.add_gradient(cls._gradient_path(fpath), GradientBlock(gradient, columns))
        cls.handles[fpath].remove()
        del cls.handles[fpath]

    @classmethod
    def add_gradient(
        cls,
        grad_path: str,
        block: GradientBlock,
        ancestors: frozenset[str] = frozenset(),
    ) -> None:
        primal_path = cls._primal_path(grad_path)
        if block.columns is None:
            block = GradientBlock(block.values, cls.primal_columns(primal_path))

        target = cls._contribution_path(grad_path)
        existing = cls.load_gradient(target) if cls.gradient_exists(target) else None
        cls.store_gradient(target, merge_gradients(existing, block))

        for child in cls.gradient_children(primal_path):
            child_grad_path = cls._gradient_path(child.path)
            if child_grad_path in ancestors:
                continue
            cls.add_gradient(child_grad_path, child.select(block), ancestors | {grad_path})

    @classmethod
    def _contribution_path(cls, grad_path: str) -> str:
        from dsbo.utils.io.workdir import get_subdir, get_file_suffix

        token = re.sub(r"[^A-Za-z0-9._-]", "_", f"{get_subdir()}__{get_file_suffix()}").strip("_")
        if not token:
            return grad_path
        return f"{grad_path}.{token}.contrib"

    @classmethod
    def gradient_exists(cls, grad_path: str) -> bool:
        return os.path.isfile(grad_path)

    @classmethod
    def _gradient_files(cls, grad_path: str) -> list[str]:
        files = [grad_path] if cls.gradient_exists(grad_path) else []
        files.extend(sorted(glob.glob(f"{grad_path}.*.contrib")))
        return files

    @classmethod
    def read_gradient(cls, grad_path: str, columns: list[str] | None = None) -> GradientBlock:
        merged = None
        for path in cls._gradient_files(grad_path):
            merged = merge_gradients(merged, cls.load_gradient(path, columns=columns))
        if merged is None:
            raise FileNotFoundError(f"No gradient has been written for '{grad_path}'.")
        return merged

    @classmethod
    def primal_columns(cls, primal_path: str) -> list[str] | None:
        """Column names of a named primal file, or ``None`` for unnamed files."""
        return None

    @classmethod
    def primal_shape(cls, primal_path: str) -> tuple[int, ...]:
        raise NotImplementedError

    @classmethod
    def primal_column_shapes(cls, primal_path: str, columns: list[str]) -> dict[str, tuple[int, ...]]:
        return {column: cls.primal_shape(primal_path) for column in columns}

    @classmethod
    def reduction_sources(cls, primal_path: str) -> list[str] | None:
        """Resolved source paths of a reduced file, or ``None`` if it is not reduced."""
        return None

    @classmethod
    def validate_primal(cls, primal_path: str) -> None:
        pass

    @classmethod
    def gradient_children(cls, primal_path: str) -> list[GradientChild]:
        """Sources of a reduced primal file paired with their gradient slice selectors."""
        child_paths = cls.reduction_sources(primal_path)
        if not child_paths:
            return []

        parent_columns = cls.primal_columns(primal_path)
        if parent_columns is None:
            entries = []
            for child_path in child_paths:
                shape = cls.primal_shape(child_path)
                scalar = shape == ()
                entries.append((child_path, 1 if scalar else shape[0], scalar))
            boundaries = cls._boundaries([count for _, count, _ in entries])
            return [
                GradientChild(child_path, cls._flat_selector(boundaries[i], boundaries[i + 1], scalar))
                for i, (child_path, _, scalar) in enumerate(entries)
            ]

        counts_by_column: dict[str, list[int]] = {}
        child_columns = []
        for child_path in child_paths:
            columns = cls.primal_columns(child_path) or list(parent_columns)
            child_columns.append(columns)
            for column, shape in cls.primal_column_shapes(child_path, columns).items():
                counts_by_column.setdefault(column, []).append(1 if shape == () else shape[0])

        return [
            GradientChild(child_path, cls._named_selector(parent_columns, child_columns[i], counts_by_column, i))
            for i, child_path in enumerate(child_paths)
        ]

    @classmethod
    def write_gradient(cls, grad_path: str, block: GradientBlock) -> None:
        """Write a gradient block to a single gradient file (backend primitive)."""
        raise NotImplementedError

    @classmethod
    def store_gradient(cls, grad_path: str, block: GradientBlock) -> None:
        tmp_path = f"{grad_path}.tmp"
        cls.write_gradient(tmp_path, block)
        os.replace(tmp_path, grad_path)

    @classmethod
    def gradient_columns(cls, grad_path: str) -> list[str] | None:
        """Stored columns of a gradient file, or ``None`` if it is unnamed (flat)."""
        raise NotImplementedError

    @classmethod
    def read_flat_gradient(cls, grad_path: str) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def read_gradient_columns(cls, grad_path: str, columns: list[str]) -> list[np.ndarray]:
        raise NotImplementedError

    @classmethod
    def gradient_dtype(cls, grad_path: str) -> np.dtype:
        raise NotImplementedError

    @classmethod
    def load_gradient(cls, grad_path: str, columns: list[str] | None = None) -> GradientBlock:
        stored = cls.gradient_columns(grad_path)
        if stored is None:
            if columns is not None:
                raise ValueError(f"Gradient '{grad_path}' was not written with named columns.")
            return GradientBlock(cls.read_flat_gradient(grad_path), None)

        selected = cls._validate_columns(columns) if columns is not None else list(stored)
        present = [column for column in selected if column in stored]
        values_by_column = dict(zip(present, cls.read_gradient_columns(grad_path, present), strict=True))
        missing = [column for column in selected if column not in values_by_column]
        if missing:
            shapes = cls.primal_column_shapes(cls._primal_path(grad_path), missing)
            dtype = cls.gradient_dtype(grad_path)
            for column in missing:
                values_by_column[column] = np.zeros(shapes[column], dtype=dtype)
        return GradientBlock(np.stack([values_by_column[column] for column in selected], axis=0), selected)

    @classmethod
    def _validate_columns(cls, columns: list[str], value: np.ndarray | None = None) -> list[str]:
        if len(set(columns)) != len(columns):
            raise ValueError("Column names must be unique.")
        if value is not None and (value.ndim == 0 or value.shape[0] != len(columns)):
            size = 0 if value.ndim == 0 else value.shape[0]
            raise ValueError(f"Received {len(columns)} columns for an array with size {size} along dimension 0.")
        return columns

    @classmethod
    def _boundaries(cls, counts: list[int]) -> list[int]:
        boundaries = [0]
        for count in counts:
            boundaries.append(boundaries[-1] + count)
        return boundaries

    @classmethod
    def _flat_selector(cls, start: int, stop: int, scalar: bool = False) -> Callable[[GradientBlock], GradientBlock]:
        def select(block: GradientBlock) -> GradientBlock:
            piece = block.values[start:stop]
            if scalar:
                piece = np.asarray(piece).reshape(())
            return GradientBlock(piece, None)

        return select

    @classmethod
    def _named_selector(
        cls,
        parent_columns: list[str],
        child_columns: list[str],
        counts_by_column: dict[str, list[int]],
        index: int,
    ) -> Callable[[GradientBlock], GradientBlock]:
        def select(block: GradientBlock) -> GradientBlock:
            values = []
            for column in child_columns:
                parent_index = parent_columns.index(column)
                boundaries = cls._boundaries(counts_by_column[column])
                values.append(np.split(block.values[parent_index], boundaries[1:-1])[index])
            return GradientBlock(np.stack(values, axis=0), list(child_columns))

        return select

    @classmethod
    def read_backward_gradient(cls, fpath: str) -> torch.Tensor:
        cls.validate_primal(fpath)
        block = cls.read_gradient(cls._gradient_path(fpath), columns=cls.primal_columns(fpath))
        return torch.from_numpy(np.asarray(block.values)).float()

    @classmethod
    def reduce(cls, sources: list[str], destination: str) -> None:
        raise NotImplementedError(f"{cls.__name__} does not support file reduction.")

    @classmethod
    def shape_from_path(cls, path: str) -> tuple[int, ...]:
        return cls.open_primal(path).shape

    @classmethod
    def _gradient_path(cls, fpath: str) -> str:
        base, stem = os.path.splitext(fpath)
        return f"{base}_gradient{stem}"

    @classmethod
    def _primal_path(cls, grad_path: str) -> str:
        base, extension = os.path.splitext(grad_path)
        suffix = "_gradient"
        return f"{base[:-len(suffix)]}{extension}" if base.endswith(suffix) else grad_path

    @classmethod
    def _get_primal_tangent(cls, input: torch.Tensor | tuple[np.ndarray | torch.Tensor, ...]) -> tuple[np.ndarray, np.ndarray | None]:
        if isinstance(input, tuple):
            return input

        primal = fw.unpack_dual(input).primal.detach().cpu().numpy()

        if fw._current_level != -1:
            tangent_val = fw.unpack_dual(input).tangent
            tangent = tangent_val.detach().cpu().numpy() if tangent_val is not None else None
            return primal, tangent
        else:
            return primal, None


class InMemoryFile(AutogradFileInterface):
    """A simple in-memory file interface for reverse-mode AD without actual file I/O.
    Does only work for simple workflows with local execution without dask workers."""
    _storage: dict[str, torch.Tensor] = {}
    _named_storage: dict[str, dict[str, torch.Tensor]] = {}

    @classmethod
    def write_named_datasets(cls, fpath: str, datasets: dict[str, np.ndarray]) -> None:
        stored = cls._named_storage.setdefault(fpath, {})
        stored.update({key: torch.from_numpy(value) for key, value in datasets.items()})

    @classmethod
    def open_dataset(cls, fpath: str) -> DatasetView:
        datasets: dict[str, torch.Tensor] = {}
        named = cls._named_storage.get(fpath)
        if named:
            datasets.update(named)
        if fpath in cls._storage:
            value = cls._storage[fpath]
            if isinstance(value, dict):
                datasets.update(
                    {
                        key: tensor
                        for key, tensor in value.items()
                        if isinstance(tensor, torch.Tensor)
                    }
                )
            else:
                datasets.setdefault("primal", value)
        return InMemoryDatasetView(datasets)

    @classmethod
    def open_primal(cls, fpath: str, columns: list[str] | None = None) -> np.ndarray:
        value = cls._storage[fpath]
        if isinstance(value, dict):
            value = value["primal"]
        return value.cpu().numpy()

    @classmethod
    def open_primal_tangent(cls, fpath: str, columns: list[str] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        value = cls._storage[fpath]
        if isinstance(value, dict):
            primal = value["primal"].cpu().numpy()
            tangent = value.get("tangent")
            tangent = tangent.cpu().numpy() if tangent is not None else None
            return primal, tangent
        return value.cpu().numpy(), None

    @classmethod
    def write_primal(cls, fpath: str, primal: np.ndarray, columns: list[str] | None = None):
        cls._storage[fpath] = torch.from_numpy(primal)

    @classmethod
    def write_primal_tangent(cls, fpath: str, primal: np.ndarray, tangent: np.ndarray, columns: list[str] | None = None):
        cls._storage[fpath] = {
            "primal": torch.from_numpy(primal),
            "tangent": torch.from_numpy(tangent),
        }

    @classmethod
    def columns(cls, fpath: str) -> list[str]:
        value = cls._storage[fpath]
        if isinstance(value, dict):
            return [key for key in value if key != "tangent"]
        return []

    @classmethod
    def primal_columns(cls, primal_path: str) -> list[str] | None:
        value = cls._storage.get(primal_path)
        if isinstance(value, dict):
            return [key for key in value if key != "tangent"]
        return None

    @classmethod
    def gradient_exists(cls, grad_path: str) -> bool:
        return grad_path in cls._storage

    @classmethod
    def store_gradient(cls, grad_path: str, block: GradientBlock) -> None:
        cls._storage[grad_path] = torch.from_numpy(np.asarray(block.values))

    @classmethod
    def load_gradient(cls, grad_path: str, columns: list[str] | None = None) -> GradientBlock:
        return GradientBlock(cls._storage[grad_path].cpu().numpy(), None)
