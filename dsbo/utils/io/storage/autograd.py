import os
import torch
import numpy as np
from abc import ABC, abstractmethod
import torch.autograd.forward_ad as fw
from torch.utils.hooks import RemovableHandle

from dsbo.utils.io.datasets.views import DatasetView, InMemoryDatasetView


class AutogradFileInterface(ABC):
    handles: dict[str, RemovableHandle] = {}
    _gradient_written: set[str] = set()
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
        cls._gradient_written.clear()
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
        grad_path = cls._gradient_path(fpath)
        if grad_path in cls._gradient_written:
            gradient = gradient + cls._load_gradient(grad_path, columns=columns)
        cls._save_gradient(grad_path, gradient, columns=columns)
        cls._gradient_written.add(grad_path)
        cls.handles[fpath].remove()
        del cls.handles[fpath]

    @classmethod
    @abstractmethod
    def _load_gradient(cls, grad_path: str, columns: list[str] | None = None) -> np.ndarray:
        pass

    @classmethod
    @abstractmethod
    def _save_gradient(cls, grad_path: str, gradient: np.ndarray, columns: list[str] | None = None):
        pass

    @classmethod
    @abstractmethod
    def read_backward_gradient(cls, fpath: str) -> torch.Tensor:
        pass

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
    def _load_gradient(cls, grad_path: str, columns: list[str] | None = None) -> np.ndarray:
        return cls._storage[grad_path]

    @classmethod
    def _save_gradient(cls, grad_path: str, gradient: np.ndarray, columns: list[str] | None = None):
        cls._storage[grad_path] = torch.from_numpy(gradient)

    @classmethod
    def read_backward_gradient(cls, fpath: str) -> torch.Tensor:
        return torch.tensor(cls._storage[cls._gradient_path(fpath)]).float()
