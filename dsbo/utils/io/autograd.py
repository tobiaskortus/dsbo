import os
import torch
import numpy as np
from abc import ABC, abstractmethod
import torch.autograd.forward_ad as fw
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle

from dsbo.utils.io.workdir import AbstractFilePath


class AutogradFileInterface(ABC):
    handles: dict[str, RemovableHandle] = {}
    _gradient_written: set[str] = set()

    @classmethod
    def reset_gradient_state(cls):
        """Call at the start of each backward pass to clear per-pass gradient tracking."""
        cls._gradient_written.clear()

    @classmethod
    def open(cls, fpath: str, device: str) -> torch.Tensor:
        if fw._current_level != -1:
            primal, tangent = cls.open_primal_tangent(fpath)
            if tangent is not None:
                primal, tangent = torch.from_numpy(primal), torch.from_numpy(tangent)
                return fw.make_dual(primal, tangent)
            else:
                return torch.from_numpy(primal)
        else:
            primal = cls.open_primal(fpath)
            x = torch.from_numpy(primal)

            if torch.is_grad_enabled():
                x.requires_grad_(True)
                handle = x.register_post_accumulate_grad_hook(
                    lambda x: cls.write_backward_gradient(x.grad.detach().cpu().numpy(), fpath)
                )
                cls.handles[fpath] = handle

            return x

    @classmethod
    @abstractmethod
    def open_primal(cls, fpath: str) -> np.ndarray:
        pass

    @classmethod
    @abstractmethod
    def open_primal_tangent(cls, fpath: str) -> tuple[np.ndarray, np.ndarray | None]:
        pass

    @classmethod
    def write(cls, fpath: str, input: torch.Tensor | tuple[np.ndarray | torch.Tensor, ...]):
        primal, tangent = cls._get_primal_tangent(input)
        if tangent is None:
            cls.write_primal(fpath, primal)
        else:
            cls.write_primal_tangent(fpath, primal, tangent)

    @classmethod
    @abstractmethod
    def write_primal(cls, fpath: str, primal: np.ndarray):
        pass

    @classmethod
    @abstractmethod
    def write_primal_tangent(cls, fpath: str, primal: np.ndarray, tangent: np.ndarray):
        pass

    @classmethod
    def write_backward_gradient(cls, gradient: np.ndarray, fpath: str):
        if fpath not in cls.handles:
            raise RuntimeError(
                f"Trying to write backward gradient for '{fpath}', but no autograd hook is "
                "registered. open() was never called for this file in the current forward pass."
            )
        grad_path = cls._gradient_path(fpath)
        if grad_path in cls._gradient_written:
            gradient = gradient + cls._load_gradient(grad_path)
        cls._gradient_written.add(grad_path)
        cls._save_gradient(grad_path, gradient)
        cls.handles[fpath].remove()
        del cls.handles[fpath]

    @classmethod
    @abstractmethod
    def _load_gradient(cls, grad_path: str) -> np.ndarray:
        pass

    @classmethod
    @abstractmethod
    def _save_gradient(cls, grad_path: str, gradient: np.ndarray):
        pass

    @classmethod
    @abstractmethod
    def read_backward_gradient(cls, fpath: str) -> torch.Tensor:
        pass

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
            tangent = fw.unpack_dual(input).tangent.detach().cpu().numpy()
            return primal, tangent
        else:
            return primal, None


class NumpyFile(AutogradFileInterface):
    @classmethod
    def open_primal(cls, fpath: str) -> np.ndarray:
        return np.load(fpath)["primal"]

    @classmethod
    def open_primal_tangent(cls, fpath: str) -> tuple[np.ndarray, np.ndarray | None]:
        x = np.load(fpath)
        assert "primal" in x, "Saved array must contain at least a 'primal' key"
        return x["primal"], x.get("tangent")

    @classmethod
    def write_primal(cls, fpath: str, primal: np.ndarray):
        np.savez_compressed(fpath, primal=primal)

    @classmethod
    def write_primal_tangent(cls, fpath: str, primal: np.ndarray, tangent: np.ndarray):
        np.savez_compressed(fpath, primal=primal, tangent=tangent)

    @classmethod
    def _load_gradient(cls, grad_path: str) -> np.ndarray:
        return np.load(grad_path)["gradient"]

    @classmethod
    def _save_gradient(cls, grad_path: str, gradient: np.ndarray):
        np.savez_compressed(grad_path, gradient=gradient)

    @classmethod
    def read_backward_gradient(cls, fpath: str) -> torch.Tensor:
        return torch.from_numpy(np.load(cls._gradient_path(fpath))["gradient"]).float()


class InMemoryFile(AutogradFileInterface):
    """A simple in-memory file interface for reverse-mode AD without actual file I/O.
    Does only work for simple workflows with local execution without dask workers."""
    _storage: dict[str, torch.Tensor] = {}

    @classmethod
    def open_primal(cls, fpath: str) -> np.ndarray:
        value = cls._storage[fpath]
        if isinstance(value, dict):
            value = value["primal"]
        return value.cpu().numpy()

    @classmethod
    def open_primal_tangent(cls, fpath: str) -> tuple[np.ndarray, np.ndarray | None]:
        value = cls._storage[fpath]
        if isinstance(value, dict):
            primal = value["primal"].cpu().numpy()
            tangent = value.get("tangent")
            tangent = tangent.cpu().numpy() if tangent is not None else None
            return primal, tangent
        return value.cpu().numpy(), None

    @classmethod
    def write_primal(cls, fpath: str, primal: np.ndarray):
        cls._storage[fpath] = torch.from_numpy(primal)

    @classmethod
    def write_primal_tangent(cls, fpath: str, primal: np.ndarray, tangent: np.ndarray):
        cls._storage[fpath] = {
            "primal": torch.from_numpy(primal),
            "tangent": torch.from_numpy(tangent),
        }

    @classmethod
    def _load_gradient(cls, grad_path: str) -> np.ndarray:
        return cls._storage[grad_path]

    @classmethod
    def _save_gradient(cls, grad_path: str, gradient: np.ndarray):
        cls._storage[grad_path] = torch.from_numpy(gradient)

    @classmethod
    def read_backward_gradient(cls, fpath: str) -> torch.Tensor:
        return torch.tensor(cls._storage[cls._gradient_path(fpath)]).float()


class AutogradFileStorage(AutogradFileInterface):
    @classmethod
    @abstractmethod
    def to_data_loader(
        cls,
        data_fpath_abstract: AbstractFilePath,
        cond_fpath_abstract: AbstractFilePath,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
       pass