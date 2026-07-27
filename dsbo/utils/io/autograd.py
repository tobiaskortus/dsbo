import os
import json
import torch
import numpy as np
import h5py
from abc import ABC, abstractmethod
import torch.autograd.forward_ad as fw
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle

from dsbo.utils.io.workdir import AbstractFilePath


class AutogradFileInterface(ABC):
    handles: dict[str, RemovableHandle] = {}
    _gradient_written: set[str] = set()

    @classmethod
    def reducible_extensions(cls) -> frozenset[str]:
        return frozenset()

    @classmethod
    def reset_gradient_state(cls):
        """Call at the start of each backward pass to clear per-pass gradient tracking."""
        cls._gradient_written.clear()
        for handle in cls.handles.values():
            handle.remove()
        cls.handles.clear()

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
    def reduce(cls, sources: list[str], destination: str) -> None:
        raise NotImplementedError(f"{cls.__name__} does not support file reduction.")

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


class HDF5File(AutogradFileInterface):
    DATASETS = ("primal", "tangent", "gradient")
    VDS_SOURCES_ATTRIBUTE = "dsbo_vds_sources"

    @classmethod
    def reducible_extensions(cls) -> frozenset[str]:
        return frozenset({".h5", ".hdf5"})

    @classmethod
    def _read_dataset(cls, file: h5py.File, name: str) -> np.ndarray:
        cls._validate_vds_sources(file)
        dataset = file[name]
        return np.asarray(dataset[()] if dataset.shape == () else dataset[:])

    @classmethod
    def open_primal(cls, fpath: str) -> np.ndarray:
        with h5py.File(fpath, "r") as file:
            return cls._read_dataset(file, "primal")

    @classmethod
    def open_primal_tangent(cls, fpath: str) -> tuple[np.ndarray, np.ndarray | None]:
        with h5py.File(fpath, "r") as file:
            if "primal" not in file:
                raise ValueError("Saved HDF5 file must contain a 'primal' dataset.")
            primal = cls._read_dataset(file, "primal")
            tangent = cls._read_dataset(file, "tangent") if "tangent" in file else None
        return primal, tangent

    @classmethod
    def write_primal(cls, fpath: str, primal: np.ndarray):
        cls._write_datasets(fpath, primal=primal)

    @classmethod
    def write_primal_tangent(cls, fpath: str, primal: np.ndarray, tangent: np.ndarray):
        cls._write_datasets(fpath, primal=primal, tangent=tangent)

    @classmethod
    def _load_gradient(cls, grad_path: str) -> np.ndarray:
        with h5py.File(grad_path, "r") as file:
            return cls._read_dataset(file, "gradient")

    @classmethod
    def _save_gradient(cls, grad_path: str, gradient: np.ndarray):
        cls._write_datasets(grad_path, gradient=gradient)

    @classmethod
    def read_backward_gradient(cls, fpath: str) -> torch.Tensor:
        return torch.from_numpy(cls._load_gradient(cls._gradient_path(fpath))).float()

    @classmethod
    def _write_datasets(cls, fpath: str, **datasets: np.ndarray) -> None:
        os.makedirs(os.path.dirname(fpath) or ".", exist_ok=True)
        with h5py.File(fpath, "w", libver="latest") as file:
            for name, value in datasets.items():
                options = {} if value.ndim == 0 else {"compression": "gzip"}
                file.create_dataset(name, data=value, **options)

    @classmethod
    def _validate_vds_sources(cls, file: h5py.File) -> None:
        manifest = file.attrs.get(cls.VDS_SOURCES_ATTRIBUTE)
        if manifest is None:
            return
        source_paths = json.loads(manifest)
        file_dir = os.path.dirname(os.path.abspath(file.filename))
        missing = [
            source for source in source_paths
            if not os.path.isfile(os.path.join(file_dir, source))
        ]
        if missing:
            raise FileNotFoundError(
                f"HDF5 virtual dataset '{file.filename}' is missing source '{missing[0]}'."
            )

    @classmethod
    def dependencies(cls, fpath: str) -> list[str]:
        with h5py.File(fpath, "r") as file:
            manifest = file.attrs.get(cls.VDS_SOURCES_ATTRIBUTE)
            if manifest is None:
                return []
            file_dir = os.path.dirname(os.path.abspath(fpath))
            return [
                os.path.normpath(os.path.join(file_dir, source))
                for source in json.loads(manifest)
            ]

    @classmethod
    def validate(cls, fpath: str) -> None:
        with h5py.File(fpath, "r") as file:
            cls._validate_vds_sources(file)

    @classmethod
    def reduce(cls, sources: list[str], destination: str) -> None:
        if not sources:
            raise ValueError("HDF5 reduction requires at least one source file.")

        metadata = cls._collect_metadata(sources)
        dataset_names = set(metadata[0])
        if any(set(m) != dataset_names for m in metadata[1:]):
            raise ValueError("All HDF5 reduction sources must expose identical datasets.")

        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
        destination_dir = os.path.dirname(os.path.abspath(destination))
        relative_sources = [
            os.path.relpath(os.path.abspath(source), destination_dir)
            for source in sources
        ]

        with h5py.File(destination, "w", libver="latest") as output:
            output.attrs[cls.VDS_SOURCES_ATTRIBUTE] = json.dumps(relative_sources)
            for dataset_name in sorted(dataset_names):
                shapes = [m[dataset_name][0] for m in metadata]
                dtype = metadata[0][dataset_name][1]
                cls._write_vds(output, dataset_name, relative_sources, shapes, dtype)

    @classmethod
    def _collect_metadata(cls, sources: list[str]) -> list[dict[str, tuple[tuple[int, ...], str]]]:
        metadata = []
        for source in sources:
            if not os.path.isfile(source):
                raise FileNotFoundError(f"Missing registered batch input: {source}")
            ext = os.path.splitext(source)[1].lower()
            if ext not in cls.reducible_extensions():
                raise TypeError(
                    f"HDF5 reduction does not support '{source}'. "
                    f"Only {', '.join(sorted(cls.reducible_extensions()))} files can be reduced."
                )
            try:
                with h5py.File(source, "r") as file:
                    if "primal" not in file:
                        raise ValueError(f"HDF5 source '{source}' has no 'primal' dataset.")
                    metadata.append({
                        name: (file[name].shape, file[name].dtype)
                        for name in cls.DATASETS
                        if name in file
                    })
            except OSError:
                raise TypeError(
                    f"HDF5 reduction source '{source}' is not a valid HDF5 file. "
                    f"The file may be a different format (e.g. PyTorch .pt, JSON, etc.)."
                )
        return metadata

    @classmethod
    def _write_vds(
        cls,
        output: h5py.File,
        dataset_name: str,
        relative_sources: list[str],
        shapes: list[tuple[int, ...]],
        dtype: str,
    ) -> None:
        if shapes[0] == ():
            cls._validate_shapes(dataset_name, shapes)
            layout = h5py.VirtualLayout(shape=(len(shapes),), dtype=dtype)
            for i, src in enumerate(relative_sources):
                layout[i] = h5py.VirtualSource(src, dataset_name, shape=())
        else:
            cls._validate_shapes(dataset_name, shapes)
            total = sum(s[0] for s in shapes)
            layout = h5py.VirtualLayout(shape=(total, *shapes[0][1:]), dtype=dtype)
            offset = 0
            for src, shape in zip(relative_sources, shapes):
                layout[offset:offset + shape[0]] = h5py.VirtualSource(src, dataset_name, shape=shape)
                offset += shape[0]
        output.create_virtual_dataset(dataset_name, layout)

    @classmethod
    def _validate_shapes(cls, dataset_name: str, shapes: list[tuple[int, ...]]) -> None:
        if shapes[0] == ():
            if any(s != () for s in shapes[1:]):
                raise ValueError(
                    f"Cannot reduce HDF5 dataset '{dataset_name}' with incompatible shapes: {shapes}."
                )
        elif any(s[1:] != shapes[0][1:] for s in shapes[1:]):
            raise ValueError(
                f"Cannot reduce HDF5 dataset '{dataset_name}' with incompatible shapes: {shapes}."
            )

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
