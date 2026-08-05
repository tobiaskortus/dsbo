import os
import json
import torch
import numpy as np
import h5py
from abc import ABC, abstractmethod
import torch.autograd.forward_ad as fw
from torch.utils.hooks import RemovableHandle

from dsbo.utils.io.datasets.views import DatasetView, HDF5DatasetView, InMemoryDatasetView


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
        pass

    @classmethod
    @abstractmethod
    def write_named_datasets(cls, fpath: str, datasets: dict[str, np.ndarray]) -> None:
        #TODO: I dont really like this conceptually but lets keep it for now.
        pass

    @classmethod
    @abstractmethod
    def open_dataset(cls, fpath: str, prefix: str | None = None) -> DatasetView:
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
    @abstractmethod
    def _load_gradient(cls, grad_path: str, columns: list[str] | None = None) -> np.ndarray:
        pass

    @classmethod
    @abstractmethod
    def _save_gradient(cls, grad_path: str, gradient: np.ndarray, columns: list[str] | None = None):
        pass

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

    # This attribute is used to reconstruct the correct order of the data when the columns argument is omitted while
    # loading a file saved with columns.
    COLUMNS_ATTRIBUTE = "dsbo_columns"

    @classmethod
    def reducible_extensions(cls) -> frozenset[str]:
        return frozenset({".h5", ".hdf5"})

    @classmethod
    def _read_dataset(cls, file: h5py.File, name: str, columns: list[str] | None = None) -> np.ndarray:
        cls._validate_vds_sources(file)
        dataset = file[name]
        if isinstance(dataset, h5py.Dataset):
            if columns is not None:
                raise ValueError(f"HDF5 dataset '{name}' was not written with named columns.")
            return np.asarray(dataset[()] if dataset.shape == () else dataset[:])

        columns = cls._validate_columns(columns) if columns is not None else cls._stored_columns(dataset)
        missing = [column for column in columns if column not in dataset]
        if missing:
            raise KeyError(f"HDF5 dataset '{name}' has no column '{missing[0]}'.")
        values = [
            np.asarray(value[()] if value.shape == () else value[:])
            for value in (dataset[column] for column in columns)
        ]
        return np.stack(values, axis=0)

    @classmethod
    def open_primal(cls, fpath: str, columns: list[str] | None = None) -> np.ndarray:
        with h5py.File(fpath, "r") as file:
            return cls._read_dataset(file, "primal", columns=columns)

    @classmethod
    def open_primal_tangent(cls, fpath: str, columns: list[str] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        with h5py.File(fpath, "r") as file:
            if "primal" not in file:
                raise ValueError("Saved HDF5 file must contain a 'primal' dataset.")
            primal = cls._read_dataset(file, "primal", columns=columns)
            tangent = cls._read_dataset(file, "tangent", columns=columns) if "tangent" in file else None
        return primal, tangent

    @classmethod
    def write_primal(cls, fpath: str, primal: np.ndarray, columns: list[str] | None = None):
        cls._write_datasets(fpath, columns=columns, primal=primal)

    @classmethod
    def write_primal_tangent(cls, fpath: str, primal: np.ndarray, tangent: np.ndarray, columns: list[str] | None = None):
        cls._write_datasets(fpath, columns=columns, primal=primal, tangent=tangent)

    @classmethod
    def write_named_datasets(cls, fpath: str, datasets: dict[str, np.ndarray]) -> None:
        if not datasets:
            return
        os.makedirs(os.path.dirname(fpath) or ".", exist_ok=True)
        with h5py.File(fpath, "a", libver="latest") as file:
            for path, value in datasets.items():
                parent_path, name = os.path.split(path)
                if not name:
                    raise ValueError(f"Invalid HDF5 dataset path: '{path}'.")
                parent = file.require_group(parent_path) if parent_path else file
                if name in parent:
                    del parent[name]
                cls._create_dataset(parent, name, value)

    @classmethod
    def open_dataset(cls, fpath: str, prefix: str | None = None) -> DatasetView:
        cls.validate(fpath)
        return HDF5DatasetView(fpath, prefix=prefix)

    @classmethod
    def columns(cls, fpath: str) -> list[str]:
        with h5py.File(fpath, "r") as file:
            primal = file["primal"]
            return cls._stored_columns(primal) if isinstance(primal, h5py.Group) else []

    @classmethod
    def _load_gradient(cls, grad_path: str, columns: list[str] | None = None) -> np.ndarray:
        with h5py.File(grad_path, "r") as file:
            gradient = file["gradient"]
            if isinstance(gradient, h5py.Dataset):
                return cls._read_dataset(file, "gradient", columns=columns)

            columns = (
                cls._validate_columns(columns)
                if columns is not None else cls._gradient_primal_columns(grad_path)
            )
            template = gradient[cls._stored_columns(gradient)[0]]
            template_value = np.asarray(template[()] if template.shape == () else template[:])
            values = []
            for column in columns:
                if column in gradient:
                    value = gradient[column]
                    values.append(np.asarray(value[()] if value.shape == () else value[:]))
                else:
                    values.append(np.zeros_like(template_value))
            return np.stack(values, axis=0)

    @classmethod
    def _save_gradient(cls, grad_path: str, gradient: np.ndarray, columns: list[str] | None = None):
        if columns is None:
            columns = cls._gradient_primal_columns(grad_path)
        if columns is not None and grad_path in cls._gradient_written:
            selected = cls._validate_columns(columns, gradient)
            with h5py.File(grad_path, "a", libver="latest") as file:
                group = file["gradient"]
                if not isinstance(group, h5py.Group):
                    raise ValueError("Cannot accumulate named gradients into an unnamed gradient.")
                stored = cls._stored_columns(group)
                for column, value in zip(selected, gradient, strict=True):
                    if column in group:
                        del group[column]
                    cls._create_dataset(group, column, value)
                    if column not in stored:
                        stored.append(column)
                group.attrs[cls.COLUMNS_ATTRIBUTE] = json.dumps(stored)
            return
        cls._write_datasets(grad_path, columns=columns, gradient=gradient)

    @classmethod
    def _gradient_primal_columns(cls, grad_path: str) -> list[str] | None:
        base, extension = os.path.splitext(grad_path)
        suffix = "_gradient"
        if not base.endswith(suffix):
            return None
        primal_path = f"{base[:-len(suffix)]}{extension}"
        with h5py.File(primal_path, "r") as file:
            primal = file["primal"]
            return cls._stored_columns(primal) if isinstance(primal, h5py.Group) else None

    @classmethod
    def read_backward_gradient(cls, fpath: str) -> torch.Tensor:
        grad_path = cls._gradient_path(fpath)
        with h5py.File(fpath, "r") as primal_file:
            cls._validate_vds_sources(primal_file)
            primal = primal_file["primal"]
            if isinstance(primal, h5py.Dataset):
                return torch.from_numpy(cls._load_gradient(grad_path)).float()

            with h5py.File(grad_path, "r") as gradient_file:
                gradient = gradient_file["gradient"]
                if isinstance(gradient, h5py.Dataset):
                    value = np.asarray(gradient[()] if gradient.shape == () else gradient[:])
                    return torch.from_numpy(value).float()
                gradient_columns = cls._stored_columns(gradient)
                dtype = gradient[gradient_columns[0]].dtype
                values = []
                for column in cls._stored_columns(primal):
                    if column in gradient:
                        value = gradient[column]
                        values.append(np.asarray(value[()] if value.shape == () else value[:]))
                    else:
                        values.append(np.zeros(primal[column].shape, dtype=dtype))
                return torch.from_numpy(np.stack(values, axis=0)).float()

    @classmethod
    def _write_datasets(cls, fpath: str, columns: list[str] | None = None, **datasets: np.ndarray) -> None:
        if columns is not None:
            cls._validate_columns(columns)
            for value in datasets.values():
                cls._validate_columns(columns, value)

        os.makedirs(os.path.dirname(fpath) or ".", exist_ok=True)
        with h5py.File(fpath, "w", libver="latest") as file:
            for name, value in datasets.items():
                if columns is None:
                    cls._create_dataset(file, name, value)
                    continue

                group = file.create_group(name, track_order=True)
                group.attrs[cls.COLUMNS_ATTRIBUTE] = json.dumps(columns)
                for column, column_value in zip(columns, value, strict=True):
                    cls._create_dataset(group, column, column_value)

    @classmethod
    def _create_dataset(cls, parent: h5py.File | h5py.Group, name: str, value: np.ndarray) -> None:
        options = {} if value.ndim == 0 else {"compression": "gzip"}
        parent.create_dataset(name, data=value, **options)

    @classmethod
    def _validate_columns(cls, columns: list[str], value: np.ndarray | None = None) -> list[str]:
        if any(
            not isinstance(column, str)
            or not column
            or "/" in column
            or "\x00" in column
            or column == "."
            for column in columns
        ):
            raise ValueError("Column names must be valid, non-empty HDF5 names without '/'.")
        if len(set(columns)) != len(columns):
            raise ValueError("Column names must be unique.")
        if value is not None and (value.ndim == 0 or value.shape[0] != len(columns)):
            size = 0 if value.ndim == 0 else value.shape[0]
            raise ValueError(f"Received {len(columns)} columns for an array with size {size} along dimension 0.")
        return columns

    @classmethod
    def _stored_columns(cls, group: h5py.Group) -> list[str]:
        columns = group.attrs.get(cls.COLUMNS_ATTRIBUTE)
        return json.loads(columns) if columns is not None else list(group.keys())

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
            with h5py.File(sources[0], "r") as first_source:
                cls._copy_group_attributes(first_source, output)
            for dataset_name in sorted(dataset_names):
                shapes = [m[dataset_name][0] for m in metadata]
                dtype = metadata[0][dataset_name][1]
                cls._write_vds(output, dataset_name, relative_sources, shapes, dtype)

    @classmethod
    def _copy_group_attributes(cls, source: h5py.Group, target: h5py.Group) -> None:
        for name, value in source.items():
            if not isinstance(value, h5py.Group):
                continue
            if cls.COLUMNS_ATTRIBUTE in value.attrs:
                target_group = target.require_group(name)
                target_group.attrs[cls.COLUMNS_ATTRIBUTE] = value.attrs[cls.COLUMNS_ATTRIBUTE]
            cls._copy_group_attributes(value, target[name] if name in target else target.create_group(name))

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
                    source_metadata: dict[str, tuple[tuple[int, ...], str]] = {}
                    file.visititems(
                        lambda name, obj: source_metadata.update([(name, (obj.shape, obj.dtype))])
                        if isinstance(obj, h5py.Dataset) else None
                    )
                    if not source_metadata:
                        raise ValueError(f"HDF5 source '{source}' contains no datasets.")
                    metadata.append(source_metadata)
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
        parent, _ = os.path.split(dataset_name)
        if parent:
            output.require_group(parent)
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

    @classmethod
    def shape_from_path(cls, path: str) -> tuple[int, ...]:
        with h5py.File(path, "r") as f:
            return f["primal"].shape

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
    def open_dataset(cls, fpath: str, prefix: str | None = None) -> DatasetView:
        if prefix is not None:
            return InMemoryDatasetView(cls._named_storage[fpath], prefix=prefix)

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
            return [k for k in value if k != "tangent"]
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
