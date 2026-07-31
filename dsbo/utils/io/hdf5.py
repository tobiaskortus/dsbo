import json
import os

import h5py
import numpy as np
import torch

from dsbo.utils.io.autograd import AutogradFileInterface


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
                for name in cls.DATASETS:
                    if name in first_source and isinstance(first_source[name], h5py.Group):
                        group = output.require_group(name)
                        group.attrs[cls.COLUMNS_ATTRIBUTE] = json.dumps(cls._stored_columns(first_source[name]))
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
                    source_metadata = {}
                    for name in cls.DATASETS:
                        if name not in file:
                            continue
                        value = file[name]
                        if isinstance(value, h5py.Dataset):
                            source_metadata[name] = (value.shape, value.dtype)
                            continue
                        for column in cls._stored_columns(value):
                            column_value = value[column]
                            if not isinstance(column_value, h5py.Dataset):
                                raise ValueError(f"HDF5 source '{source}' has invalid column '{name}/{column}'.")
                            source_metadata[f"{name}/{column}"] = (column_value.shape, column_value.dtype)
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
