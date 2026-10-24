import json
import os
import uuid
from collections.abc import Iterator

import h5py
import numpy as np

from dsbo.utils.io.storage.autograd import AutogradFileInterface
from dsbo.utils.io.datasets.views import (
    STORE_COLUMN,
    STORE_FLAT,
    STORE_ROW,
    DatasetView,
    HDF5ColumnStoreDatasetView,
    HDF5DatasetView,
    HDF5RowStoreDatasetView,
)


class HDF5Statistics:
    STATS_COUNT_ATTRIBUTE = "dsbo_stats_count"
    STATS_MEAN_ATTRIBUTE = "dsbo_stats_mean"
    STATS_SQUARED_DEVIATION_ATTRIBUTE = "dsbo_stats_squared_deviation"

    @classmethod
    def _statistics(cls, value: np.ndarray) -> tuple[int, np.ndarray, np.ndarray] | None:
        value = np.asarray(value)
        if value.ndim not in (1, 2) or value.shape[0] == 0 or value.dtype.kind not in "biuf":
            return None
        values = value.astype(np.float64, copy=False)
        mean = values.mean(axis=0)
        return value.shape[0], mean, np.square(values - mean).sum(axis=0)

    @classmethod
    def _write_statistics(cls, dataset: h5py.Dataset, statistics: tuple[int, np.ndarray, np.ndarray]) -> None:
        count, mean, squared_deviation = statistics
        dataset.attrs[cls.STATS_COUNT_ATTRIBUTE] = count
        dataset.attrs[cls.STATS_MEAN_ATTRIBUTE] = mean
        dataset.attrs[cls.STATS_SQUARED_DEVIATION_ATTRIBUTE] = squared_deviation

    @classmethod
    def _read_statistics(cls, dataset: h5py.Dataset) -> tuple[int, np.ndarray, np.ndarray] | None:
        attributes = dataset.attrs
        if any(
            name not in attributes
            for name in (
                cls.STATS_COUNT_ATTRIBUTE,
                cls.STATS_MEAN_ATTRIBUTE,
                cls.STATS_SQUARED_DEVIATION_ATTRIBUTE,
            )
        ):
            return None
        return (
            int(attributes[cls.STATS_COUNT_ATTRIBUTE]),
            np.asarray(attributes[cls.STATS_MEAN_ATTRIBUTE]),
            np.asarray(attributes[cls.STATS_SQUARED_DEVIATION_ATTRIBUTE]),
        )

    @staticmethod
    def _combine_statistics(values: list[tuple[int, np.ndarray, np.ndarray] | None]) -> tuple[int, np.ndarray, np.ndarray] | None:
        if not values or any(value is None for value in values):
            return None
        count, mean, squared_deviation = values[0]
        for batch_count, batch_mean, batch_squared_deviation in values[1:]:
            total = count + batch_count
            delta = batch_mean - mean
            mean += delta * batch_count / total
            squared_deviation += (
                batch_squared_deviation
                + delta * delta * count * batch_count / total
            )
            count = total
        return count, mean, squared_deviation


class HDF5Writer(HDF5Statistics):
    VDS_SOURCES_ATTRIBUTE = "dsbo_vds_sources"
    PARTITION_SIZES_ATTRIBUTE = "dsbo_partition_sizes"
    ROW_STORE_DATASET = "__dsbo_row_store__"
    ROW_STORE_KEYS_ATTRIBUTE = "dsbo_keys"
    STORE_ATTRIBUTE = "dsbo_store"
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
            file.attrs[cls.STORE_ATTRIBUTE] = STORE_FLAT
            for path, value in datasets.items():
                parent_path, name = os.path.split(path)
                if not name:
                    raise ValueError(f"Invalid HDF5 dataset path: '{path}'.")
                parent = file.require_group(parent_path) if parent_path else file
                if name in parent:
                    del parent[name]
                cls._create_dataset(parent, name, value)

    @classmethod
    def write_row_store(cls, fpath: str, datasets: dict[str, np.ndarray]) -> None:
        keys, values = cls._prepare_row_store(datasets)
        os.makedirs(os.path.dirname(fpath) or ".", exist_ok=True)
        with h5py.File(fpath, "w", libver="latest") as file:
            file.attrs[cls.STORE_ATTRIBUTE] = STORE_ROW
            cls._create_dataset(file, cls.ROW_STORE_DATASET, values)
            file[cls.ROW_STORE_DATASET].attrs[cls.ROW_STORE_KEYS_ATTRIBUTE] = json.dumps(keys)

    @classmethod
    def open_dataset(cls, fpath: str) -> DatasetView:
        file = h5py.File(fpath, "r")
        try:
            cls._validate_vds_sources(file)
            store = file.attrs.get(cls.STORE_ATTRIBUTE)
            if store == STORE_ROW:
                return HDF5RowStoreDatasetView(file)
            if store == STORE_COLUMN:
                return HDF5ColumnStoreDatasetView(fpath, file=file)
            if store == STORE_FLAT:
                return HDF5DatasetView(fpath, file=file)
            raise ValueError(
                f"HDF5 file '{fpath}' has no '{cls.STORE_ATTRIBUTE}' store-type tag."
            )
        except Exception:
            file.close()
            raise

    @classmethod
    def columns(cls, fpath: str) -> list[str]:
        with h5py.File(fpath, "r") as file:
            primal = file["primal"]
            return cls._stored_columns(primal) if isinstance(primal, h5py.Group) else []

    @classmethod
    def gradient_file_contributions(
        cls, fpath: str, gradient_path: str,
    ) -> Iterator[tuple[str, str | np.ndarray, list[str] | None]]:
        """Validate a native adjoint and transpose DSBO concatenations, if any.

        Physical primals retain the native file itself as their contribution;
        validation reads dataset metadata only and requires no DSBO store tag.
        """
        with h5py.File(fpath, "r") as primal_file, h5py.File(gradient_path, "r") as gradient_file:
            cls._validate_vds_sources(primal_file)
            primal = primal_file.get("primal")
            gradient = gradient_file.get("gradient")
            columns = None
            if isinstance(primal, h5py.Dataset) and isinstance(gradient, h5py.Dataset):
                pairs = [(primal, gradient)]
            elif isinstance(primal, h5py.Group) and isinstance(gradient, h5py.Group):
                columns = cls._validate_columns(cls._stored_columns(gradient))
                if not columns or set(columns) != set(gradient.keys()):
                    raise ValueError("Native HDF5 gradient columns must name every stored gradient column.")
                unknown = [column for column in columns if column not in cls._stored_columns(primal)]
                if unknown:
                    raise ValueError(f"HDF5 primal has no gradient column '{unknown[0]}'.")
                pairs = [(primal[column], gradient[column]) for column in columns]
            else:
                raise ValueError("Native HDF5 '/gradient' layout must match the primal dataset or named-column group.")
            for primal_dataset, gradient_dataset in pairs:
                if not isinstance(primal_dataset, h5py.Dataset) or not isinstance(gradient_dataset, h5py.Dataset):
                    raise ValueError("Native HDF5 gradient columns must be datasets.")
                if gradient_dataset.shape != primal_dataset.shape:
                    raise ValueError(
                        f"Gradient shape {gradient_dataset.shape} does not match HDF5 dataset "
                        f"'{primal_dataset.name}' shape {primal_dataset.shape}."
                    )
                if gradient_dataset.dtype.kind not in "biufc":
                    raise ValueError("Native HDF5 gradient datasets must have numeric values.")
            virtual = cls.VDS_SOURCES_ATTRIBUTE in primal_file.attrs

        if not virtual:
            yield fpath, gradient_path, columns
            return

        gradient = cls.load_gradient(gradient_path, columns=columns)
        yield from cls._gradient_contributions(fpath, gradient, columns)

    @classmethod
    def _partition_offsets(cls, dataset: h5py.Dataset, num_sources: int) -> np.ndarray:
        sizes = json.loads(dataset.attrs[cls.PARTITION_SIZES_ATTRIBUTE])
        if len(sizes) != num_sources or sum(sizes) != dataset.shape[0]:
            raise ValueError(f"Invalid HDF5 concatenation partitions for '{dataset.name}'.")
        return np.cumsum([0, *sizes])

    @classmethod
    def _gradient_contributions(
        cls,
        fpath: str,
        gradient: np.ndarray,
        columns: list[str] | None = None,
    ) -> Iterator[tuple[str, np.ndarray, list[str] | None]]:
        """Transpose DSBO virtual concatenation into physical-primal adjoints.

        Named columns occupy tensor axis 0, but each column's physical
        dataset is concatenated along its own leading (sample) axis.
        Repeated sources intentionally yield separate additive contributions.
        """
        with h5py.File(fpath, "r") as file:
            sources = cls._vds_sources(file)
            if not sources:
                yield fpath, gradient, columns
                return
            primal = file["primal"]
            if isinstance(primal, h5py.Group):
                columns = columns if columns is not None else cls._stored_columns(primal)
                cls._validate_columns(columns, gradient)
                datasets = [primal[column] for column in columns]
                values = list(gradient)
            else:
                if columns is not None:
                    raise ValueError("HDF5 dataset 'primal' was not written with named columns.")
                datasets, values = [primal], [gradient]

            partitions = []
            for dataset, value in zip(datasets, values, strict=True):
                if value.shape != dataset.shape:
                    raise ValueError(
                        f"Gradient shape {value.shape} does not match HDF5 dataset "
                        f"'{dataset.name}' shape {dataset.shape}."
                    )
                partitions.append(cls._partition_offsets(dataset, len(sources)))

            contributions = []
            for index, source in enumerate(sources):
                with h5py.File(source, "r") as source_file:
                    pieces = [
                        value[offsets[index]:offsets[index + 1]].reshape(source_file[dataset.name].shape)
                        for dataset, value, offsets in zip(datasets, values, partitions, strict=True)
                    ]
                source_gradient = np.stack(pieces, axis=0) if columns is not None else pieces[0]
                contributions.append((source, source_gradient, columns))

        for source, source_gradient, source_columns in contributions:
            yield from cls._gradient_contributions(source, source_gradient, source_columns)

    @classmethod
    def _read_gradient(
        cls,
        grad_path: str,
        columns: list[str] | None = None,
        *,
        primal_columns: list[str] | None = None,
        primal_shapes: dict[str, tuple[int, ...]] | None = None,
    ) -> np.ndarray:
        with h5py.File(grad_path, "r") as file:
            gradient = file["gradient"]
            if isinstance(gradient, h5py.Dataset):
                return cls._read_dataset(file, "gradient", columns=columns)

            if columns is None:
                columns = primal_columns if primal_columns is not None else cls._gradient_primal_columns(grad_path)
            if columns is None:
                columns = cls._stored_columns(gradient)
            columns = cls._validate_columns(columns)
            gradient_columns = cls._stored_columns(gradient)
            dtype = gradient[gradient_columns[0]].dtype
            template_shape = gradient[gradient_columns[0]].shape
            values = []
            for column in columns:
                if column in gradient:
                    value = gradient[column]
                    values.append(np.asarray(value[()] if value.shape == () else value[:]))
                else:
                    shape = primal_shapes[column] if primal_shapes is not None else template_shape
                    values.append(np.zeros(shape, dtype=dtype))
            return np.stack(values, axis=0)

    @classmethod
    def load_gradient(cls, grad_path: str, columns: list[str] | None = None) -> np.ndarray:
        return cls._read_gradient(grad_path, columns)

    @classmethod
    def load_backward_gradient(cls, fpath: str, grad_path: str) -> np.ndarray:
        """Load the resolved backward gradient for a primal, zero-filling
        named columns that the gradient file does not contain."""
        with h5py.File(fpath, "r") as primal_file:
            cls._validate_vds_sources(primal_file)
            primal = primal_file["primal"]
            if isinstance(primal, h5py.Dataset):
                return cls._read_gradient(grad_path)
            primal_columns = cls._stored_columns(primal)
            primal_shapes = {column: primal[column].shape for column in primal_columns}
            return cls._read_gradient(grad_path, primal_columns=primal_columns, primal_shapes=primal_shapes)

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
    def _write_datasets(cls, fpath: str, columns: list[str] | None = None, **datasets: np.ndarray) -> None:
        if columns is not None:
            cls._validate_columns(columns)
            for value in datasets.values():
                cls._validate_columns(columns, value)

        os.makedirs(os.path.dirname(fpath) or ".", exist_ok=True)
        with h5py.File(fpath, "w", libver="latest") as file:
            file.attrs[cls.STORE_ATTRIBUTE] = STORE_COLUMN if columns is not None else STORE_FLAT
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
        dataset = parent.create_dataset(name, data=value)
        statistics = cls._statistics(value)
        if statistics is not None:
            cls._write_statistics(dataset, statistics)

    @staticmethod
    def _prepare_row_store(datasets: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
        if not datasets:
            raise ValueError("Row-store storage requires at least one feature.")
        keys = list(datasets)
        if any(not isinstance(key, str) or not key or "\x00" in key for key in keys):
            raise ValueError("Feature keys must be non-empty strings without NUL characters.")
        arrays = []
        for value in datasets.values():
            array = np.asarray(value)
            if array.ndim > 1:
                raise ValueError("Row-store features must be scalars or one-dimensional arrays.")
            arrays.append(array.reshape(-1))
        row_count = arrays[0].shape[0]
        if any(value.shape[0] != row_count for value in arrays[1:]):
            raise ValueError("All row-store features must contain the same number of rows.")
        return keys, np.column_stack(arrays)

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
    def _vds_sources(cls, file: h5py.File) -> list[str]:
        manifest = file.attrs.get(cls.VDS_SOURCES_ATTRIBUTE)
        if manifest is None:
            return []
        file_dir = os.path.dirname(os.path.abspath(file.filename))
        sources = [
            os.path.normpath(os.path.join(file_dir, source))
            for source in json.loads(manifest)
        ]
        missing = [source for source in sources if not os.path.isfile(source)]
        if missing:
            raise FileNotFoundError(
                f"HDF5 virtual dataset '{file.filename}' is missing source '{missing[0]}'."
            )
        return sources

    @classmethod
    def _validate_vds_sources(cls, file: h5py.File) -> None:
        cls._vds_sources(file)

    @classmethod
    def aggregate_gradients(cls, sources: list[str], destination: str) -> None:
        columns = cls._union_gradient_columns(sources)
        total = None
        for path in sources:
            value = cls.load_gradient(path, columns=columns)
            total = value if total is None else total + value
        cls.write_gradient(destination, total, columns=columns)

    @classmethod
    def write_gradient(
        cls,
        fpath: str,
        gradient: np.ndarray,
        columns: list[str] | None = None,
        *,
        merge: bool = False,
    ) -> None:
        """Publish a gradient file atomically, or merge named columns in place."""
        if merge and columns is not None:
            selected = cls._validate_columns(columns, gradient)
            with h5py.File(fpath, "a", libver="latest") as file:
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

        temporary = f"{fpath}.{uuid.uuid4().hex}.partial"
        try:
            cls._write_datasets(temporary, columns=columns, gradient=gradient)
            os.replace(temporary, fpath)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def _union_gradient_columns(cls, sources: list[str]) -> list[str] | None:
        columns: list[str] = []
        for path in sources:
            with h5py.File(path, "r") as file:
                gradient = file["gradient"]
                if isinstance(gradient, h5py.Group):
                    columns.extend(cls._stored_columns(gradient))
        return list(dict.fromkeys(columns)) or None

    @classmethod
    def reduce(cls, sources: list[str], destination: str) -> None:
        if not sources:
            raise ValueError("HDF5 reduction requires at least one source file.")

        metadata, store_types = cls._collect_metadata(sources)
        dataset_names = set(metadata[0])
        if any(set(m) != dataset_names for m in metadata[1:]):
            raise ValueError("All HDF5 reduction sources must expose identical datasets.")
        if any(store != store_types[0] for store in store_types[1:]):
            raise ValueError("HDF5 reduction sources must share the same store type.")

        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
        destination_dir = os.path.dirname(os.path.abspath(destination))
        relative_sources = [
            os.path.relpath(os.path.abspath(source), destination_dir)
            for source in sources
        ]

        with h5py.File(destination, "w", libver="latest") as output:
            output.attrs[cls.VDS_SOURCES_ATTRIBUTE] = json.dumps(relative_sources)
            output.attrs[cls.STORE_ATTRIBUTE] = store_types[0]
            with h5py.File(sources[0], "r") as first_source:
                cls._copy_group_attributes(first_source, output)
            for dataset_name in sorted(dataset_names):
                shapes = [m[dataset_name][0] for m in metadata]
                dtype = metadata[0][dataset_name][1]
                statistics = cls._combine_statistics([m[dataset_name][2] for m in metadata])
                keys = metadata[0][dataset_name][3]
                if any(m[dataset_name][3] != keys for m in metadata[1:]):
                    raise ValueError(f"Cannot reduce HDF5 dataset '{dataset_name}' with incompatible keys.")
                cls._write_vds(output, dataset_name, relative_sources, shapes, dtype, statistics, keys)

    @classmethod
    def _copy_group_attributes(cls, source: h5py.Group, target: h5py.Group) -> None:
        for name, value in source.items():
            if not isinstance(value, h5py.Group):
                continue
            target_group = target.require_group(name)
            if cls.COLUMNS_ATTRIBUTE in value.attrs:
                target_group.attrs[cls.COLUMNS_ATTRIBUTE] = value.attrs[cls.COLUMNS_ATTRIBUTE]
            cls._copy_group_attributes(value, target_group)

    @classmethod
    def _collect_metadata(
        cls,
        sources: list[str],
        *,
        include_statistics: bool = True,
    ) -> tuple[list[dict[str, tuple[tuple[int, ...], np.dtype, tuple | None, list[str] | None]]], list[str]]:
        metadata = []
        store_types = []
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
                    source_metadata = {}
                    file.visititems(
                        lambda name, obj: source_metadata.update(
                            [(name, (obj.shape, obj.dtype, cls._read_statistics(obj) if include_statistics else None))]
                        )
                        if isinstance(obj, h5py.Dataset) else None
                    )
                    for name, item in list(source_metadata.items()):
                        dataset = file[name]
                        stored_keys = dataset.attrs.get(cls.ROW_STORE_KEYS_ATTRIBUTE)
                        source_metadata[name] = (*item, json.loads(stored_keys) if stored_keys is not None else None)
                    if not source_metadata:
                        raise ValueError(f"HDF5 source '{source}' contains no datasets.")
                    store_type = file.attrs.get(cls.STORE_ATTRIBUTE)
                    if store_type not in (STORE_FLAT, STORE_COLUMN, STORE_ROW):
                        raise ValueError(f"HDF5 source '{source}' has an unknown or missing store-type tag.")
                    metadata.append(source_metadata)
                    store_types.append(store_type)
            except OSError:
                raise TypeError(
                    f"HDF5 reduction source '{source}' is not a valid HDF5 file. "
                    f"The file may be a different format (e.g. PyTorch .pt, JSON, etc.)."
                )
        return metadata, store_types

    @classmethod
    def _write_vds(
        cls,
        output: h5py.File,
        dataset_name: str,
        relative_sources: list[str],
        shapes: list[tuple[int, ...]],
        dtype: np.dtype,
        statistics: tuple | None,
        keys: list[str] | None,
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
        dataset = output.create_virtual_dataset(dataset_name, layout)
        dataset.attrs[cls.PARTITION_SIZES_ATTRIBUTE] = json.dumps(
            [1 if shape == () else shape[0] for shape in shapes]
        )
        if statistics is not None:
            cls._write_statistics(dataset, statistics)
        if keys is not None:
            dataset.attrs[cls.ROW_STORE_KEYS_ATTRIBUTE] = json.dumps(keys)

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
        with h5py.File(path, "r") as file:
            primal = file["primal"]
            if isinstance(primal, h5py.Dataset):
                return tuple(primal.shape)
            columns = cls._stored_columns(primal)
            if not columns:
                raise ValueError(f"HDF5 dataset '{path}' contains no columns under 'primal'.")
            template = primal[columns[0]]
            num_samples = 1 if template.shape == () else template.shape[0]
            return num_samples, len(columns)


class HDF5File(AutogradFileInterface, HDF5Writer):
    """DSBO file channel for HDF5: the autograd layer backed by :class:`HDF5Writer`."""
    writer = HDF5Writer
