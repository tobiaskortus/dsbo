import json
import os
import re

import numpy as np
import torch
import uproot

from dsbo.utils.io.storage.autograd import AutogradFileInterface
from dsbo.utils.io.datasets.views import (
    STORE_COLUMN,
    STORE_FLAT,
    STORE_ROW,
    DatasetView,
    ROOTColumnStoreDatasetView,
    ROOTDatasetView,
    ROOTRowStoreDatasetView,
)


class ROOTFile(AutogradFileInterface):
    """
    Store autograd arrays in ROOT TTrees using Uproot.

    Reduced files contain a manifest rather than materialized trees. Uproot cannot create or resolve a persisted
    TChain, so reads use its multi-file concatenation API to provide equivalent virtual behavior.

    Resulting file layout:

    data.root
    ├── primal             TTree
    ├── tangent            TTree, optional
    └── dsbo_metadata      TObjString containing JSON

    data_gradient.root
    ├── gradient           TTree
    └── dsbo_metadata

    A virtually reduced file contains no data TTrees:
    reduced.root
    ├── dsbo_metadata
    └── dsbo_vds_sources   TObjString containing relative paths
    """

    METADATA_OBJECT = "dsbo_metadata"
    VDS_SOURCES_OBJECT = "dsbo_vds_sources"
    FORMAT_VERSION = 2
    VALUE_BRANCH = "values"
    COLUMN_BRANCH_PREFIX = "column_"
    ROW_STORE_DATASET = "__dsbo_row_store__"

    @classmethod
    def reducible_extensions(cls) -> frozenset[str]:
        return frozenset({".root"})

    @classmethod
    def open_primal(cls, fpath: str, columns: list[str] | None = None) -> np.ndarray:
        return cls._read_dataset(fpath, "primal", columns=columns)

    @classmethod
    def open_primal_tangent(cls, fpath: str, columns: list[str] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        metadata, _ = cls._load_container(fpath)
        if "primal" not in metadata["datasets"]:
            raise ValueError("Saved ROOT file must contain a 'primal' TTree.")
        primal = cls._read_dataset(fpath, "primal", columns=columns)
        tangent = cls._read_dataset(fpath, "tangent", columns=columns) if "tangent" in metadata["datasets"] else None
        return primal, tangent

    @classmethod
    def write_primal(cls, fpath: str, primal: np.ndarray, columns: list[str] | None = None):
        cls._write_datasets(fpath, columns=columns, primal=primal)

    @classmethod
    def write_primal_tangent(cls, fpath: str, primal: np.ndarray, tangent: np.ndarray, columns: list[str] | None = None):
        if np.shape(primal) != np.shape(tangent):
            raise ValueError("ROOT primal and tangent arrays must have identical shapes.")
        cls._write_datasets(fpath, columns=columns, primal=primal, tangent=tangent)

    @classmethod
    def write_named_datasets(cls, fpath: str, datasets: dict[str, np.ndarray]) -> None:
        if not datasets:
            return
        stored: dict[str, np.ndarray] = {}
        if os.path.isfile(fpath):
            metadata, sources = cls._load_container(fpath)
            if sources is not None:
                raise ValueError("Cannot append named datasets to a reduced ROOT file.")
            stored = {
                name: cls._read_dataset(fpath, name)
                for name in metadata["datasets"]
            }
        stored.update(datasets)
        cls._write_datasets(fpath, **stored)

    @classmethod
    def write_row_store(cls, fpath: str, datasets: dict[str, np.ndarray]) -> None:
        keys, values = cls._prepare_row_store(datasets)
        cls._write_datasets(
            fpath,
            columns=keys,
            **{cls.ROW_STORE_DATASET: values.T},
        )

    @classmethod
    def open_dataset(cls, fpath: str) -> DatasetView:
        metadata, _ = cls._load_container(fpath)
        store = metadata.get("store")
        if store == STORE_ROW:
            return ROOTRowStoreDatasetView(fpath, cls.ROW_STORE_DATASET)
        if store == STORE_COLUMN:
            return ROOTColumnStoreDatasetView(fpath)
        if store == STORE_FLAT:
            return ROOTDatasetView(fpath)
        raise ValueError(f"ROOT file '{fpath}' has a missing or unknown 'store' tag.")

    @classmethod
    def columns(cls, fpath: str) -> list[str]:
        metadata, _ = cls._load_container(fpath)
        primal = metadata["datasets"]["primal"]
        return [] if primal["columns"] is None else list(primal["columns"])

    @classmethod
    def _read_dataset(cls, fpath: str, name: str, columns: list[str] | None = None) -> np.ndarray:
        metadata, sources = cls._load_container(fpath)
        if sources is not None:
            cls._validate_vds_sources(fpath, sources)
        if name not in metadata["datasets"]:
            raise ValueError(f"Saved ROOT file has no '{name}' TTree.")

        dataset = metadata["datasets"][name]
        shape = tuple(dataset["shape"])
        dtype = np.dtype(dataset["dtype"])
        stored_columns = dataset["columns"]
        branches = dataset["branches"]

        if stored_columns is None:
            if columns is not None:
                raise ValueError(f"ROOT TTree '{name}' was not written with named columns.")
            values = cls._read_flat_branches(fpath, name, branches)[branches[0]]
            return cls._reshape(values, shape, dtype, name)

        selected = cls._validate_columns(columns) if columns is not None else list(stored_columns)
        missing = [column for column in selected if column not in stored_columns]
        if missing:
            raise KeyError(f"ROOT TTree '{name}' has no column '{missing[0]}'.")
        branch_by_column = dict(zip(stored_columns, branches, strict=True))
        selected_branches = [branch_by_column[column] for column in selected]
        flat_values = cls._read_flat_branches(fpath, name, selected_branches)
        column_shape = shape[1:]
        values = [
            cls._reshape(flat_values[branch_by_column[column]], column_shape, dtype, f"{name}/{column}")
            for column in selected
        ]
        return np.stack(values, axis=0)

    @classmethod
    def _read_flat_branches(cls, fpath: str, name: str, branches: list[str]) -> dict[str, np.ndarray]:
        _, sources = cls._load_container(fpath)
        if sources is None:
            with uproot.open(fpath) as file:
                if name not in file:
                    raise ValueError(f"Saved ROOT file has no '{name}' TTree.")
                tree = file[name]
                missing = [branch for branch in branches if branch not in tree]
                if missing:
                    raise KeyError(f"ROOT TTree '{name}' has no branch '{missing[0]}'.")
                return {branch: tree[branch].array(library="np") for branch in branches}

        leaf_sources = cls._leaf_sources(fpath)
        specifications = [{source: name} for source in leaf_sources]
        values = uproot.concatenate(specifications, expressions=branches, library="np", how=dict)
        if isinstance(values, dict):
            return values
        if len(values) == 0:
            return {branch: np.empty(0) for branch in branches}
        raise TypeError(f"Uproot returned an unexpected result while reading virtual ROOT TTree '{name}'.")

    @classmethod
    def _reshape(cls, value: np.ndarray, shape: tuple[int, ...], dtype: np.dtype, name: str) -> np.ndarray:
        value = np.asarray(value, dtype=dtype)
        expected_size = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if value.size != expected_size:
            raise ValueError(f"ROOT dataset '{name}' has {value.size} values, but its metadata declares shape {shape}.")
        return value.reshape(shape)

    @classmethod
    def _load_gradient(cls, grad_path: str, columns: list[str] | None = None) -> np.ndarray:
        primal_metadata = cls._gradient_primal_metadata(grad_path)
        primal_columns = primal_metadata["columns"]
        if primal_columns is None:
            return cls._read_dataset(grad_path, "gradient", columns=columns)

        selected = cls._validate_columns(columns) if columns is not None else list(primal_columns)
        missing = [column for column in selected if column not in primal_columns]
        if missing:
            raise KeyError(f"ROOT primal TTree has no column '{missing[0]}'.")

        gradient_metadata, _ = cls._load_container(grad_path)
        stored_columns = gradient_metadata["datasets"]["gradient"]["columns"]
        if stored_columns is None:
            raise ValueError("Cannot load named gradients from an unnamed ROOT gradient TTree.")
        present = [column for column in selected if column in stored_columns]
        present_values = cls._read_dataset(grad_path, "gradient", columns=present) if present else None
        values_by_column = dict(zip(present, present_values, strict=True)) if present_values is not None else {}
        dtype = np.dtype(gradient_metadata["datasets"]["gradient"]["dtype"])
        column_shape = tuple(primal_metadata["shape"])[1:]
        values = [values_by_column.get(column, np.zeros(column_shape, dtype=dtype)) for column in selected]
        return np.stack(values, axis=0)

    @classmethod
    def _save_gradient(cls, grad_path: str, gradient: np.ndarray, columns: list[str] | None = None):
        primal_metadata = cls._gradient_primal_metadata(grad_path)
        primal_columns = primal_metadata["columns"]
        if primal_columns is None:
            cls._write_datasets(grad_path, gradient=gradient)
            return

        selected = list(primal_columns) if columns is None else cls._validate_columns(columns, gradient)
        missing = [column for column in selected if column not in primal_columns]
        if missing:
            raise KeyError(f"ROOT primal TTree has no column '{missing[0]}'.")

        if grad_path not in cls._gradient_written or not os.path.isfile(grad_path):
            cls._write_datasets(grad_path, columns=selected, gradient=gradient)
            return

        metadata, _ = cls._load_container(grad_path)
        stored_columns = metadata["datasets"]["gradient"]["columns"]
        if stored_columns is None:
            raise ValueError("Cannot accumulate named gradients into an unnamed ROOT gradient TTree.")
        stored_values = cls._read_dataset(grad_path, "gradient")
        values = dict(zip(stored_columns, stored_values, strict=True))
        values.update(dict(zip(selected, gradient, strict=True)))
        merged_columns = list(stored_columns) + [column for column in selected if column not in stored_columns]
        merged_gradient = np.stack([values[column] for column in merged_columns], axis=0)
        cls._write_datasets(grad_path, columns=merged_columns, gradient=merged_gradient)

    @classmethod
    def _gradient_primal_metadata(cls, grad_path: str) -> dict:
        base, extension = os.path.splitext(grad_path)
        suffix = "_gradient"
        if not base.endswith(suffix):
            raise ValueError(f"ROOT gradient path '{grad_path}' does not end in '{suffix}{extension}'.")
        primal_path = f"{base[:-len(suffix)]}{extension}"
        metadata, _ = cls._load_container(primal_path)
        if "primal" not in metadata["datasets"]:
            raise ValueError("Saved ROOT file must contain a 'primal' TTree.")
        return metadata["datasets"]["primal"]

    @classmethod
    def read_backward_gradient(cls, fpath: str) -> torch.Tensor:
        metadata, _ = cls._load_container(fpath)
        if "primal" not in metadata["datasets"]:
            raise ValueError("Saved ROOT file must contain a 'primal' TTree.")
        primal_metadata = metadata["datasets"]["primal"]
        gradient = cls._load_gradient(cls._gradient_path(fpath), columns=primal_metadata["columns"])
        return torch.from_numpy(gradient).float()

    @classmethod
    def _write_datasets(
        cls,
        fpath: str,
        columns: list[str] | None = None,
        row_store: list[str] | None = None,
        **datasets: np.ndarray,
    ) -> None:
        arrays = {name: cls._prepare_array(value) for name, value in datasets.items()}
        if (
            row_store is None
            and columns is not None
            and set(arrays) == {cls.ROW_STORE_DATASET}
        ):
            row_store = columns
            columns = None
            arrays[cls.ROW_STORE_DATASET] = np.ascontiguousarray(
                arrays[cls.ROW_STORE_DATASET].T
            )
        if row_store is not None:
            cls._validate_columns(row_store)
            if set(arrays) != {cls.ROW_STORE_DATASET}:
                raise ValueError("Row-store metadata can only describe the row-store dataset.")
        if columns is not None:
            cls._validate_columns(columns)
            for value in arrays.values():
                cls._validate_columns(columns, value)

        os.makedirs(os.path.dirname(fpath) or ".", exist_ok=True)
        store_type = STORE_ROW if row_store is not None else (STORE_COLUMN if columns is not None else STORE_FLAT)
        metadata = {"version": cls.FORMAT_VERSION, "store": store_type, "datasets": {}}
        with uproot.recreate(fpath, compression=None) as file:
            for name, value in arrays.items():
                dataset_metadata, branch_values, branch_types = cls._dataset_for_write(value, columns)
                if row_store is not None:
                    dataset_metadata["keys"] = list(row_store)
                    statistics = {
                        key: cls._statistics(value[:, index])
                        for index, key in enumerate(row_store)
                    }
                    if all(item is not None for item in statistics.values()):
                        dataset_metadata["statistics"] = statistics
                tree = file.mktree(name, branch_types, title=f"DSBO {name}")
                cls._extend_in_baskets(tree, branch_values)
                metadata["datasets"][name] = dataset_metadata
            file[cls.METADATA_OBJECT] = json.dumps(metadata, separators=(",", ":"))

    @classmethod
    def _extend_in_baskets(cls, tree, branch_values: dict[str, np.ndarray]) -> None:
        first = next(iter(branch_values.values()))
        if not first.size:
            return
        num_entries = first.shape[0]
        bytes_per_entry = max(
            1,
            sum(value.nbytes for value in branch_values.values()) // num_entries,
        )
        entries_per_basket = max(1, cls._basket_target_bytes() // bytes_per_entry)
        for start in range(0, num_entries, entries_per_basket):
            stop = min(start + entries_per_basket, num_entries)
            tree.extend({branch: value[start:stop] for branch, value in branch_values.items()})

    @staticmethod
    def _basket_target_bytes() -> int:
        return 1 << 20

    @classmethod
    def _prepare_array(cls, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value)
        dtype = value.dtype
        supported = (
            dtype.kind == "b"
            or dtype.kind in "iu" and dtype.itemsize in (1, 2, 4, 8)
            or dtype.kind == "f" and dtype.itemsize in (4, 8)
        )
        if not supported:
            raise TypeError(f"ROOT storage does not support NumPy dtype '{dtype}'.")
        native_dtype = dtype.newbyteorder("=")
        if value.ndim == 0:
            return np.array(value, dtype=native_dtype, copy=True)
        return np.ascontiguousarray(value, dtype=native_dtype)

    @classmethod
    def _dataset_for_write(
        cls, value: np.ndarray, columns: list[str] | None
    ) -> tuple[dict, dict[str, np.ndarray], dict[str, np.dtype]]:
        if columns is None:
            branches = [cls.VALUE_BRANCH]
            branch_value, branch_type = cls._branch_for_write(value)
            branch_values = {cls.VALUE_BRANCH: branch_value}
            branch_types = {cls.VALUE_BRANCH: branch_type}
        else:
            branches = [cls._column_branch(column) for column in columns]
            prepared = [cls._branch_for_write(column_value) for column_value in value]
            branch_values = {
                branch: branch_value
                for branch, (branch_value, _) in zip(branches, prepared, strict=True)
            }
            branch_types = {
                branch: branch_type
                for branch, (_, branch_type) in zip(branches, prepared, strict=True)
            }
        metadata = {
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "columns": list(columns) if columns is not None else None,
            "branches": branches,
        }
        if columns is None:
            statistics = cls._statistics(value)
            if statistics is not None:
                metadata["statistics"] = statistics
        else:
            statistics = {
                column: cls._statistics(column_value)
                for column, column_value in zip(columns, value, strict=True)
            }
            if all(value is not None for value in statistics.values()):
                metadata["statistics"] = statistics
        return metadata, branch_values, branch_types

    @staticmethod
    def _prepare_row_store(datasets: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
        if not datasets:
            raise ValueError("Row-store storage requires at least one feature.")
        keys = list(datasets)
        ROOTFile._validate_columns(keys)
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

    @staticmethod
    def _statistics(value: np.ndarray) -> dict | None:
        if value.ndim != 1 or value.size == 0 or value.dtype.kind not in "biuf":
            return None
        values = value.astype(np.float64, copy=False)
        mean = float(values.mean())
        return {
            "count": value.size,
            "mean": mean,
            "squared_deviation": float(np.square(values - mean).sum()),
        }

    @staticmethod
    def _combine_statistics(values: list[dict | None]) -> dict | None:
        if not values or any(value is None for value in values):
            return None
        count = values[0]["count"]
        mean = values[0]["mean"]
        squared_deviation = values[0]["squared_deviation"]
        for value in values[1:]:
            batch_count = value["count"]
            total = count + batch_count
            delta = value["mean"] - mean
            mean += delta * batch_count / total
            squared_deviation += (
                value["squared_deviation"]
                + delta * delta * count * batch_count / total
            )
            count = total
        return {
            "count": count,
            "mean": mean,
            "squared_deviation": squared_deviation,
        }

    @classmethod
    def _branch_for_write(cls, value: np.ndarray) -> tuple[np.ndarray, np.dtype]:
        if value.ndim == 0:
            return value.reshape(1), value.dtype
        trailing_shape = value.shape[1:]
        if any(size == 0 for size in trailing_shape):
            return value.reshape(-1), value.dtype
        branch_type = np.dtype((value.dtype, trailing_shape)) if trailing_shape else value.dtype
        return value, branch_type

    @classmethod
    def _column_branch(cls, column: str) -> str:
        if (
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", column)
            and column != cls.VALUE_BRANCH
            and not column.startswith(cls.COLUMN_BRANCH_PREFIX)
        ):
            return column
        return f"{cls.COLUMN_BRANCH_PREFIX}{column.encode('utf-8').hex()}"

    @classmethod
    def _validate_columns(cls, columns: list[str], value: np.ndarray | None = None) -> list[str]:
        if any(not isinstance(column, str) or not column or "\x00" in column for column in columns):
            raise ValueError("ROOT column names must be non-empty strings without NUL characters.")
        if len(set(columns)) != len(columns):
            raise ValueError("Column names must be unique.")
        if value is not None and (value.ndim == 0 or value.shape[0] != len(columns)):
            size = 0 if value.ndim == 0 else value.shape[0]
            raise ValueError(f"Received {len(columns)} columns for an array with size {size} along dimension 0.")
        return columns

    @classmethod
    def _load_container(cls, fpath: str) -> tuple[dict, list[str] | None]:
        with uproot.open(fpath) as file:
            if cls.METADATA_OBJECT not in file:
                raise ValueError(f"ROOT file '{fpath}' has no DSBO metadata.")
            try:
                metadata = json.loads(str(file[cls.METADATA_OBJECT]))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"ROOT file '{fpath}' has invalid DSBO metadata.") from exc
            if metadata.get("version") != cls.FORMAT_VERSION or not isinstance(metadata.get("datasets"), dict):
                raise ValueError(f"ROOT file '{fpath}' has unsupported DSBO metadata.")
            if cls.VDS_SOURCES_OBJECT not in file:
                return metadata, None
            try:
                sources = json.loads(str(file[cls.VDS_SOURCES_OBJECT]))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"ROOT file '{fpath}' has an invalid virtual source manifest.") from exc
            if not isinstance(sources, list) or not all(isinstance(source, str) for source in sources):
                raise ValueError(f"ROOT file '{fpath}' has an invalid virtual source manifest.")
            return metadata, sources

    @classmethod
    def _resolved_sources(cls, fpath: str, sources: list[str]) -> list[str]:
        file_dir = os.path.dirname(os.path.abspath(fpath))
        return [os.path.normpath(os.path.join(file_dir, source)) for source in sources]

    @classmethod
    def _validate_vds_sources(cls, fpath: str, sources: list[str]) -> None:
        resolved = cls._resolved_sources(fpath, sources)
        missing = [source for source in resolved if not os.path.isfile(source)]
        if missing:
            raise FileNotFoundError(f"ROOT virtual dataset '{fpath}' is missing source '{missing[0]}'.")

    @classmethod
    def _leaf_sources(cls, fpath: str, seen: set[str] | None = None, dependencies: set[str] | None = None) -> list[str]:
        absolute_path = os.path.abspath(fpath)
        seen = set() if seen is None else seen
        if absolute_path in seen:
            raise ValueError(f"ROOT virtual dataset dependency cycle includes '{absolute_path}'.")
        seen.add(absolute_path)
        metadata, sources = cls._load_container(absolute_path)
        if dependencies is not None:
            file_dir = os.path.dirname(absolute_path)
            dependencies.update(
                os.path.normpath(os.path.join(file_dir, dependency))
                for dependency in metadata.get("dependencies", [])
            )
        if sources is None:
            seen.remove(absolute_path)
            return [absolute_path]
        cls._validate_vds_sources(absolute_path, sources)
        leaves = []
        for source in cls._resolved_sources(absolute_path, sources):
            if dependencies is not None:
                dependencies.add(source)
            leaves.extend(cls._leaf_sources(source, seen, dependencies))
        seen.remove(absolute_path)
        return leaves

    @classmethod
    def dependencies(cls, fpath: str) -> list[str]:
        _, sources = cls._load_container(fpath)
        return [] if sources is None else cls._resolved_sources(fpath, sources)

    @classmethod
    def validate(cls, fpath: str) -> None:
        cls._leaf_sources(fpath)

    @classmethod
    def reduce(cls, sources: list[str], destination: str) -> None:
        if not sources:
            raise ValueError("ROOT reduction requires at least one source file.")

        metadata, leaf_sources = cls._collect_metadata(sources)
        stores = [item.get("store") for item in metadata]
        if any(store not in (STORE_FLAT, STORE_COLUMN, STORE_ROW) for store in stores):
            raise ValueError("ROOT reduction sources must expose a valid 'store' tag.")
        if any(store != stores[0] for store in stores[1:]):
            raise ValueError("ROOT reduction sources must share the same store type.")
        destination_path = os.path.realpath(destination)
        dependency_paths = {os.path.realpath(source) for source in sources}
        for source in sources:
            source_dependencies: set[str] = set()
            cls._leaf_sources(source, dependencies=source_dependencies)
            dependency_paths.update(os.path.realpath(path) for path in source_dependencies)
        same_file = os.path.exists(destination) and any(
            os.path.exists(path) and os.path.samefile(destination, path)
            for path in dependency_paths
        )
        if destination_path in dependency_paths or same_file:
            raise ValueError("ROOT reduction destination must not be a source or transitive dependency.")

        dataset_names = set(metadata[0]["datasets"])
        if any(set(item["datasets"]) != dataset_names for item in metadata[1:]):
            raise ValueError("All ROOT reduction sources must expose identical datasets.")

        reduced_datasets = {
            name: cls._reduce_dataset_metadata(name, [item["datasets"][name] for item in metadata])
            for name in sorted(dataset_names)
        }
        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
        destination_dir = os.path.dirname(os.path.abspath(destination))
        destination_metadata = {
            "version": cls.FORMAT_VERSION,
            "store": stores[0],
            "datasets": reduced_datasets,
            "dependencies": [
                os.path.relpath(path, destination_dir)
                for path in sorted(dependency_paths)
            ],
        }
        relative_sources = [
            os.path.relpath(os.path.abspath(source), destination_dir)
            for source in leaf_sources
        ]
        with uproot.recreate(destination, compression=None) as file:
            file[cls.METADATA_OBJECT] = json.dumps(destination_metadata, separators=(",", ":"))
            file[cls.VDS_SOURCES_OBJECT] = json.dumps(relative_sources, separators=(",", ":"))

    @classmethod
    def _collect_metadata(cls, sources: list[str]) -> tuple[list[dict], list[str]]:
        metadata = []
        resolved_leaves = []
        for source in sources:
            if not os.path.isfile(source):
                raise FileNotFoundError(f"Missing registered batch input: {source}")
            extension = os.path.splitext(source)[1].lower()
            if extension not in cls.reducible_extensions():
                raise TypeError(
                    f"ROOT reduction does not support '{source}'. "
                    f"Only {', '.join(sorted(cls.reducible_extensions()))} files can be reduced."
                )
            try:
                source_metadata, _ = cls._load_container(source)
            except OSError as exc:
                raise TypeError(f"ROOT reduction source '{source}' is not a valid ROOT file.") from exc
            leaf_sources = cls._leaf_sources(source)
            for leaf_source in leaf_sources:
                leaf_metadata, leaf_manifest = cls._load_container(leaf_source)
                if leaf_manifest is not None:
                    raise ValueError(f"ROOT virtual dataset '{source}' did not resolve to physical leaves.")
                metadata.append(leaf_metadata)
                resolved_leaves.append(leaf_source)
        return metadata, resolved_leaves

    @classmethod
    def _reduce_dataset_metadata(cls, name: str, datasets: list[dict]) -> dict:
        first = datasets[0]
        if any(dataset["dtype"] != first["dtype"] for dataset in datasets[1:]):
            raise ValueError(f"Cannot reduce ROOT dataset '{name}' with incompatible dtypes.")
        first_columns = first["columns"]
        if any(
            (dataset["columns"] is None) != (first_columns is None)
            or dataset["columns"] is not None and set(dataset["columns"]) != set(first_columns)
            for dataset in datasets[1:]
        ):
            raise ValueError(f"Cannot reduce ROOT dataset '{name}' with incompatible columns.")
        if any(
            dataset["branches"] != [cls._column_branch(column) for column in dataset["columns"]]
            if dataset["columns"] is not None
            else dataset["branches"] != [cls.VALUE_BRANCH]
            for dataset in datasets
        ):
            raise ValueError(f"Cannot reduce ROOT dataset '{name}' with incompatible branches.")
        first_keys = first.get("keys")
        if any(dataset.get("keys") != first_keys for dataset in datasets[1:]):
            raise ValueError(f"Cannot reduce ROOT dataset '{name}' with incompatible keyed rows.")

        shapes = [tuple(dataset["shape"]) for dataset in datasets]
        columns = first_columns
        if columns is None:
            reduced_shape = cls._concatenated_shape(name, shapes)
            partitions = [1 if shape == () else shape[0] for shape in shapes]
        else:
            if any(not shape or shape[0] != len(columns) for shape in shapes):
                raise ValueError(f"Cannot reduce ROOT dataset '{name}' with incompatible shapes: {shapes}.")
            column_shape = cls._concatenated_shape(name, [shape[1:] for shape in shapes])
            reduced_shape = (len(columns), *column_shape)
            partitions = [1 if len(shape) == 1 else shape[1] for shape in shapes]
        result = {
            "shape": list(reduced_shape),
            "dtype": first["dtype"],
            "columns": columns,
            "branches": first["branches"],
            "partitions": partitions,
        }
        if first_keys is not None:
            result["keys"] = first_keys
            statistics = {
                key: cls._combine_statistics(
                    [dataset.get("statistics", {}).get(key) for dataset in datasets]
                )
                for key in first_keys
            }
            if all(value is not None for value in statistics.values()):
                result["statistics"] = statistics
        elif columns is None:
            statistics = cls._combine_statistics([dataset.get("statistics") for dataset in datasets])
            if statistics is not None:
                result["statistics"] = statistics
        else:
            statistics = {
                column: cls._combine_statistics(
                    [dataset.get("statistics", {}).get(column) for dataset in datasets]
                )
                for column in columns
            }
            if all(value is not None for value in statistics.values()):
                result["statistics"] = statistics
        return result

    @classmethod
    def _concatenated_shape(cls, name: str, shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
        if shapes[0] == ():
            if any(shape != () for shape in shapes[1:]):
                raise ValueError(f"Cannot reduce ROOT dataset '{name}' with incompatible shapes: {shapes}.")
            return (len(shapes),)
        if any(shape == () or shape[1:] != shapes[0][1:] for shape in shapes[1:]):
            raise ValueError(f"Cannot reduce ROOT dataset '{name}' with incompatible shapes: {shapes}.")
        return sum(shape[0] for shape in shapes), *shapes[0][1:]

    @classmethod
    def shape_from_path(cls, path: str) -> tuple[int, ...]:
        metadata, _ = cls._load_container(path)
        primal = metadata["datasets"]["primal"]
        shape = tuple(primal["shape"])
        columns = primal["columns"]
        if columns is None:
            return shape
        num_samples = 1 if len(shape) == 1 else shape[1]
        return num_samples, len(columns)
