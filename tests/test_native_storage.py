"""Format compatibility is checked independently with h5py, never a fallback."""

import gc
import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import distribution
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from dsbo.utils.io.storage import HDF5Writer as W
from dsbo.utils.io.storage import _hdf5


DTYPES = ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64", "complex64", "complex128", ">i4", ">f8", ">c16"]
INCLUDE_DIRECTORY = Path(__file__).resolve().parents[1] / "cpp/include"
LICENSE_DIRECTORY = Path(__file__).resolve().parents[1] / "LICENSES"


# Cover every dtype on a matrix, then exercise other shapes with representative
# half, real, and complex types rather than repeating the full Cartesian product.
@pytest.mark.parametrize(
    "dtype,shape",
    [(dtype, (2, 3)) for dtype in DTYPES] + [
        (dtype, shape)
        for dtype in ["float16", "float64", "complex128"]
        for shape in [(), (4,), (2, 1, 3), (0, 2), (2, 0)]
    ],
)
def test_numeric_roundtrip_and_independent_file_reader(tmp_path, dtype, shape):
    values = np.arange(np.prod(shape, dtype=int)).reshape(shape).astype(dtype)
    if np.dtype(dtype).kind == "c":
        values += (values * 1j).astype(dtype)
    path = tmp_path / "numeric.h5"
    W.write_primal(path, values)
    loaded = W.open_primal(path)
    assert loaded.shape == values.shape
    assert loaded.dtype == values.dtype
    np.testing.assert_array_equal(loaded, values)
    with h5py.File(path) as file:
        assert file.attrs[W.STORE_ATTRIBUTE] == "flat"
        assert file["primal"].dtype == values.dtype
        np.testing.assert_array_equal(file["primal"][()], values)
        should_have_stats = values.ndim in (1, 2) and values.shape[0] > 0 and values.dtype.kind in "biuf"
        assert (W.STATS_COUNT_ATTRIBUTE in file["primal"].attrs) == should_have_stats
        if should_have_stats:
            attributes = file["primal"].attrs
            mean = values.astype(np.float64).mean(axis=0)
            np.testing.assert_allclose(attributes[W.STATS_MEAN_ATTRIBUTE], mean)
            np.testing.assert_allclose(attributes[W.STATS_SQUARED_DEVIATION_ATTRIBUTE], ((values.astype(np.float64) - mean) ** 2).sum(axis=0))


@pytest.mark.parametrize("dtype", DTYPES)
def test_h5py_legacy_files_read_by_native_backend(tmp_path, dtype):
    values = np.arange(6).reshape(2, 3).astype(dtype)
    path = tmp_path / "legacy.h5"
    # These untagged physical files follow the original writer's ordinary
    # dataset representation; primal reads do not require a DSBO store tag.
    with h5py.File(path, "w", libver="latest") as file:
        file.create_dataset("primal", data=values)
    np.testing.assert_array_equal(W.open_primal(path), values)
    with pytest.raises(ValueError, match="store-type tag"):
        W.open_dataset(path)


@pytest.mark.parametrize("named", [False, True], ids=["flat", "column"])
@pytest.mark.parametrize("operation", ["shape", "read", "integer", "slice", "empty_slice", "indices", "empty_indices", "open_primal", "open_primal_tangent", "shape_from_path"])
def test_null_dataset_reads_and_shapes_rejected(tmp_path, named, operation):
    path = tmp_path / "null.h5"
    name = "primal/value" if named else "primal"
    with h5py.File(path, "w") as file:
        file.create_dataset(name, data=h5py.Empty("f8"))
        assert file[name].shape is None

    reader = _hdf5.Reader(str(path))
    try:
        operations = {
            "shape": lambda: reader.shape(name),
            "read": lambda: reader.read(name),
            "integer": lambda: reader.read(name, 0),
            "slice": lambda: reader.read(name, slice(None)),
            "empty_slice": lambda: reader.read(name, slice(0, 0)),
            "indices": lambda: reader.read_indices(name, [0]),
            "empty_indices": lambda: reader.read_indices(name, []),
            "open_primal": lambda: W.open_primal(path),
            "open_primal_tangent": lambda: W.open_primal_tangent(path),
            "shape_from_path": lambda: W.shape_from_path(path),
        }
        with pytest.raises(ValueError, match="H5S_NULL"):
            operations[operation]()
    finally:
        reader.close()


@pytest.mark.parametrize("named", [False, True], ids=["flat", "column"])
def test_null_tangent_rejected(tmp_path, named):
    path = tmp_path / "null-tangent.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("primal/value" if named else "primal", data=3.5)
        file.create_dataset("tangent/value" if named else "tangent", data=h5py.Empty("f8"))
    with pytest.raises(ValueError, match="H5S_NULL"):
        W.open_primal_tangent(path)


@pytest.mark.parametrize("attribute", [W.STATS_COUNT_ATTRIBUTE, W.STATS_MEAN_ATTRIBUTE, W.STATS_SQUARED_DEVIATION_ATTRIBUTE])
def test_null_numeric_metadata_rejected(tmp_path, attribute):
    path = tmp_path / "null-statistics.h5"
    W.write_primal(path, np.arange(3.))
    with h5py.File(path, "r+") as file:
        del file["primal"].attrs[attribute]
        file["primal"].attrs.create(attribute, data=h5py.Empty("u8" if attribute == W.STATS_COUNT_ATTRIBUTE else "f8"))
    reader = _hdf5.Reader(str(path))
    try:
        with pytest.raises(ValueError, match="H5S_NULL"):
            reader.statistics("primal")
    finally:
        reader.close()


@pytest.mark.parametrize("attribute", [W.STATS_MEAN_ATTRIBUTE, W.STATS_SQUARED_DEVIATION_ATTRIBUTE])
@pytest.mark.parametrize("shape", [(0,), (1,), (3,), (1, 2)], ids=["empty", "short", "long", "same-size-wrong-rank"])
@pytest.mark.parametrize("source_index", [0, 1], ids=["first-source", "second-source"])
def test_statistics_shape_mismatch_rejected_before_reduction(tmp_path, attribute, shape, source_index):
    sources = [tmp_path / "a.h5", tmp_path / "b.h5"]
    for index, source in enumerate(sources):
        W.write_primal(source, np.arange(6.).reshape(3, 2) + index)
    with h5py.File(sources[source_index], "r+") as file:
        del file["primal"].attrs[attribute]
        file["primal"].attrs.create(attribute, data=np.zeros(shape))
    destination = tmp_path / "destination.h5"
    W.write_primal(destination, np.array([42.]))
    before = destination.read_bytes()

    # A regression can index past a native array, so keep it out of pytest's process.
    result = subprocess.run([
        sys.executable, "-c", '''
import sys
import pytest
from dsbo.utils.io.storage import HDF5Writer as W, _hdf5

sources, destination, corrupted = sys.argv[1:3], sys.argv[3], sys.argv[4]
for inputs in (sources, [corrupted]):
    with pytest.raises(ValueError, match="statistics.*shapes"):
        W.reduce(inputs, destination)
reader = _hdf5.Reader(corrupted)
try:
    with pytest.raises(ValueError, match="statistics.*shapes"):
        reader.statistics("primal")
finally:
    reader.close()
''', *map(str, sources), str(destination), str(sources[source_index]),
    ], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"Child exited with {result.returncode}:\n{result.stdout}\n{result.stderr}"
    assert destination.read_bytes() == before


@pytest.mark.parametrize("dtype", [h5py.string_dtype("utf-8"), np.dtype("S16")], ids=["variable", "fixed"])
@pytest.mark.parametrize("shape", [None, (0,), (2,), (2, 2)], ids=["null", "empty", "two", "matrix"])
@pytest.mark.parametrize("attribute", [W.STORE_ATTRIBUTE, W.VDS_SOURCES_ATTRIBUTE])
def test_string_metadata_rejects_non_singleton_attributes(tmp_path, dtype, shape, attribute):
    path = tmp_path / "malformed-metadata.h5"
    value = "flat" if attribute == W.STORE_ATTRIBUTE else "[]"
    with h5py.File(path, "w") as file:
        file.create_dataset("primal", data=np.arange(3.))
        if attribute != W.STORE_ATTRIBUTE:
            file.attrs[W.STORE_ATTRIBUTE] = "flat"
        data = h5py.Empty(dtype) if shape is None else np.full(shape, value, dtype=dtype)
        file.attrs.create(attribute, data=data, dtype=dtype)

    # A regression can corrupt native memory, so isolate every malformed read.
    result = subprocess.run([
        sys.executable, "-c", '''
import sys
from dsbo.utils.io.storage import HDF5Writer as W
try:
    W.open_dataset(sys.argv[1])
except ValueError as error:
    assert sys.argv[2] in str(error), str(error)
    assert "exactly one element" in str(error), str(error)
else:
    raise AssertionError("Non-singleton string metadata was accepted")
''', str(path), attribute,
    ], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"Child exited with {result.returncode}:\n{result.stdout}\n{result.stderr}"


@pytest.mark.parametrize("dtype", [h5py.string_dtype("utf-8"), np.dtype("S16")], ids=["variable", "fixed"])
@pytest.mark.parametrize("shape", [(), (1,), (1, 1)])
def test_string_metadata_accepts_scalar_and_singleton_attributes(tmp_path, dtype, shape):
    path = tmp_path / "singleton-metadata.h5"
    values = np.arange(3.)
    with h5py.File(path, "w") as file:
        file.create_dataset("primal", data=values)
        file.attrs.create(W.STORE_ATTRIBUTE, data=np.full(shape, "flat", dtype=dtype), dtype=dtype)
        file.attrs.create(W.VDS_SOURCES_ATTRIBUTE, data=np.full(shape, "[]", dtype=dtype), dtype=dtype)
    with W.open_dataset(path) as view:
        np.testing.assert_array_equal(view.read("primal", slice(None)), values)


def test_noncontiguous_inputs_scalar_rank_and_owned_output(tmp_path):
    for i, values in enumerate((np.array(3.5), np.arange(24).reshape(4, 6).T, np.arange(10)[::-2])):
        path = tmp_path / f"{i}.h5"
        W.write_primal(path, values)
        loaded = W.open_primal(path)
        gc.collect()
        np.testing.assert_array_equal(loaded, values)
        assert loaded.shape == values.shape
        if loaded.dtype.kind == "f":
            tensor = torch.from_numpy(loaded)
            del loaded
            gc.collect()
            np.testing.assert_array_equal(tensor.numpy(), values)


def test_statistics_unsupported_arrays_and_noncanonical_bool(tmp_path):
    for value in (np.array(1.0), np.ones((2, 2, 2)), np.array(["text"]), np.array([1j]), np.empty(0)):
        assert W._statistics(value) is None
    values = np.array([0, 2, 255], dtype=np.uint8).view(np.bool_)
    W.write_primal(tmp_path / "bool.h5", values)
    with h5py.File(tmp_path / "bool.h5") as file:
        np.testing.assert_allclose(file["primal"].attrs[W.STATS_MEAN_ATTRIBUTE], values.astype(float).mean())


@pytest.mark.parametrize("matrix", [False, True], ids=["scalar-stats", "vector-stats"])
@pytest.mark.parametrize("case", ["cancellation", "short-cancellation", "constant", "short-constant", "large-offset", "near-ulp", "squared-sum"])
def test_statistics_numerical_accuracy_and_persistence(tmp_path, matrix, case):
    offset = 1e12 + 0.1
    # Analytical references avoid using another ordinary floating-point reduction
    # as the accuracy oracle. All offsets below are representable in float64.
    if case == "cancellation":
        values = np.r_[1e16, np.ones(100_000), -1e16]
        mean, deviation = 100_000 / 100_002, 2e32
    elif case == "short-cancellation":
        values = np.array([1e16, 1., -1e16])
        mean, deviation = 1 / 3, 2e32
    elif case in ("constant", "short-constant"):
        values = np.full(1_000_000 if case == "constant" else 9, offset)
        mean, deviation = offset, 0.
    elif case == "large-offset":
        values = offset + np.tile([-0.25, 0., 0.25], 33_334)
        mean, deviation = offset, 33_334 / 8
    elif case == "near-ulp":
        step = np.spacing(offset)
        values = np.array([offset, offset, offset + step])
        mean, deviation = offset, step * step * (2 / 3)
    else:
        values = np.r_[1e8, np.tile([1., -1.], 50_000), -1e8]
        mean, deviation = 0., 2e16 + 100_000
    expected_mean = np.array([mean, -mean]) if matrix else np.array(mean)
    expected_deviation = np.full(2, deviation) if matrix else np.array(deviation)
    if matrix:
        values = np.column_stack((values, -values))

    path = tmp_path / "accurate-statistics.h5"
    W.write_primal(path, values)
    reader = _hdf5.Reader(str(path))
    try:
        persisted = reader.statistics("primal")
    finally:
        reader.close()
    with h5py.File(path) as file:
        attributes = file["primal"].attrs
        independent = (
            attributes[W.STATS_COUNT_ATTRIBUTE],
            attributes[W.STATS_MEAN_ATTRIBUTE],
            attributes[W.STATS_SQUARED_DEVIATION_ATTRIBUTE],
        )
    for count, actual_mean, actual_deviation in (W._statistics(values), persisted, independent):
        assert count == len(values)
        assert actual_mean.shape == actual_deviation.shape == expected_mean.shape
        # Exact mean equality for constants also catches one-ULP drift at 1e12.
        if case in ("constant", "short-constant", "near-ulp"):
            np.testing.assert_array_equal(actual_mean, expected_mean)
        else:
            np.testing.assert_allclose(actual_mean, expected_mean, rtol=2e-15, atol=0)
        np.testing.assert_allclose(actual_deviation, expected_deviation, rtol=2e-15, atol=0)


@pytest.mark.parametrize("values,mean,deviation", [
    ([np.nan, 1.], np.nan, np.nan),
    ([np.inf, 1.], np.inf, np.nan),
    ([-np.inf, 1.], -np.inf, np.nan),
    ([np.inf, np.inf], np.inf, np.nan),
    ([np.inf, -np.inf], np.nan, np.nan),
    ([1e200, -1e200], 0., np.inf),
])
def test_statistics_nonfinite_propagation(values, mean, deviation):
    count, actual_mean, actual_deviation = W._statistics(np.array(values))
    assert count == len(values)
    np.testing.assert_equal(actual_mean, mean)
    np.testing.assert_equal(actual_deviation, deviation)


@pytest.mark.parametrize("matrix", [False, True], ids=["scalar-stats", "vector-stats"])
@pytest.mark.parametrize("constant", [False, True], ids=["large-offset", "constant"])
def test_combined_large_offset_statistics(tmp_path, matrix, constant):
    offset = 1e12 + 0.1
    if constant:
        batches = [np.full(500_000, offset), np.full(500_000, offset)]
        deviation = 0.
    else:
        residuals = np.tile([-0.25, 0.25], 50_000)
        batches = [offset - 0.5 + residuals, offset + 0.5 + residuals]
        deviation = 200_000 * (0.5 ** 2 + 0.25 ** 2)
    if matrix:
        batches = [np.column_stack((batch, -batch)) for batch in batches]
    sources = [tmp_path / "a.h5", tmp_path / "b.h5"]
    for source, batch in zip(sources, batches):
        W.write_primal(source, batch)
    destination = tmp_path / "combined.h5"
    W.reduce(sources, destination)
    reader = _hdf5.Reader(str(destination))
    try:
        persisted = reader.statistics("primal")
    finally:
        reader.close()
    direct = W._combine_statistics([W._statistics(batch) for batch in batches])
    expected_mean = np.array([offset, -offset]) if matrix else np.array(offset)
    expected_deviation = np.full(2, deviation) if matrix else np.array(deviation)
    for count, actual_mean, actual_deviation in (persisted, direct):
        assert count == sum(len(batch) for batch in batches)
        assert actual_mean.shape == actual_deviation.shape == expected_mean.shape
        np.testing.assert_array_equal(actual_mean, expected_mean)
        np.testing.assert_allclose(actual_deviation, expected_deviation, rtol=2e-15, atol=0)


@pytest.mark.parametrize("shape", [(3,), (3, 2), (3, 0)], ids=["scalar-stats", "vector-stats", "zero-width-stats"])
def test_combined_statistics_match_concatenated_data(tmp_path, shape):
    values = np.arange(np.prod(shape), dtype=float).reshape(shape)
    sources = [tmp_path / "a.h5", tmp_path / "b.h5"]
    batches = [values[:2], values[2:]]
    for source, batch in zip(sources, batches):
        W.write_primal(source, batch)
    destination = tmp_path / "combined.h5"
    W.reduce(sources, destination)
    reader = _hdf5.Reader(str(destination))
    try:
        combined = reader.statistics("primal")
    finally:
        reader.close()
    direct = W._combine_statistics([W._statistics(batch) for batch in batches])
    mean = values.mean(axis=0)
    deviation = ((values - mean) ** 2).sum(axis=0)
    for count, actual_mean, actual_deviation in (combined, direct):
        assert count == len(values)
        assert actual_mean.shape == actual_deviation.shape == mean.shape
        np.testing.assert_allclose(actual_mean, mean)
        np.testing.assert_allclose(actual_deviation, deviation)
    assert W._combine_statistics([]) is None
    assert W._combine_statistics([combined, None]) is None


@pytest.mark.parametrize("columns", [[""], ["a/a"], ["."], ["a\x00b"], ["same", "same"], [12]])
def test_column_validation(tmp_path, columns):
    with pytest.raises(ValueError, match="Column names"):
        W.write_primal(tmp_path / "bad.h5", np.zeros((len(columns), 2)), columns)


def test_unicode_names_metadata_and_column_view_axes(tmp_path):
    path = tmp_path / "数値.h5"
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    W.write_primal(path, values, ["β", 'quote"\\name'])
    with h5py.File(path) as file:
        assert list(file["primal"]) == ["β", 'quote"\\name']
        assert json.loads(file["primal"].attrs[W.COLUMNS_ATTRIBUTE]) == ["β", 'quote"\\name']
    np.testing.assert_array_equal(W.open_primal(path, ['quote"\\name', "β"]), values[[1, 0]])
    with W.open_dataset(path) as view:
        assert view.shape("primal") == (3, 4, 2)
        np.testing.assert_array_equal(view.read("primal", slice(None)), np.moveaxis(values, 0, -1))
        np.testing.assert_array_equal(view.read_indices(["primal"], [2, 0, 2])[:, 0], np.moveaxis(values, 0, -1)[[2, 0, 2]])


@pytest.mark.parametrize("store", ["flat", "row", "column"])
def test_lazy_slices_indices_and_close(tmp_path, store):
    path = tmp_path / "data.h5"
    values = np.arange(24, dtype=np.float64).reshape(6, 4)
    if store == "flat":
        W.write_primal(path, values)
        key, expected, keys = "primal", values, ["primal"]
    elif store == "column":
        W.write_primal(path, values.T, ["d", "c", "b", "a"])
        key, expected, keys = "primal", values, ["primal"]
    else:
        W.write_row_store(path, {"x": values[:, 0], "y": values[:, 1]})
        key, expected, keys = "x", values[:, 0], ["x", "y"]
    view = W.open_dataset(path)
    for index in (0, -1, slice(None), slice(1, 5, 2), slice(None, None, -1), slice(3, 1), slice(None, None, -2)):
        np.testing.assert_array_equal(view.read(key, index), expected[index])
    for indices in ([4, 1, 4, -1], [], [0, 2, 5]):
        actual = view.read_indices(keys, indices)
        expected_many = values[indices, :2] if store == "row" else values[indices, None, :]
        np.testing.assert_array_equal(actual, expected_many)
    with pytest.raises(IndexError):
        view.read(key, 6)
    with pytest.raises(IndexError):
        view.read_indices(keys, [-7])
    view.close()
    view.close()
    with pytest.raises(RuntimeError, match="closed"):
        view.read(key, 0)


def test_named_dataset_append_and_nested_unicode(tmp_path):
    path = tmp_path / "named.h5"
    W.write_named_datasets(path, {"nested/β": np.array([1., 2.])})
    W.write_named_datasets(path, {"other": np.array([3., 4.])})
    W.write_named_datasets(path, {"nested/β": np.array([5., 6.])})
    with W.open_dataset(path) as view:
        assert view.keys == ("nested/β", "other")
        np.testing.assert_array_equal(view.read("nested/β", slice(None)), [5, 6])
    with h5py.File(path) as file:
        assert list(file["nested"]) == ["β"]


@pytest.mark.parametrize("links", ["hard-cycles", "hard-aliases", "soft-links", "external-links"])
def test_dataset_discovery_matches_object_visitation(tmp_path, links):
    path = tmp_path / "links.h5"
    external = tmp_path / "external.h5"
    if links == "external-links":
        with h5py.File(external, "w") as file:
            file.create_dataset("remote/values", data=np.arange(4.) + 20)

    with h5py.File(path, "w", track_order=True) as file:
        file.attrs[W.STORE_ATTRIBUTE] = "flat"
        nested = file.create_group("nested", track_order=True)
        nested.create_dataset("β", data=np.arange(4.) + 10)
        file.create_dataset("primal", data=np.arange(4.))
        file.create_group("empty")
        file["named_type"] = np.dtype("i4")
        if links == "hard-cycles":
            file["cycle"] = file["/"]
            nested["self"] = nested
            nested.create_group("child")["parent"] = nested
        elif links == "hard-aliases":
            file["a_group"] = nested
            file["z_group"] = nested
            file["a_primal"] = file["primal"]
            file["z_primal"] = file["primal"]
            nested["a_values"] = nested["β"]
        elif links == "soft-links":
            file["a_soft_dataset"] = h5py.SoftLink("/primal")
            file["a_soft_group"] = h5py.SoftLink("/nested")
            file["dangling"] = h5py.SoftLink("/missing")
            file["soft_cycle"] = h5py.SoftLink("/soft_cycle")
            nested["dangling"] = h5py.SoftLink("/missing")
        else:
            file["a_external_dataset"] = h5py.ExternalLink(external.name, "/remote/values")
            file["a_external_group"] = h5py.ExternalLink(external.name, "/remote")
            file["dangling_file"] = h5py.ExternalLink("missing.h5", "/remote")
            file["dangling_object"] = h5py.ExternalLink(external.name, "/missing")
            nested["dangling"] = h5py.ExternalLink("missing.h5", "/remote")

        expected = {}
        for root in ("", "nested", "/nested", "empty"):
            datasets = {}
            prefix = root + "/" if root else ""
            parent = file[root] if root else file
            parent.visititems(
                lambda name, value: datasets.update({prefix + name: value[()].tolist()})
                if isinstance(value, h5py.Dataset) else None
            )
            expected[root] = datasets
        assert len(expected[""]) == 2

    # Keep a traversal regression from hanging or exhausting the pytest process.
    result = subprocess.run([
        sys.executable, "-c", '''
import json
import sys
import numpy as np
from dsbo.utils.io.storage import HDF5Writer as W, _hdf5

path, expected = sys.argv[1], json.loads(sys.argv[2])
reader = _hdf5.Reader(path)
try:
    for root, datasets in expected.items():
        assert reader.keys(root) == sorted(datasets), (root, reader.keys(root), datasets)
        for name, values in datasets.items():
            np.testing.assert_array_equal(reader.read(name), values)
finally:
    reader.close()
with W.open_dataset(path) as view:
    assert view.keys == tuple(sorted(expected[""])), view.keys
    for name, values in expected[""].items():
        np.testing.assert_array_equal(view.read(name, slice(None)), values)
''', str(path), json.dumps(expected),
    ], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"Child exited with {result.returncode}:\n{result.stdout}\n{result.stderr}"


def test_dataset_discovery_root_validation_and_closed_reader(tmp_path):
    path = tmp_path / "discovery.h5"
    W.write_primal(path, np.arange(3.))
    reader = _hdf5.Reader(str(path))
    try:
        assert reader.keys() == ["primal"]
        with pytest.raises(KeyError, match="does not exist"):
            reader.keys("missing")
        with pytest.raises(ValueError, match="not a group"):
            reader.keys("primal")
    finally:
        reader.close()
    with pytest.raises(RuntimeError, match="closed"):
        reader.keys()


@pytest.mark.parametrize("left,right", [(a, b) for a in DTYPES[:14] for b in DTYPES[:14]])
def test_gradient_numeric_promotion(tmp_path, left, right):
    a = np.array([1, 0], dtype=left)
    b = np.array([1, 1], dtype=right)
    sources = [tmp_path / "a.h5", tmp_path / "b.h5"]
    W.write_gradient(sources[0], a)
    W.write_gradient(sources[1], b)
    output = tmp_path / "sum.h5"
    W.aggregate_gradients(sources, output)
    actual = W.load_gradient(output)
    assert actual.dtype == (a + b).dtype
    np.testing.assert_array_equal(actual, a + b)


def test_sparse_gradient_aggregation_and_merge(tmp_path):
    primal, first, second, total = [tmp_path / f"{name}.h5" for name in ("primal", "first", "second", "total")]
    W.write_primal(primal, np.ones((3, 2)), ["c", "b", "a"])
    W.write_gradient(first, np.array([[1., 2.]], dtype=np.float32), ["a"])
    W.write_gradient(second, np.array([[3., 4.], [5., 6.]], dtype=np.float64), ["b", "a"])
    W.aggregate_gradients([first, second], total)
    np.testing.assert_array_equal(W.load_backward_gradient(primal, total), [[0, 0], [3, 4], [6, 8]])
    W.write_gradient(total, np.array([[9., 10.]]), ["c"], merge=True)
    np.testing.assert_array_equal(W.load_backward_gradient(primal, total), [[9, 10], [3, 4], [6, 8]])
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("named", [False, True], ids=["flat", "column"])
@pytest.mark.parametrize("null_target", ["gradient", "primal", "both"])
def test_null_gradients_rejected_before_shape_validation(tmp_path, named, null_target):
    primal, gradient = tmp_path / "primal.h5", tmp_path / "gradient.h5"
    values = np.array([3.5]) if named else np.array(3.5)
    columns = ["value"] if named else None
    W.write_primal(primal, values, columns)
    W.write_gradient(gradient, values, columns)
    for path, root in ((primal, "primal"), (gradient, "gradient")):
        if null_target not in (root, "both"):
            continue
        name = f"{root}/value" if named else root
        with h5py.File(path, "r+") as file:
            del file[name]
            file.create_dataset(name, data=h5py.Empty("f8"))

    with pytest.raises(ValueError, match="H5S_NULL"):
        list(W.gradient_file_contributions(primal, gradient))
    with pytest.raises(ValueError, match="H5S_NULL"):
        W.load_backward_gradient(primal, gradient)
    if null_target in ("gradient", "both"):
        with pytest.raises(ValueError, match="H5S_NULL"):
            W.load_gradient(gradient)


def test_sparse_scalar_gradients_do_not_treat_null_columns_as_missing(tmp_path):
    primal, gradient = tmp_path / "primal.h5", tmp_path / "gradient.h5"
    W.write_primal(primal, np.array([1., 2., 3.]), ["missing", "valid", "null"])
    W.write_gradient(gradient, np.array([4., 5.]), ["valid", "null"])
    np.testing.assert_array_equal(W.load_backward_gradient(primal, gradient), [0., 4., 5.])
    with h5py.File(gradient, "r+") as file:
        del file["gradient/null"]
        file.create_dataset("gradient/null", data=h5py.Empty("f8"))
    with pytest.raises(ValueError, match="H5S_NULL"):
        W.load_backward_gradient(primal, gradient)
    with pytest.raises(ValueError, match="H5S_NULL"):
        list(W.gradient_file_contributions(primal, gradient))


@pytest.mark.parametrize("named", [False, True], ids=["flat", "column"])
@pytest.mark.parametrize("shape", [(), (0,), (0, 2), (2, 0)])
def test_scalar_and_empty_gradients_remain_supported(tmp_path, named, shape):
    primal, gradient = tmp_path / "primal.h5", tmp_path / "gradient.h5"
    values = np.full(shape, 3.5)
    if named:
        values = np.expand_dims(values, 0)
    columns = ["value"] if named else None
    W.write_primal(primal, values, columns)
    W.write_gradient(gradient, values, columns)
    assert list(W.gradient_file_contributions(primal, gradient)) == [(str(primal), str(gradient), columns)]
    loaded = W.load_backward_gradient(primal, gradient)
    assert loaded.shape == values.shape
    np.testing.assert_array_equal(loaded, values)


@pytest.mark.parametrize("named", [False, True], ids=["flat", "column"])
def test_null_dataset_reduction_preserves_destination(tmp_path, named):
    source, destination = tmp_path / "null.h5", tmp_path / "destination.h5"
    with h5py.File(source, "w") as file:
        file.attrs[W.STORE_ATTRIBUTE] = "column" if named else "flat"
        file.create_dataset("primal/value" if named else "primal", data=h5py.Empty("f8"))
    W.write_primal(destination, np.array(3.5))
    before = destination.read_bytes()
    with pytest.raises(ValueError, match="H5S_NULL"):
        W.reduce([source], destination)
    assert destination.read_bytes() == before


@pytest.mark.parametrize("left,right", [((), (3,)), ((2, 1), (1, 3)), ((1, 2), (0, 2))])
@pytest.mark.parametrize("dtype", ["float16", "float32", "int64", "complex64"])
def test_gradient_broadcasting_matches_numpy(tmp_path, left, right, dtype):
    a = np.ones(left, dtype=dtype)
    b = np.ones(right, dtype=dtype)
    W.write_gradient(tmp_path / "a.h5", a)
    W.write_gradient(tmp_path / "b.h5", b)
    W.aggregate_gradients([tmp_path / "a.h5", tmp_path / "b.h5"], tmp_path / "sum.h5")
    np.testing.assert_array_equal(W.load_gradient(tmp_path / "sum.h5"), a + b)


def test_hdf5_2_native_complex_input(tmp_path):
    path = tmp_path / "native-complex.h5"
    values = (np.arange(6).reshape(2, 3) * (1 + 2j)).astype(np.complex128)
    with h5py.File(path, "w", libver="latest") as file:
        file.attrs[W.STORE_ATTRIBUTE] = "flat"
        dataset = file.create_dataset("primal", values.shape, dtype=h5py.h5t.COMPLEX_IEEE_F64LE)
        dataset[:] = values
    np.testing.assert_array_equal(W.open_primal(path), values)
    with W.open_dataset(path) as view:
        np.testing.assert_array_equal(view.read("primal", slice(None)), values)
        np.testing.assert_array_equal(view.read_indices(["primal"], [1, 0, 1])[:, 0], values[[1, 0, 1]])


@pytest.mark.parametrize("columns", [None, ["second", "first"]])
def test_nested_repeated_vds_gradient_transpose(tmp_path, columns):
    a, b, inner, outer, grad = [tmp_path / f"{name}.h5" for name in ("a", "b", "inner", "outer", "adjoint")]
    first = np.arange(4, dtype=np.float32).reshape(2, 2) if columns else np.arange(2, dtype=np.float32)
    second = np.arange(2, dtype=np.float32).reshape(2, 1) if columns else np.arange(1, dtype=np.float32)
    W.write_primal(a, first, columns)
    W.write_primal(b, second, columns)
    W.reduce([a, b], inner)
    W.reduce([inner, a, inner], outer)
    primal = W.open_primal(outer)
    gradient = np.arange(primal.size, dtype=np.float32).reshape(primal.shape)
    W.write_gradient(grad, gradient, columns)
    contributions = list(W.gradient_file_contributions(outer, grad))
    assert [Path(path).name for path, _, _ in contributions] == ["a.h5", "b.h5", "a.h5", "a.h5", "b.h5"]
    offset = 0
    for path, value, selected in contributions:
        count = W.open_primal(path).shape[-1]
        expected = gradient[:, offset:offset + count] if columns else gradient[offset:offset + count]
        np.testing.assert_array_equal(value, expected)
        assert selected == columns
        offset += count
    with h5py.File(outer) as file:
        dataset = file["primal/second"] if columns else file["primal"]
        assert dataset.is_virtual
        assert all(not os.path.isabs(mapping.file_name) for mapping in dataset.virtual_sources())


@pytest.mark.parametrize("shape", [(), (0, 2), (2, 0), (0,)])
def test_scalar_and_empty_virtual_concatenations(tmp_path, shape):
    source, output = tmp_path / "source.h5", tmp_path / "output.h5"
    values = np.ones(shape, dtype=np.float32)
    W.write_primal(source, values)
    W.reduce([source, source], output)
    expected = np.array([values, values]) if not shape else np.concatenate([values, values])
    np.testing.assert_array_equal(W.open_primal(output), expected)
    with h5py.File(output) as file:
        assert file["primal"].is_virtual
        assert json.loads(file["primal"].attrs[W.PARTITION_SIZES_ATTRIBUTE]) == [1 if not shape else shape[0]] * 2


def test_vds_relocation_and_missing_sources(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    W.write_primal(original / "source.h5", np.arange(3.))
    W.reduce([original / "source.h5"] * 2, original / "nested" / "reduced.h5")
    copied = tmp_path / "copy"
    shutil.copytree(original, copied)
    shutil.rmtree(original)
    np.testing.assert_array_equal(W.open_primal(copied / "nested/reduced.h5"), [0, 1, 2, 0, 1, 2])
    (copied / "source.h5").unlink()
    with pytest.raises(FileNotFoundError, match="missing source"):
        W.open_primal(copied / "nested/reduced.h5")


def test_gradient_validation_is_metadata_only_and_untagged(tmp_path):
    primal, grad = tmp_path / "primal.h5", tmp_path / "adjoint.h5"
    with h5py.File(primal, "w") as file:
        file.create_dataset("primal", shape=(10**9,), dtype="f8")
    with h5py.File(grad, "w") as file:
        file.create_dataset("gradient", shape=(10**9,), dtype="f8")
    assert list(W.gradient_file_contributions(primal, grad)) == [(str(primal), str(grad), None)]
    with h5py.File(grad, "w") as file:
        file.create_dataset("gradient", data=np.array([b"not numeric"]))
    with pytest.raises(ValueError, match="Gradient shape"):
        list(W.gradient_file_contributions(primal, grad))
    with h5py.File(primal, "w") as file:
        file.create_dataset("primal", shape=(1,), dtype="f8")
    with pytest.raises(ValueError, match="numeric values"):
        list(W.gradient_file_contributions(primal, grad))


def test_failures_and_gradient_atomicity(tmp_path):
    with pytest.raises(FileNotFoundError):
        W.open_primal(tmp_path / "missing.h5")
    with pytest.raises(ValueError, match="at least one"):
        W.reduce([], tmp_path / "none.h5")
    source = tmp_path / "source.h5"
    W.write_primal(source, np.array([1.]))
    before = source.read_bytes()
    with pytest.raises(ValueError, match="overwrite a source"):
        W.reduce([source], source)
    assert source.read_bytes() == before
    gradient = tmp_path / "gradient.h5"
    W.write_gradient(gradient, np.array([2.]))
    before = gradient.read_bytes()
    with pytest.raises(ValueError):
        W.write_gradient(gradient, np.ones((2, 1)), ["one"])
    assert gradient.read_bytes() == before
    assert not list(tmp_path.glob("*.partial"))


def test_environment_and_concurrent_native_calls(tmp_path):
    environment = _hdf5.environment()
    assert environment["version"][0] >= 2
    assert environment["threadsafe"] is True

    def work(index):
        path = tmp_path / f"thread-{index}.h5"
        expected = np.arange(100, dtype=np.float32) + index
        W.write_primal(path, expected)
        with W.open_dataset(path) as reader:
            for _ in range(5):
                np.testing.assert_array_equal(reader.read("primal", slice(None)), expected)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(work, range(12)))


def test_python_extension_keeps_dependency_symbols_private():
    extension = ctypes.CDLL(_hdf5.__file__)
    for symbol in ("H5open", "H5get_libversion", "zlibVersion", "inflate", "deflate"):
        with pytest.raises(AttributeError):
            getattr(extension, symbol)


@pytest.mark.parametrize("first", ["h5py", "dsbo"])
def test_bundled_hdf5_coexists_with_h5py(tmp_path, first):
    subprocess.run([
        sys.executable, "-c", f"import {first}\n" + '''
import h5py
import numpy as np
from dsbo.utils.io.storage import HDF5Writer as W, _hdf5
assert _hdf5.environment()["threadsafe"]
values = np.arange(24., dtype=np.float64).reshape(6, 4)
with h5py.File("compressed.h5", "w") as file:
    file.create_dataset("primal", data=values, compression="gzip", shuffle=True)
np.testing.assert_array_equal(W.open_primal("compressed.h5"), values)
W.write_primal("native.h5", values)
with h5py.File("native.h5") as file:
    np.testing.assert_array_equal(file["primal"][:], values)
''',
    ], cwd=tmp_path, check=True)


def test_bundled_source_archives_and_notices():
    root = Path(__file__).resolve().parents[1]
    configuration = (root / "cpp/cmake/BundledHDF5.cmake").read_text()
    archives = re.findall(r'URL "[^"\n]*/vendor/([^"\n]+)"\s+URL_HASH SHA256=([0-9a-f]{64})', configuration)
    assert len(archives) == 2
    for name, checksum in archives:
        archive = root / "cpp/vendor" / name
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == checksum
        with tarfile.open(archive) as source:
            member = next(member for member in source if member.name.removeprefix("./") == name.removesuffix(".tar.gz") + "/LICENSE")
            upstream_license = source.extractfile(member).read()
        notice = "HDF5.txt" if name.startswith("hdf5-") else "zlib.txt"
        assert (LICENSE_DIRECTORY / notice).read_bytes().strip() == upstream_license.strip()


def test_third_party_notices_are_distribution_metadata():
    package = distribution("dsbo")
    expected = {f"LICENSES/{path.name}" for path in LICENSE_DIRECTORY.iterdir() if path.is_file()}
    assert set(package.metadata.get_all("License-File", [])) == expected
    # Editable installs may expose source-tree egg-info instead of wheel
    # dist-info. Both must list the same notices and locate their actual bytes.
    files = package.files or []
    for name in expected:
        recorded = [path for path in files if path.as_posix() == name or path.as_posix().endswith("/licenses/" + name)]
        assert len(recorded) == 1, name
        assert Path(package.locate_file(recorded[0])).read_bytes() == (LICENSE_DIRECTORY / Path(name).name).read_bytes()


def test_native_headers_are_distribution_data():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "dsbo/utils/io/storage/include").exists()
    headers = sorted(INCLUDE_DIRECTORY.rglob("*.hpp"))
    assert {path.relative_to(INCLUDE_DIRECTORY).as_posix() for path in headers} == {
        "dsbo/hdf5.hpp", "dsbo/third_party/json.hpp",
    }
    package = distribution("dsbo")
    files = package.files or []
    for header in headers:
        relative = header.relative_to(INCLUDE_DIRECTORY)
        # Source-tree metadata lists the canonical inputs; wheel metadata lists
        # the installed copies. Neither requires a second source header tree.
        names = {header.relative_to(root).as_posix(), "dsbo/utils/io/storage/include/" + relative.as_posix()}
        recorded = [path for path in files if path.as_posix() in names]
        assert len(recorded) == 1, relative
        assert Path(package.locate_file(recorded[0])).read_bytes() == header.read_bytes()


def test_production_storage_does_not_import_h5py():
    subprocess.run([
        sys.executable, "-c",
        "import sys; from dsbo.utils.io.storage import HDF5Writer; assert 'h5py' not in sys.modules",
    ], check=True)
