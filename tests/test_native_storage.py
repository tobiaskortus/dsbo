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
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

import dsbo.utils.io.storage as storage
from dsbo.utils.io.gradient import (
    resolve_gradient,
    use_gradient_pass,
    use_gradient_writer,
)
from dsbo.utils.io.storage import HDF5Writer as W
from dsbo.utils.io.storage import _hdf5
from dsbo.utils.io.workdir import Input, Output, Workdir


DTYPES = ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64", "complex64", "complex128", ">i4", ">f8", ">c16"]
STORAGE_DIRECTORY = Path(storage.__file__).resolve().parent


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(), (4,), (2, 3), (2, 1, 3), (0, 2), (2, 0)])
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


def test_python_does_not_require_external_hdf5(tmp_path):
    env = os.environ.copy()
    for variable in ("HDF5_ROOT", "HDF5_INCLUDE_DIR", "HDF5_LIBRARY_DIR", "PKG_CONFIG", "PKG_CONFIG_PATH"):
        env[variable] = str(tmp_path / "missing-sdk")
    subprocess.run([
        sys.executable, "-c", '''
import numpy as np
from dsbo.utils.io.storage import HDF5Writer as W, _hdf5
assert _hdf5.environment()["threadsafe"]
W.write_primal("test.h5", np.arange(3.))
np.testing.assert_array_equal(W.open_primal("test.h5"), np.arange(3.))
''',
    ], cwd=tmp_path, env=env, check=True)


def test_bundled_source_archives_and_notices():
    root = Path(__file__).resolve().parents[1]
    configuration = (root / "cpp/python/CMakeLists.txt").read_text()
    archives = re.findall(r'URL "[^"\n]*/vendor/([^"\n]+)"\s+URL_HASH SHA256=([0-9a-f]{64})', configuration)
    assert len(archives) == 2
    for name, checksum in archives:
        archive = root / "cpp/vendor" / name
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == checksum
        with tarfile.open(archive) as source:
            member = next(member for member in source if member.name.removeprefix("./") == name.removesuffix(".tar.gz") + "/LICENSE")
            upstream_license = source.extractfile(member).read()
        notice = "HDF5.txt" if name.startswith("hdf5-") else "zlib.txt"
        assert (STORAGE_DIRECTORY / "licenses" / notice).read_bytes().strip() == upstream_license.strip()


def test_packaged_native_headers_and_notices_can_be_copied(tmp_path):
    sdk = tmp_path / "sdk"
    shutil.copytree(STORAGE_DIRECTORY / "include", sdk / "include")
    shutil.copytree(STORAGE_DIRECTORY / "licenses", sdk / "licenses")
    assert (sdk / "include/dsbo/hdf5.hpp").is_file()
    assert (sdk / "include/dsbo/third_party/json.hpp").is_file()
    assert (sdk / "licenses/HDF5.txt").is_file()
    assert {path.name for path in sdk.iterdir()} == {"include", "licenses"}
    assert not any(path.suffix in {".so", ".dylib", ".dll", ".py", ".pyc"} for path in sdk.rglob("*"))
    for path in sdk.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (STORAGE_DIRECTORY / path.relative_to(sdk)).read_bytes()


@pytest.fixture(scope="session")
def native_executable(tmp_path_factory):
    if os.environ.get("DSBO_TEST_NATIVE") != "1":
        pytest.skip("Set DSBO_TEST_NATIVE=1 and HDF5_ROOT to build/test copied C++ headers")
    root = Path(__file__).resolve().parents[1]
    build = tmp_path_factory.mktemp("native-sdk")
    sdk = build / "sdk"
    shutil.copytree(STORAGE_DIRECTORY / "include", sdk / "include")
    shutil.copytree(STORAGE_DIRECTORY / "licenses", sdk / "licenses")
    subprocess.run([
        "cmake", "-S", str(root / "cpp"), "-B", str(build / "build"),
        f"-DDSBO_SDK_INCLUDE_DIR={sdk / 'include'}", "-DCMAKE_BUILD_TYPE=Debug",
    ], check=True)
    subprocess.run(["cmake", "--build", str(build / "build"), "--parallel", "2"], check=True)
    subprocess.run(["ctest", "--test-dir", str(build / "build"), "--output-on-failure"], check=True)
    return build / "build" / "dsbo_native_step"


def test_native_executable_forward_tangent_and_gradient_routing(tmp_path, native_executable):
    from dsbo.utils.io.storage import HDF5File

    with Workdir.create(str(tmp_path / "work")):
        source = Workdir.register_file("native", "input.h5")
        target = Workdir.register_file("native", "output.h5")
        inp, out = Input(source), Output(target)
        values, tangent = np.arange(4, dtype=np.float64), np.ones(4)
        W.write_primal_tangent(inp(), values, tangent)
        subprocess.run([str(native_executable), "forward", inp(), out()], check=True)
        actual, actual_tangent = W.open_primal_tangent(out())
        np.testing.assert_array_equal(actual, 2 * values)
        np.testing.assert_array_equal(actual_tangent, 2 * tangent)
        with use_gradient_pass():
            with use_gradient_writer("downstream"):
                W.write_gradient(Input(target).gradient(), np.ones(4))
            with use_gradient_writer("native"):
                subprocess.run([str(native_executable), "backward", out.gradient(), inp.gradient()], check=True)
            np.testing.assert_array_equal(HDF5File.read_backward_gradient(inp()).numpy(), np.full(4, 2))
            assert Path(resolve_gradient(inp())).is_file()


def test_container_without_python(tmp_path):
    image = os.environ.get("DSBO_TEST_CONTAINER_IMAGE")
    if not image:
        pytest.skip("Set DSBO_TEST_CONTAINER_IMAGE to a built examples/cpp_container image")
    subprocess.run(["docker", "run", "--rm", "--entrypoint", "/bin/sh", image, "-c", "! command -v python && ! command -v python3"], check=True)
    source, output, incoming, outgoing = [tmp_path / f"{name}.h5" for name in ("source", "output", "incoming", "outgoing")]
    W.write_primal(source, np.arange(4, dtype=np.float64))
    W.write_gradient(incoming, np.ones(4))
    for mode, first, second in (("forward", source, output), ("backward", incoming, outgoing)):
        subprocess.run(["docker", "run", "--rm", "-v", f"{tmp_path}:/work", image, mode, f"/work/{first.name}", f"/work/{second.name}"], check=True)
    np.testing.assert_array_equal(W.open_primal(output), np.arange(4) * 2)
    np.testing.assert_array_equal(W.load_gradient(outgoing), np.full(4, 2))


def test_production_storage_does_not_import_h5py():
    subprocess.run([
        sys.executable, "-c",
        "import sys; from dsbo.utils.io.storage import HDF5Writer; assert 'h5py' not in sys.modules",
    ], check=True)


def test_bundled_python_ignores_non_threadsafe_hdf5(tmp_path):
    prefix = os.environ.get("DSBO_TEST_UNSAFE_HDF5")
    if not prefix:
        pytest.skip("Set DSBO_TEST_UNSAFE_HDF5 to test isolation from a non-thread-safe HDF5 >=2.0")
    prefix = Path(prefix).resolve()
    library = prefix / ("lib64" if (prefix / "lib64").is_dir() else "lib")
    env = os.environ.copy()
    variable = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    env[variable] = str(library)
    env["HDF5_ROOT"] = str(prefix)
    shared = library / ("libhdf5.dylib" if sys.platform == "darwin" else "libhdf5.so")
    subprocess.run([
        sys.executable, "-c", f'''import ctypes
unsafe = ctypes.CDLL({str(shared)!r}, mode=ctypes.RTLD_GLOBAL)
safe = ctypes.c_bool()
assert unsafe.H5is_library_threadsafe(ctypes.byref(safe)) >= 0 and not safe.value
from dsbo.utils.io.storage import _hdf5
assert _hdf5.environment()["threadsafe"]
''',
    ], cwd=tmp_path, env=env, check=True)


def test_native_non_threadsafe_hdf5_is_rejected(tmp_path):
    prefix = os.environ.get("DSBO_TEST_UNSAFE_HDF5")
    if not prefix:
        pytest.skip("Set DSBO_TEST_UNSAFE_HDF5 to test standalone C++ dependency validation")
    prefix = Path(prefix).resolve()
    library = prefix / ("lib64" if (prefix / "lib64").is_dir() else "lib")
    root = Path(__file__).resolve().parents[1]
    executable = tmp_path / "unsafe-native-step"
    subprocess.run([
        "c++", "-std=c++17", "-pthread", f"-I{STORAGE_DIRECTORY / 'include'}", f"-I{prefix / 'include'}",
        str(root / "cpp/examples/workflow_step.cpp"), f"-L{library}", f"-Wl,-rpath,{library}",
        "-lhdf5", "-o", str(executable),
    ], check=True)
    result = subprocess.run([str(executable)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "thread-safe HDF5 installation" in result.stderr
