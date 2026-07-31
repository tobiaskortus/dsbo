from __future__ import annotations

import os
import threading
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
import torch

from dsbo.config import config
from dsbo.utils.io.autograd import InMemoryFile
from dsbo.utils.io.hdf5 import HDF5File
from dsbo.utils.io.reduce import LazyReductionResolver, reduce_registered_file
from dsbo.utils.io.root import ROOTFile


def write_h5_files(tempdir: str, *values: float) -> list[str]:
    paths = []
    for i, v in enumerate(values):
        p = os.path.join(tempdir, f"src_{i}.h5")
        HDF5File.write(p, torch.tensor([v]))
        paths.append(p)
    return paths


@pytest.fixture
def tempdir():
    with TemporaryDirectory() as tmp:
        yield tmp


def test_resolver_does_not_materialize_before_resolve(tempdir):
    sources = write_h5_files(tempdir, 1.0, 2.0, 3.0)
    destination = os.path.join(tempdir, "reduced.h5")

    resolver = LazyReductionResolver(sources, destination)
    assert not os.path.isfile(destination)

    path = resolver.resolve()
    assert path == destination
    assert os.path.isfile(destination)
    torch.testing.assert_close(HDF5File.open(destination, "cpu"), torch.tensor([1.0, 2.0, 3.0]))


def test_resolver_writes_and_reads_marker(tempdir):
    sources = write_h5_files(tempdir, 3.0, 4.0)
    destination = os.path.join(tempdir, "reduced.h5")

    resolver = LazyReductionResolver(sources, destination)
    resolver.resolve()

    marker_path = destination + ".resolve_id"
    assert os.path.isfile(marker_path)
    with open(marker_path) as f:
        assert f.read().strip() == resolver._resolve_id


def test_resolver_skips_reduce_on_same_object_reuse(tempdir):
    sources = write_h5_files(tempdir, 4.0, 5.0)
    destination = os.path.join(tempdir, "reduced.h5")

    resolver = LazyReductionResolver(sources, destination)
    resolver.resolve()

    with patch("dsbo.utils.io.reduce.reduce_registered_file") as mock_reduce:
        resolver.resolve()
        mock_reduce.assert_not_called()


def test_resolver_called_on_new_run_with_existing_files(tempdir):
    destination = os.path.join(tempdir, "reduced.h5")

    sources_a = write_h5_files(tempdir, 1.0, 2.0)
    LazyReductionResolver(sources_a, destination).resolve()
    torch.testing.assert_close(HDF5File.open(destination, "cpu"), torch.tensor([1.0, 2.0]))

    sources_b = write_h5_files(tempdir, 3.0, 4.0)
    LazyReductionResolver(sources_b, destination).resolve()
    torch.testing.assert_close(HDF5File.open(destination, "cpu"), torch.tensor([3.0, 4.0]))


def test_concurrent_resolvers_only_one_reduces(tempdir):
    sources = write_h5_files(tempdir, 5.0, 6.0)
    destination = os.path.join(tempdir, "reduced.h5")

    resolver = LazyReductionResolver(sources, destination)

    with patch("dsbo.utils.io.reduce.reduce_registered_file", wraps=reduce_registered_file) as mock_reduce:
        threads = [threading.Thread(target=resolver.resolve) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert mock_reduce.call_count == 1


def test_resolver_raises_reducer_error_on_missing_source(tempdir):
    sources = [os.path.join(tempdir, "nonexistent.h5")]
    destination = os.path.join(tempdir, "reduced.h5")

    resolver = LazyReductionResolver(sources, destination)
    with pytest.raises(FileNotFoundError, match="nonexistent"):
        resolver.resolve()


def test_resolver_preserves_context_in_error_message(tempdir):
    sources = [os.path.join(tempdir, "missing.h5")]
    destination = os.path.join(tempdir, "reduced.h5")

    resolver = LazyReductionResolver(sources, destination, context="MyModule input data.h5")
    with pytest.raises(FileNotFoundError, match="MyModule input data.h5"):
        resolver.resolve()


def test_resolver_rejects_backend_without_reduction(tempdir):
    sources = write_h5_files(tempdir, 5.0, 6.0)
    destination = os.path.join(tempdir, "reduced.h5")

    resolver = LazyReductionResolver(sources, destination)
    with config.use(file_backend=InMemoryFile):
        with pytest.raises(NotImplementedError, match="does not support file reduction"):
            resolver.resolve()


def test_resolver_uses_root_backend(tempdir):
    sources = []
    for index, value in enumerate((5.0, 6.0)):
        source = os.path.join(tempdir, f"source_{index}.root")
        ROOTFile.write(source, torch.tensor([value]))
        sources.append(source)
    destination = os.path.join(tempdir, "reduced.root")

    with config.use(file_backend=ROOTFile):
        LazyReductionResolver(sources, destination).resolve()

    with torch.no_grad():
        torch.testing.assert_close(ROOTFile.open(destination, "cpu"), torch.tensor([5.0, 6.0]))


def test_resolver_rejects_non_hdf5_source_with_error(tempdir):
    source = os.path.join(tempdir, "model.pt")
    torch.save(torch.tensor([1.0]), source)
    destination = os.path.join(tempdir, "reduced.h5")

    resolver = LazyReductionResolver([source], destination)
    with pytest.raises(TypeError, match="does not support"):
        resolver.resolve()
