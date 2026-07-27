import os
from pathlib import Path
from unittest.mock import patch
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch
import torch.autograd.forward_ad as fw

from dsbo.utils.io.autograd import HDF5File, InMemoryFile
from dsbo.utils.io.sharded import ShardedHDF5Storage

def test_hdf5_file_primal_only():
    #dual_level==-1 and no_grad
    with torch.no_grad():
        x = torch.tensor([1.0, 2.0])
        x = x * 3

        with TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, "x.h5")
            HDF5File.write(file_path, x)
            x_loaded = HDF5File.open(file_path, "cpu")

            torch.testing.assert_close(x_loaded, torch.tensor([3.0, 6.0]))


def test_hdf5_file_forward_ad_tangent():
    with fw.dual_level():
        x = fw.make_dual(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 0.0]))
        x = x * 3

        with TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, "x.h5")
            HDF5File.write(file_path, x)
            x_loaded = HDF5File.open(file_path, "cpu")
            tangent = fw.unpack_dual(x_loaded).tangent
            
            torch.testing.assert_close(x_loaded, torch.tensor([3.0, 6.0]))
            torch.testing.assert_close(tangent, torch.tensor([3.0, 0.0]))


def test_hdf5_file_backward_gradient():
    with TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "x.h5")

        primal = torch.tensor([2.0, 3.0])
        HDF5File.write(file_path, primal)

        with torch.enable_grad():
            x = HDF5File.open(file_path, "cpu")
            y = (x * x).sum()

            # Check that the autograd hook was registered during open()
            assert file_path in HDF5File.handles
            
            y.backward()

        assert os.path.exists(os.path.join(tempdir, "x_gradient.h5"))
        grad = HDF5File.read_backward_gradient(file_path)
        torch.testing.assert_close(grad, 2.0 * primal)


def test_hdf5_file_reduce_creates_virtual_dataset():
    import h5py

    with TemporaryDirectory() as tempdir:
        sources = [os.path.join(tempdir, f"source_{i}.h5") for i in range(2)]
        HDF5File.write(sources[0], torch.tensor([[1.0], [2.0]]))
        HDF5File.write(sources[1], torch.tensor([[3.0]]))
        destination = os.path.join(tempdir, "reduced.h5")

        HDF5File.reduce(sources, destination)

        with h5py.File(destination, "r") as file:
            assert file["primal"].is_virtual
            assert all(not source.file_name.startswith("/") for source in file["primal"].virtual_sources())
        torch.testing.assert_close(
            HDF5File.open(destination, "cpu"),
            torch.tensor([[1.0], [2.0], [3.0]]),
        )


def test_hdf5_virtual_dataset_rejects_missing_sources():
    with TemporaryDirectory() as tempdir:
        source = os.path.join(tempdir, "source.h5")
        destination = os.path.join(tempdir, "reduced.h5")
        HDF5File.write(source, torch.tensor([[1.0]]))
        HDF5File.reduce([source], destination)
        assert HDF5File.dependencies(destination) == [source]
        os.remove(source)

        with pytest.raises(FileNotFoundError, match="missing source"):
            HDF5File.validate(destination)
        with pytest.raises(FileNotFoundError, match="missing source"):
            HDF5File.open(destination, "cpu")


def test_sharded_hdf5_uses_autograd_schema_and_vds_reduction():
    with TemporaryDirectory() as tempdir:
        ShardedHDF5Storage.max_rows_per_shard = 2
        source = os.path.join(tempdir, "source.h5")
        ShardedHDF5Storage.write(source, torch.tensor([[1.0], [2.0], [3.0]]))
        shards = sorted(Path(tempdir).glob("source.*.h5"))
        assert len(shards) == 2
        destination = os.path.join(tempdir, "reduced.h5")
        ShardedHDF5Storage.reduce([str(path) for path in shards], destination)
        torch.testing.assert_close(
            HDF5File.open(destination, "cpu"),
            torch.tensor([[1.0], [2.0], [3.0]]),
        )


def test_inmemory_file_primal_only():
    with torch.no_grad():
        x = torch.tensor([1.0, 2.0])
        x = x * 3
        file_path = "inmem_x"

        InMemoryFile._storage.clear()
        InMemoryFile.write(file_path, x)

        x_loaded = InMemoryFile.open(file_path, "cpu")

        torch.testing.assert_close(x_loaded, torch.tensor([3.0, 6.0]))


def test_inmemory_file_forward_ad_tangent():
    with fw.dual_level():
        x = fw.make_dual(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 0.0]))
        x = x * 3
        file_path = "inmem_x"

        InMemoryFile._storage.clear()
        InMemoryFile.write(file_path, x)

        x_loaded = InMemoryFile.open(file_path, "cpu")
        tangent = fw.unpack_dual(x_loaded).tangent
        
        torch.testing.assert_close(x_loaded, torch.tensor([3.0, 6.0]))
        torch.testing.assert_close(tangent, torch.tensor([3.0, 0.0]))


def test_inmemory_file_backward_gradient():
    file_path = "inmem_x"
    InMemoryFile._storage.clear()
    primal = torch.tensor([2.0, 3.0])
    InMemoryFile.write(file_path, primal)

    with torch.enable_grad():
        x = InMemoryFile.open(file_path, "cpu")
        y = (x * x).sum()

        assert file_path in InMemoryFile._storage

        y.backward()

    grad = InMemoryFile.read_backward_gradient(file_path)
    torch.testing.assert_close(grad, 2.0 * primal)
