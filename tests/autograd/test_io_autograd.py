import os
import h5py
import torch
import pytest
import numpy as np
from tempfile import TemporaryDirectory
import torch.autograd.forward_ad as fw

from dsbo.utils.io.autograd import HDF5File, InMemoryFile


def _record_results(path: str, outputs: dict[str, torch.Tensor]) -> None:
    HDF5File.write_named_datasets(
        path,
        {key: value.numpy() for key, value in outputs.items()},
    )


def aggregate(ds) -> tuple[torch.Tensor, torch.Tensor]:
    xs = torch.stack([ds[i][0] for i in range(len(ds))], dim=0)
    ys = torch.stack([ds[i][1] for i in range(len(ds))], dim=0)
    return xs, ys


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


def test_hdf5_file_named_columns():
    with TemporaryDirectory() as tempdir, torch.no_grad():
        file_path = os.path.join(tempdir, "x.h5")
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        columns = ["first", "second", "third"]

        HDF5File.write(file_path, values, columns=columns)

        with h5py.File(file_path, "r") as file:
            assert isinstance(file["primal"], h5py.Group)
            np.testing.assert_array_equal(file["primal/second"][:], values[1].numpy())
        torch.testing.assert_close(HDF5File.open(file_path, "cpu"), values)
        # Make sure the given column order is preserved when loading
        torch.testing.assert_close(HDF5File.open(file_path, "cpu", columns=["third", "first"]), values[[2, 0]])


def test_hdf5_file_named_columns_forward_ad_tangent():
    with TemporaryDirectory() as tempdir, fw.dual_level():
        file_path = os.path.join(tempdir, "x.h5")
        primal = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        tangent = torch.tensor([[0.0, 1.0], [2.0, 3.0]])

        HDF5File.write(file_path, fw.make_dual(primal, tangent), columns=["first", "second"])

        loaded = HDF5File.open(file_path, "cpu", columns=["second"])

        torch.testing.assert_close(loaded, primal[[1]])
        torch.testing.assert_close(fw.unpack_dual(loaded).tangent, tangent[[1]])


def test_hdf5_file_named_columns_backward_gradient():
    HDF5File.reset_gradient_state()
    with TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "x.h5")
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        columns = ["first", "second", "third"]

        HDF5File.write(file_path, values, columns=columns)

        x = HDF5File.open(file_path, "cpu", columns=["third", "first"])
        weights = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
        (x * weights).sum().backward()

        # Backward gradient updates according to how the data was loaded, column "third" and "first",
        # while "second" is 0.
        torch.testing.assert_close(
            HDF5File.read_backward_gradient(file_path),
            torch.stack((weights[1], torch.zeros(2), weights[0])),
        )
        with h5py.File(HDF5File._gradient_path(file_path), "r") as file:
            assert list(file["gradient"].keys()) == ["third", "first"]


def test_hdf5_file_named_columns_backward_gradient_without_selection():
    HDF5File.reset_gradient_state()
    with TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "x.h5")
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        HDF5File.write(file_path, values, columns=["first", "second"])

        x = HDF5File.open(file_path, "cpu")
        (x * x).sum().backward()

        torch.testing.assert_close(HDF5File.read_backward_gradient(file_path), 2.0 * values)


def test_hdf5_file_named_columns_accumulate_gradients():
    HDF5File.reset_gradient_state()
    with TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "x.h5")
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        HDF5File.write(file_path, values, columns=["first", "second"])

        first = HDF5File.open(file_path, "cpu", columns=["first"])
        (first * 2.0).sum().backward()
        second = HDF5File.open(file_path, "cpu", columns=["second"])
        (second * 3.0).sum().backward()

        # The second gradient update has to leave the first one intact.
        torch.testing.assert_close(HDF5File.read_backward_gradient(file_path), torch.tensor([[2.0, 2.0], [3.0, 3.0]]))


@pytest.mark.parametrize("selection_first", [False, True])
def test_hdf5_file_named_columns_accumulate_full_and_selected(selection_first):
    HDF5File.reset_gradient_state()
    with TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "x.h5")
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        HDF5File.write(file_path, values, columns=["first", "second"])

        def backward_selection():
            selected = HDF5File.open(file_path, "cpu", columns=["second"])
            (selected * 2.0).sum().backward()

        def backward_full():
            HDF5File.open(file_path, "cpu").sum().backward()

        operations = (backward_selection, backward_full) if selection_first else (backward_full, backward_selection)
        for operation in operations:
            operation()

        torch.testing.assert_close(HDF5File.read_backward_gradient(file_path), torch.tensor([[1.0, 1.0], [3.0, 3.0]]))


def test_hdf5_file_named_columns_validate_dimension_zero():
    with TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "x.h5")

        with pytest.raises(ValueError, match="2 columns.*size 3.*dimension 0"):
            HDF5File.write(file_path, torch.ones((3, 2)), columns=["first", "second"])


def test_hdf5_file_named_columns_reduce():
    with TemporaryDirectory() as tempdir, torch.no_grad():
        sources = [os.path.join(tempdir, f"source_{i}.h5") for i in range(2)]
        HDF5File.write(sources[0], torch.tensor([[1.0, 2.0], [4.0, 5.0]]), columns=["first", "second"])
        HDF5File.write(sources[1], torch.tensor([[3.0], [6.0]]), columns=["first", "second"])
        destination = os.path.join(tempdir, "reduced.h5")

        HDF5File.reduce(sources, destination)

        torch.testing.assert_close(
            HDF5File.open(destination, "cpu"),
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        )


def test_hdf5_file_reduce_creates_virtual_dataset():
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


def test_vds_reduction_concatenates_multiple_hdf5_files():
    with TemporaryDirectory() as tempdir:
        source1 = os.path.join(tempdir, "s1.h5")
        source2 = os.path.join(tempdir, "s2.h5")
        HDF5File.write(source1, torch.tensor([[1.0], [2.0]]))
        HDF5File.write(source2, torch.tensor([[3.0]]))
        destination = os.path.join(tempdir, "reduced.h5")
        HDF5File.reduce([source1, source2], destination)
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


def test_nested_record_vds_merge_roundtrip():
    with TemporaryDirectory() as tempdir:
        sources = []
        for c in range(3):
            path = os.path.join(tempdir, f"cand_{c}.h5")
            _record_results(path, {"energy": torch.tensor([c, c + 0.5])})
            _record_results(path, {"loss": torch.tensor([10.0 + 2 * c, 11.0 + 2 * c])})
            sources.append(path)
        merged = os.path.join(tempdir, "merged.h5")
        HDF5File.reduce(sources, merged)

        with h5py.File(merged, "r") as file:
            energy = file["energy"]
            loss = file["loss"]
            assert energy.is_virtual
            assert loss.is_virtual
            np.testing.assert_allclose(energy[:], [0, 0.5, 1, 1.5, 2, 2.5])
            np.testing.assert_allclose(loss[:], [10, 11, 12, 13, 14, 15])
        assert len(HDF5File.dependencies(merged)) == len(sources)


def test_reduce_rejects_incompatible_nested_records():
    with TemporaryDirectory() as tempdir:
        p1 = os.path.join(tempdir, "a.h5")
        p2 = os.path.join(tempdir, "b.h5")
        _record_results(p1, {"k": torch.tensor([1.0])})
        _record_results(p2, {"k": torch.tensor([2.0]), "j": torch.tensor([3.0])})
        with pytest.raises(ValueError, match="identical datasets"):
            HDF5File.reduce([p1, p2], os.path.join(tempdir, "m.h5"))