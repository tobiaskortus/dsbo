import os
import h5py
import numpy as np
import pytest
import torch
import uproot
from tempfile import TemporaryDirectory
from unittest.mock import patch
import torch.autograd.forward_ad as fw

from dsbo.config import config, get_active_file_backend
from dsbo.utils.io.autograd import InMemoryFile
from dsbo.utils.io.datasets import ModuleSurrogateDataset, WorkflowSurrogateDataset
from dsbo.utils.io.hdf5 import HDF5File
from dsbo.utils.io.root import ROOTFile


@pytest.fixture(
    params=[
        pytest.param((HDF5File, ".h5"), id="hdf5"),
        pytest.param((ROOTFile, ".root"), id="root"),
    ]
)
def file_backend(request):
    backend, extension = request.param
    with config.use(file_backend=backend):
        assert get_active_file_backend() is backend
        yield backend, extension


def _record_results(backend, path: str, outputs: dict[str, torch.Tensor]) -> None:
    backend.write_named_datasets(
        path,
        {key: value.numpy() for key, value in outputs.items()},
    )


def aggregate(ds) -> tuple[torch.Tensor, torch.Tensor]:
    xs = torch.stack([ds[i][0] for i in range(len(ds))], dim=0)
    ys = torch.stack([ds[i][1] for i in range(len(ds))], dim=0)
    return xs, ys

def test_file_backend_forward_ad_tangent(file_backend):
    backend, extension = file_backend
    with fw.dual_level():
        x = fw.make_dual(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 0.0]))
        x = x * 3

        with TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, f"x{extension}")
            backend.write(file_path, x)
            x_loaded = backend.open(file_path, "cpu")
            tangent = fw.unpack_dual(x_loaded).tangent
            
            torch.testing.assert_close(x_loaded, torch.tensor([3.0, 6.0]))
            torch.testing.assert_close(tangent, torch.tensor([3.0, 0.0]))


def test_file_backend_backward_gradient(file_backend):
    backend, extension = file_backend
    backend.reset_gradient_state()
    with TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, f"x{extension}")

        primal = torch.tensor([2.0, 3.0])
        backend.write(file_path, primal)

        with torch.enable_grad():
            x = backend.open(file_path, "cpu")
            y = (x * x).sum()

            # Check that the autograd hook was registered during open()
            assert file_path in backend.handles
            
            y.backward()

        assert os.path.exists(os.path.join(tempdir, f"x_gradient{extension}"))
        grad = backend.read_backward_gradient(file_path)
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


def test_file_backend_named_columns_forward_ad_tangent(file_backend):
    backend, extension = file_backend
    backend.reset_gradient_state()
    with TemporaryDirectory() as tempdir, fw.dual_level():
        file_path = os.path.join(tempdir, f"x{extension}")
        primal = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
        tangent = torch.arange(8, 16, dtype=torch.float32).reshape(2, 2, 2)

        backend.write(file_path, fw.make_dual(primal, tangent), columns=["first", "second"])

        loaded = backend.open(file_path, "cpu", columns=["second"])

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
def test_file_backend_named_columns_accumulate_full_and_selected(file_backend, selection_first):
    backend, extension = file_backend
    backend.reset_gradient_state()
    with TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, f"x{extension}")
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        backend.write(file_path, values, columns=["first", "second"])

        def backward_selection():
            selected = backend.open(file_path, "cpu", columns=["second"])
            (selected * 2.0).sum().backward()

        def backward_full():
            backend.open(file_path, "cpu").sum().backward()

        operations = (backward_selection, backward_full) if selection_first else (backward_full, backward_selection)
        for operation in operations:
            operation()

        torch.testing.assert_close(backend.read_backward_gradient(file_path), torch.tensor([[1.0, 1.0], [3.0, 3.0]]))


def test_hdf5_file_named_columns_validate_dimension_zero():
    with TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "x.h5")

        with pytest.raises(ValueError, match="2 columns.*size 3.*dimension 0"):
            HDF5File.write(file_path, torch.ones((3, 2)), columns=["first", "second"])


def test_file_backend_named_columns_reduce(file_backend):
    backend, extension = file_backend
    with TemporaryDirectory() as tempdir, torch.no_grad():
        sources = [os.path.join(tempdir, f"source_{i}{extension}") for i in range(2)]
        backend.write(sources[0], torch.tensor([[1.0, 2.0], [4.0, 5.0]]), columns=["first", "second"])
        backend.write(sources[1], torch.tensor([[3.0], [6.0]]), columns=["first", "second"])
        destination = os.path.join(tempdir, f"reduced{extension}")

        backend.reduce(sources, destination)

        torch.testing.assert_close(
            backend.open(destination, "cpu"),
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


@pytest.mark.parametrize(
    "value",
    [
        np.array(3.5, dtype=np.float64),
        np.arange(6, dtype=np.int64).reshape(2, 3),
        np.empty((0, 2), dtype=np.float32),
    ],
)
def test_file_backend_round_trips_shape_and_dtype(file_backend, value):
    backend, extension = file_backend
    with TemporaryDirectory() as tempdir, torch.no_grad():
        file_path = os.path.join(tempdir, f"x{extension}")
        backend.write(file_path, torch.from_numpy(value))

        loaded = backend.open(file_path, "cpu")

        assert loaded.shape == value.shape
        assert loaded.numpy().dtype == value.dtype
        np.testing.assert_array_equal(loaded.numpy(), value)


def test_root_file_writes_ttree_layout():
    with TemporaryDirectory() as tempdir, fw.dual_level():
        file_path = os.path.join(tempdir, "x.root")
        primal = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
        tangent = torch.arange(8, 16, dtype=torch.float32).reshape(2, 2, 2)
        ROOTFile.write(file_path, fw.make_dual(primal, tangent), columns=["first", "second"])

        with uproot.open(file_path) as file:
            assert file["primal"].classname == "TTree"
            assert file["primal"].keys() == ["first", "second"]
            assert file["tangent"].keys() == ["first", "second"]
            assert file["primal/first"].array(library="np").shape == (2, 2)
            assert ROOTFile.METADATA_OBJECT in file


def test_root_file_reduce_creates_virtual_manifest():
    with TemporaryDirectory() as tempdir, torch.no_grad():
        sources = [os.path.join(tempdir, f"source_{index}.root") for index in range(2)]
        ROOTFile.write(sources[0], torch.tensor([[1.0, 2.0], [4.0, 5.0]]), columns=["first", "second"])
        ROOTFile.write(sources[1], torch.tensor([[6.0], [3.0]]), columns=["second", "first"])
        destination = os.path.join(tempdir, "reduced.root")

        ROOTFile.reduce(sources, destination)

        with uproot.open(destination) as file:
            assert "primal" not in file
            assert ROOTFile.METADATA_OBJECT in file
            assert ROOTFile.VDS_SOURCES_OBJECT in file
        assert ROOTFile.dependencies(destination) == sources
        with patch("dsbo.utils.io.root.uproot.concatenate", wraps=uproot.concatenate) as concatenate:
            torch.testing.assert_close(
                ROOTFile.open(destination, "cpu"),
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            )
            concatenate.assert_called_once()

        os.remove(sources[0])
        with pytest.raises(FileNotFoundError, match="missing source"):
            ROOTFile.validate(destination)
        with pytest.raises(FileNotFoundError, match="missing source"):
            ROOTFile.open(destination, "cpu")


def test_root_file_reduce_primal_tangent_and_scalars():
    with TemporaryDirectory() as tempdir:
        sources = [os.path.join(tempdir, f"source_{index}.root") for index in range(2)]
        with fw.dual_level():
            ROOTFile.write(
                sources[0],
                fw.make_dual(torch.tensor([[1.0], [2.0]]), torch.tensor([[0.0], [1.0]])),
            )
            ROOTFile.write(
                sources[1],
                fw.make_dual(torch.tensor([[3.0]]), torch.tensor([[2.0]])),
            )
        destination = os.path.join(tempdir, "reduced.root")
        ROOTFile.reduce(sources, destination)

        with fw.dual_level():
            loaded = ROOTFile.open(destination, "cpu")
            torch.testing.assert_close(loaded, torch.tensor([[1.0], [2.0], [3.0]]))
            torch.testing.assert_close(
                fw.unpack_dual(loaded).tangent,
                torch.tensor([[0.0], [1.0], [2.0]]),
            )

        scalar_sources = [os.path.join(tempdir, f"scalar_{index}.root") for index in range(2)]
        with torch.no_grad():
            ROOTFile.write(scalar_sources[0], torch.tensor(4.0))
            ROOTFile.write(scalar_sources[1], torch.tensor(5.0))
            scalar_destination = os.path.join(tempdir, "reduced_scalars.root")
            ROOTFile.reduce(scalar_sources, scalar_destination)
            torch.testing.assert_close(ROOTFile.open(scalar_destination, "cpu"), torch.tensor([4.0, 5.0]))


def test_root_file_reduce_empty_arrays():
    with TemporaryDirectory() as tempdir, torch.no_grad():
        sources = [os.path.join(tempdir, f"empty_{index}.root") for index in range(2)]
        for source in sources:
            ROOTFile.write(source, torch.empty((0, 2)))
        destination = os.path.join(tempdir, "reduced_empty.root")
        ROOTFile.reduce(sources, destination)
        assert ROOTFile.open(destination, "cpu").shape == (0, 2)

        named_sources = [os.path.join(tempdir, f"named_empty_{index}.root") for index in range(2)]
        ROOTFile.write(named_sources[0], torch.empty((2, 3, 0)), columns=["first", "second"])
        ROOTFile.write(named_sources[1], torch.empty((2, 4, 0)), columns=["second", "first"])
        named_destination = os.path.join(tempdir, "reduced_named_empty.root")
        ROOTFile.reduce(named_sources, named_destination)
        assert ROOTFile.open(named_destination, "cpu").shape == (2, 7, 0)


def test_root_file_nested_virtual_reduction():
    with TemporaryDirectory() as tempdir, torch.no_grad():
        leaves = [os.path.join(tempdir, f"leaf_{index}.root") for index in range(3)]
        for source, value in zip(leaves, (1.0, 2.0, 3.0), strict=True):
            ROOTFile.write(source, torch.tensor(value))

        first_reduction = os.path.join(tempdir, "first.root")
        ROOTFile.reduce(leaves[:2], first_reduction)
        nested_reduction = os.path.join(tempdir, "nested.root")
        ROOTFile.reduce([first_reduction, leaves[2]], nested_reduction)
        torch.testing.assert_close(ROOTFile.open(nested_reduction, "cpu"), torch.tensor([1.0, 2.0, 3.0]))

        with pytest.raises(ValueError, match="transitive dependency"):
            ROOTFile.reduce([nested_reduction], leaves[0])
        with pytest.raises(ValueError, match="transitive dependency"):
            ROOTFile.reduce([nested_reduction], first_reduction)
        hard_link = os.path.join(tempdir, "leaf_link.root")
        os.link(leaves[0], hard_link)
        with pytest.raises(ValueError, match="transitive dependency"):
            ROOTFile.reduce([nested_reduction], hard_link)

        os.remove(leaves[1])
        with pytest.raises(FileNotFoundError, match="missing source"):
            ROOTFile.reduce([nested_reduction], os.path.join(tempdir, "invalid.root"))


def test_file_backend_module_surrogate_dataset(file_backend):
    backend, extension = file_backend
    with TemporaryDirectory() as tempdir, torch.no_grad():
        data_path = os.path.join(tempdir, f"data{extension}")
        cond_path = os.path.join(tempdir, f"cond{extension}")
        backend.write(data_path, torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        backend.write(cond_path, torch.tensor([[2.0], [4.0], [8.0]]))

        with ModuleSurrogateDataset(data_path, cond_path, get_active_file_backend()) as dataset:
            assert len(dataset) == 3
            x, cond = aggregate(dataset)

        torch.testing.assert_close(x.mean(dim=0), torch.zeros(2), atol=1e-6, rtol=0)
        torch.testing.assert_close(cond.mean(dim=0), torch.zeros(1), atol=1e-6, rtol=0)


def test_file_backend_workflow_surrogate_dataset_records(file_backend):
    backend, extension = file_backend
    with TemporaryDirectory() as tempdir:
        sources = []
        for candidate in range(2):
            path = os.path.join(tempdir, f"candidate_{candidate}{extension}")
            backend.write_named_datasets(
                path,
                {"energy": np.array([candidate, candidate + 0.5])},
            )
            backend.write_named_datasets(
                path,
                {"loss": np.array([10.0 + candidate, 11.0 + candidate])},
            )
            sources.append(path)

        records_path = os.path.join(tempdir, f"records{extension}")
        backend.reduce(sources, records_path)
        with WorkflowSurrogateDataset(
            records_path,
            torch.tensor([[1.0], [2.0]]),
            get_active_file_backend(),
        ) as dataset:
            assert dataset.keys == ["energy", "loss"]
            assert dataset.n == 2
            assert len(dataset) == 4
            x, y = aggregate(dataset)

        assert x.shape == (4, 1)
        assert y.shape == (4, 2)
        torch.testing.assert_close(y.mean(dim=0), torch.zeros(2), atol=1e-6, rtol=0)


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
            _record_results(HDF5File, path, {"energy": torch.tensor([c, c + 0.5])})
            _record_results(HDF5File, path, {"loss": torch.tensor([10.0 + 2 * c, 11.0 + 2 * c])})
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
        _record_results(HDF5File, p1, {"k": torch.tensor([1.0])})
        _record_results(HDF5File, p2, {"k": torch.tensor([2.0]), "j": torch.tensor([3.0])})
        with pytest.raises(ValueError, match="identical datasets"):
            HDF5File.reduce([p1, p2], os.path.join(tempdir, "m.h5"))
