import json
import os
import h5py
import numpy as np
import pytest
import torch
from tensordict import TensorDict
from tempfile import TemporaryDirectory
from unittest.mock import patch
import torch.autograd.forward_ad as fw

from dsbo.config import config, get_active_file_backend
from dsbo.utils.io.gradient import gradient_path
from dsbo.utils.io.datasets import ModuleSurrogateDataset, WorkflowSurrogateDataset
from dsbo.utils.io.storage.hdf5 import HDF5File, HDF5Writer
from dsbo.utils.io.storage.utils import to_tensordict


@pytest.fixture(
    params=[pytest.param((HDF5File, ".h5"), id="hdf5")]
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


@pytest.mark.parametrize("reversed_order", [False, True])
def test_file_backend_backward_gradient_double_open(file_backend, reversed_order):
    backend, extension = file_backend
    backend.reset_gradient_state()
    with TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, f"x{extension}")
        primal = torch.tensor([2.0, 3.0])
        backend.write(file_path, primal)

        with torch.enable_grad():
            a = backend.open(file_path, "cpu")
            b = backend.open(file_path, "cpu")
            if reversed_order:
                y = (2 * a + b).sum()
            else:
                y = (a + 2 * b).sum()
            expected = torch.tensor([3.0, 3.0])
            assert len(backend.handles[file_path]) == 2

            y.backward()

        grad = backend.read_backward_gradient(file_path)
        torch.testing.assert_close(grad, expected)
        # Both hooks removed themselves; no dangling handles remain.
        backend.reset_gradient_state()
        assert file_path not in backend.handles


def test_file_backend_backward_gradient_double_open_partial_participation(file_backend):
    backend, extension = file_backend
    backend.reset_gradient_state()
    with TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, f"x{extension}")
        primal = torch.tensor([2.0, 3.0])
        backend.write(file_path, primal)

        with torch.enable_grad():
            backend.open(file_path, "cpu")
            b = backend.open(file_path, "cpu")
            (b * 3.0).sum().backward()

        grad = backend.read_backward_gradient(file_path)
        torch.testing.assert_close(grad, torch.tensor([3.0, 3.0]))
        backend.reset_gradient_state()
        assert file_path not in backend.handles


def test_hdf5_file_named_columns():
    with TemporaryDirectory() as tempdir, torch.no_grad():
        file_path = os.path.join(tempdir, "x.h5")
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        td = TensorDict({"first": values[0], "second": values[1], "third": values[2]}, batch_size=(2,))

        HDF5File.write(file_path, td)

        with h5py.File(file_path, "r") as file:
            assert isinstance(file["primal"], h5py.Group)
            np.testing.assert_array_equal(file["primal/second"][:], values[1].numpy())
        loaded = HDF5File.open(file_path, "cpu")
        assert isinstance(loaded, TensorDict)
        for key in ("first", "second", "third"):
            torch.testing.assert_close(loaded[key], td[key])
        subset = HDF5File.open(file_path, "cpu", columns=["third", "first"])
        assert list(subset.keys()) == ["third", "first"]
        torch.testing.assert_close(subset["third"], values[2])
        torch.testing.assert_close(subset["first"], values[0])


def test_file_backend_named_columns_forward_ad_tangent(file_backend):
    backend, extension = file_backend
    backend.reset_gradient_state()
    with TemporaryDirectory() as tempdir, fw.dual_level():
        file_path = os.path.join(tempdir, f"x{extension}")
        primal = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
        tangent = torch.arange(8, 16, dtype=torch.float32).reshape(2, 2, 2)
        primal_td = TensorDict(
            {"first": fw.make_dual(primal[0], tangent[0]), "second": fw.make_dual(primal[1], tangent[1])},
            batch_size=(2,),
        )

        backend.write(file_path, primal_td)

        loaded = backend.open(file_path, "cpu", columns=["second"])
        assert isinstance(loaded, TensorDict)
        dual = fw.unpack_dual(loaded["second"])
        torch.testing.assert_close(dual.primal, primal[1])
        torch.testing.assert_close(dual.tangent, tangent[1])


def test_hdf5_file_named_columns_backward_gradient():
    HDF5File.reset_gradient_state()
    with TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "x.h5")
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        td = TensorDict({"first": values[0], "second": values[1], "third": values[2]}, batch_size=(2,))

        HDF5File.write(file_path, td)

        x = HDF5File.open(file_path, "cpu", columns=["third", "first"])
        weights = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
        ((x["third"] * weights[0]).sum() + (x["first"] * weights[1]).sum()).backward()

        gradient = HDF5File.read_backward_gradient(file_path)
        assert isinstance(gradient, TensorDict)
        assert set(gradient.keys()) == {"third", "first"}
        torch.testing.assert_close(gradient["third"], weights[0])
        torch.testing.assert_close(gradient["first"], weights[1])
        with h5py.File(gradient_path(file_path), "r") as file:
            assert list(file["gradient"].keys()) == ["third", "first"]


def test_hdf5_file_named_columns_backward_gradient_without_selection():
    HDF5File.reset_gradient_state()
    with TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "x.h5")
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        td = TensorDict({"first": values[0], "second": values[1]}, batch_size=(2,))

        HDF5File.write(file_path, td)

        x = HDF5File.open(file_path, "cpu")
        sum((value * value).sum() for value in x.values()).backward()

        gradient = HDF5File.read_backward_gradient(file_path)
        assert isinstance(gradient, TensorDict)
        torch.testing.assert_close(gradient["first"], 2.0 * values[0])
        torch.testing.assert_close(gradient["second"], 2.0 * values[1])


def test_hdf5_file_named_columns_accumulate_gradients():
    HDF5File.reset_gradient_state()
    with TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "x.h5")
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        td = TensorDict({"first": values[0], "second": values[1]}, batch_size=(2,))

        HDF5File.write(file_path, td)

        first = HDF5File.open(file_path, "cpu", columns=["first"])
        sum(value.sum() for value in (first * 2.0).values()).backward()
        second = HDF5File.open(file_path, "cpu", columns=["second"])
        sum(value.sum() for value in (second * 3.0).values()).backward()

        # The second gradient update has to leave the first one intact.
        gradient = HDF5File.read_backward_gradient(file_path)
        assert isinstance(gradient, TensorDict)
        torch.testing.assert_close(gradient["first"], torch.full((2,), 2.0))
        torch.testing.assert_close(gradient["second"], torch.full((2,), 3.0))


@pytest.mark.parametrize("selection_first", [False, True])
def test_file_backend_named_columns_accumulate_full_and_selected(file_backend, selection_first):
    backend, extension = file_backend
    backend.reset_gradient_state()
    with TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, f"x{extension}")
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        td = TensorDict({"first": values[0], "second": values[1]}, batch_size=(2,))

        backend.write(file_path, td)

        def backward_selection():
            selected = backend.open(file_path, "cpu", columns=["second"])
            sum(value.sum() for value in (selected * 2.0).values()).backward()

        def backward_full():
            loaded = backend.open(file_path, "cpu")
            sum(value.sum() for value in loaded.values()).backward()

        operations = (backward_selection, backward_full) if selection_first else (backward_full, backward_selection)
        for operation in operations:
            operation()

        gradient = backend.read_backward_gradient(file_path)
        assert isinstance(gradient, TensorDict)
        torch.testing.assert_close(gradient["first"], torch.ones(2))
        torch.testing.assert_close(gradient["second"], torch.full((2,), 3.0))


def test_tensordict_batch_validation():
    with pytest.raises(AssertionError, match="batch size"):
        to_tensordict({"first": torch.ones(3), "second": torch.ones(2)})
    with pytest.raises(AssertionError, match="batch size"):
        to_tensordict({"first": torch.ones(3), "second": torch.tensor(1.0)})
    with pytest.raises(ValueError, match="at least one key"):
        to_tensordict({})


def test_file_backend_named_columns_reduce(file_backend):
    backend, extension = file_backend
    with TemporaryDirectory() as tempdir, torch.no_grad():
        sources = [os.path.join(tempdir, f"source_{i}{extension}") for i in range(2)]
        backend.write(sources[0], TensorDict({"first": torch.tensor([1.0, 2.0]), "second": torch.tensor([4.0, 5.0])}, batch_size=(2,)))
        backend.write(sources[1], TensorDict({"first": torch.tensor([3.0]), "second": torch.tensor([6.0])}, batch_size=(1,)))
        destination = os.path.join(tempdir, f"reduced{extension}")

        backend.reduce(sources, destination)

        loaded = backend.open(destination, "cpu")
        assert isinstance(loaded, TensorDict)
        torch.testing.assert_close(loaded["first"], torch.tensor([1.0, 2.0, 3.0]))
        torch.testing.assert_close(loaded["second"], torch.tensor([4.0, 5.0, 6.0]))


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
        with h5py.File(destination, "r") as file:
            manifest = json.loads(file.attrs[HDF5Writer.VDS_SOURCES_ATTRIBUTE])
        assert manifest == [os.path.relpath(source, os.path.dirname(destination))]
        os.remove(source)

        with pytest.raises(FileNotFoundError, match="missing source"):
            HDF5File.open(destination, "cpu")


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



def test_file_backend_module_surrogate_dataset(file_backend):
    backend, extension = file_backend
    with TemporaryDirectory() as tempdir, torch.no_grad():
        data_path = os.path.join(tempdir, f"data{extension}")
        cond_path = os.path.join(tempdir, f"cond{extension}")
        backend.write(data_path, torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        backend.write(cond_path, torch.tensor([[2.0], [4.0], [8.0]]))

        with ModuleSurrogateDataset(data_path, cond_path, get_active_file_backend()) as dataset:
            assert len(dataset) == 3
            batched = dataset.__getitems__([2, 0, 2])
            individual = [dataset[index] for index in (2, 0, 2)]
            for actual, expected in zip(batched, individual, strict=True):
                torch.testing.assert_close(actual[0], expected[0])
                torch.testing.assert_close(actual[1], expected[1])
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

        def load_dataset():
            with WorkflowSurrogateDataset(
                records_path,
                torch.tensor([[1.0], [2.0]]),
                get_active_file_backend(),
            ) as dataset:
                assert dataset.keys == ["energy", "loss"]
                assert dataset.n == 2
                assert len(dataset) == 4
                np.testing.assert_allclose(dataset._records.read("energy", slice(1, 3)), [0.5, 1.0])
                assert dataset._records.partition_sizes("energy") == (2, 2)
                with patch.object(
                    dataset._records,
                    "read_many",
                    side_effect=AssertionError("normalization scanned record values"),
                ):
                    dataset.compute_stats()
                torch.testing.assert_close(dataset.mean_y, torch.tensor([0.75, 11.0]))
                batched = dataset.__getitems__([0, 1])
                individual = [dataset[0], dataset[1]]
                for actual, expected in zip(batched, individual, strict=True):
                    torch.testing.assert_close(actual[0], expected[0])
                    torch.testing.assert_close(actual[1], expected[1])
                return aggregate(dataset)

        x, y = load_dataset()

        assert x.shape == (4, 1)
        assert y.shape == (4, 2)
        torch.testing.assert_close(y.mean(dim=0), torch.zeros(2), atol=1e-6, rtol=0)


def test_workflow_surrogate_constant_large_offset_normalization(file_backend, tmp_path):
    backend, extension = file_backend
    sources = [tmp_path / f"candidate_{candidate}{extension}" for candidate in range(2)]
    offset = 1e12 + 0.1
    for source in sources:
        backend.write_named_datasets(source, {"energy": np.full(500_000, offset)})
    records_path = tmp_path / f"records{extension}"
    backend.reduce(sources, records_path)

    with WorkflowSurrogateDataset(
        records_path, torch.tensor([[1.0], [2.0]]), get_active_file_backend()
    ) as dataset:
        with patch.object(
            dataset._records,
            "read_many",
            side_effect=AssertionError("normalization scanned record values"),
        ):
            dataset.compute_stats()
        torch.testing.assert_close(dataset.mean_y, torch.tensor([offset]), rtol=0, atol=0)
        torch.testing.assert_close(dataset.std_y, torch.tensor([1e-8]), rtol=0, atol=0)
        for _, normalized in dataset.__getitems__([0, len(dataset) - 1]):
            torch.testing.assert_close(normalized, torch.zeros(1), rtol=0, atol=0)


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
        with h5py.File(merged, "r") as file:
            assert len(json.loads(file.attrs[HDF5Writer.VDS_SOURCES_ATTRIBUTE])) == len(sources)


def test_reduce_rejects_incompatible_nested_records():
    with TemporaryDirectory() as tempdir:
        p1 = os.path.join(tempdir, "a.h5")
        p2 = os.path.join(tempdir, "b.h5")
        _record_results(HDF5File, p1, {"k": torch.tensor([1.0])})
        _record_results(HDF5File, p2, {"k": torch.tensor([2.0]), "j": torch.tensor([3.0])})
        with pytest.raises(ValueError, match="identical datasets"):
            HDF5File.reduce([p1, p2], os.path.join(tempdir, "m.h5"))


def test_file_backend_workflow_surrogate_row_store(file_backend):
    backend, extension = file_backend
    with TemporaryDirectory() as tempdir:
        sources = []
        for candidate in range(2):
            path = os.path.join(tempdir, f"candidate_{candidate}{extension}")
            backend.write_row_store(
                path,
                {
                    "energy": np.array([candidate, candidate + 0.5], dtype=np.float32),
                    "loss": np.array([10.0 + candidate, 11.0 + candidate], dtype=np.float32),
                },
            )
            sources.append(path)

        if backend is HDF5File:
            with h5py.File(sources[0], "r") as file:
                assert list(file) == [HDF5File.ROW_STORE_DATASET]
                assert file[HDF5File.ROW_STORE_DATASET].shape == (2, 2)
                assert file[HDF5File.ROW_STORE_DATASET].compression is None

        records_path = os.path.join(tempdir, f"records{extension}")
        backend.reduce(sources, records_path)
        expected = torch.tensor(
            [[0.0, 10.0], [0.5, 11.0], [1.0, 11.0], [1.5, 12.0]],
            dtype=torch.float32,
        )
        with WorkflowSurrogateDataset(
            records_path, torch.tensor([[1.0], [2.0]]), get_active_file_backend()
        ) as dataset:
            dataset.compute_stats()
            torch.testing.assert_close(dataset.mean_y, torch.tensor([0.75, 11.0]))
            for i in range(len(dataset)):
                _, y = dataset[i]
                torch.testing.assert_close(
                    y * dataset.std_y + dataset.mean_y, expected[i]
                )


def test_workflow_surrogate_normalization_uses_train_rows_only(file_backend):
    """Normalization must be fit on the *training* rows only, so validation data
    never leaks into the training moments. Regression test for the prior bug
    where compute_stats() ran on the whole dataset before the split."""
    backend, extension = file_backend
    from dsbo.utils.io.datasets import design_split

    with TemporaryDirectory() as tempdir:
        sources = []
        # Train candidates have small y; val candidates have much larger y.
        ys = [1.0, 2.0, 100.0, 200.0]
        for c, y in enumerate(ys):
            path = os.path.join(tempdir, f"candidate_{c}{extension}")
            backend.write_row_store(path, {"y": np.array([y], dtype=np.float32)})
            sources.append(path)
        records_path = os.path.join(tempdir, f"records{extension}")
        backend.reduce(sources, records_path)
        design = torch.tensor([[float(c)] for c in range(len(ys))])

        with WorkflowSurrogateDataset(
            records_path, design, get_active_file_backend()
        ) as dataset:
            train_rows, val_rows = design_split(len(ys), dataset.n, val_split=0.5, seed=0)

            dataset.compute_stats(train_rows)
            train_mean_y = dataset.mean_y

            # Brute-force expected mean over exactly the training rows, keeping
            # the per-feature shape (n_features,) that _fit_stats produces.
            # Each training row is one (candidate, single feature) sample.
            expected_train = torch.as_tensor(
                [[ys[r // dataset.n]] for r in train_rows], dtype=torch.float32
            ).mean(dim=0)
            assert train_mean_y.shape == (1,)
            torch.testing.assert_close(train_mean_y, expected_train)

            # And it must NOT equal the whole-data mean (which would leak val).
            whole = torch.as_tensor(ys, dtype=torch.float32).mean()
            assert not torch.allclose(train_mean_y, whole)
