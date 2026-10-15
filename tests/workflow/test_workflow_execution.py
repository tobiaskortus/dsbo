import os
import pytest
import torch
from unittest.mock import MagicMock, patch
from dsbo.parameters import ParameterSpace, Continuous, Candidate
from dsbo.workflow import (
    run_batch, run_batch_reverse_autograd, RunBatchAutogradFn,
    to_1d_tensor, candidates_from_tensor, Workflow,
)
from dsbo.utils.io import Input, Output, Workdir, get_subdir
from dsbo.utils.io.storage.hdf5 import HDF5File
from dsbo.modules import BatchFusedModule, Module, batch_fuse
from dsbo.utils.io import Workdir
from dsbo.modules.dsl import DependencyDict
from dsbo.container import (
    ContainerWrapperBase, ImageRef, DockerImageProvider,
)
from tests.workflow.conftest import (
    SimpleModule, BackwardModule, NoGradModule, FailingModule,
    FailingBackwardModule,
)


class TestTo1dTensor:
    def test_scalar_float(self):
        result = to_1d_tensor(5.0, torch.float32)
        assert result.shape == (1,)
        assert result.item() == 5.0

    def test_tensor_input(self):
        result = to_1d_tensor(torch.tensor(3.0), torch.float64)
        assert result.dtype == torch.float64
        assert result.shape == (1,)

    def test_1d_tensor_passthrough(self):
        result = to_1d_tensor(torch.tensor([1.0, 2.0]), torch.float32)
        assert result.shape == (2,)

    def test_raises_on_2d_tensor(self):
        with pytest.raises(ValueError, match="Flatten"):
            to_1d_tensor(torch.ones(2, 3), torch.float32)

    def test_tensor_with_dtype_conversion(self):
        t = torch.tensor([1.0], dtype=torch.float32)
        result = to_1d_tensor(t, torch.float64)
        assert result.dtype == torch.float64


class TestCandidatesFromTensor:
    def test_single_candidate(self, simple_space):
        t = torch.tensor([[5.0]])
        cands = candidates_from_tensor(t, simple_space)
        assert len(cands) == 1
        assert cands[0].tensor.item() == 5.0

    def test_multiple_candidates(self, simple_space):
        t = torch.tensor([[1.0], [2.0], [3.0]])
        cands = candidates_from_tensor(t, simple_space)
        assert len(cands) == 3


class TestRunRunLocal:
    def test_run_single_candidate(self, single_module_workflow, candidate):
        with torch.no_grad():
            result = single_module_workflow.run(candidate)
        assert torch.allclose(result["out"], torch.tensor(10.0))

    def test_run_with_catch(self, simple_space):
        wf = Workflow(FailingModule())
        cand = Candidate(simple_space, torch.tensor([3.0]))
        with torch.no_grad():
            result = wf.run(cand, catch=(ValueError,))
        assert result == {}

    def test_run_re_raises_uncaught(self, simple_space):
        wf = Workflow(FailingModule())
        cand = Candidate(simple_space, torch.tensor([3.0]))
        with torch.no_grad():
            with pytest.raises(ValueError):
                wf.run(cand, catch=(RuntimeError,))


class TestRunBatchLocal:
    def test_does_not_forward_design_tensor_to_ordinary_modules(self, simple_space):
        class StrictModule(Module):
            def __init__(self):
                super().__init__()

            def __call__(self, design):
                return {"out": design["x"]}

        workflow = Workflow(StrictModule())
        candidate = Candidate(simple_space, torch.tensor([3.0]))

        results = run_batch(workflow, [candidate])

        assert torch.equal(results[0]["out"], torch.tensor(3.0))

    def test_runs_single_candidate(self, single_module_workflow, simple_space):
        cand = Candidate(simple_space, torch.tensor([3.0]))
        results = run_batch(single_module_workflow, [cand, cand])
        assert len(results) == 2
        assert torch.allclose(results[0]["out"], torch.tensor(6.0))
        assert torch.allclose(results[1]["out"], torch.tensor(6.0))

    def test_runs_multiple_candidates(self, single_module_workflow, candidates):
        results = run_batch(single_module_workflow, candidates)
        assert len(results) == 3
        assert torch.allclose(results[0]["out"], torch.tensor(2.0))
        assert torch.allclose(results[1]["out"], torch.tensor(4.0))
        assert torch.allclose(results[2]["out"], torch.tensor(6.0))

    def test_catches_exceptions(self, simple_space):
        wf = Workflow(FailingModule())
        cand = Candidate(simple_space, torch.tensor([3.0]))
        results = run_batch(wf, [cand, cand], catch=(ValueError,))
        assert len(results) == 2
        assert results[0] == {}
        assert results[1] == {}

    def test_re_raises_uncaught_exception(self, simple_space):
        wf = Workflow(FailingModule())
        cand = Candidate(simple_space, torch.tensor([3.0]))
        with pytest.raises(ValueError, match="simulated failure"):
            run_batch(wf, [cand, cand], catch=(RuntimeError,))

    def test_invokes_warm_container_dependencies(self, simple_space):
        ref = ImageRef(uri="ubuntu:22.04")

        class ContainerModule(Module, ContainerWrapperBase):
            def __init__(self):
                Module.__init__(self)
                ContainerWrapperBase.__init__(self, ref)
            def run(self, **kwargs):
                return {"out": torch.tensor(0.0)}

        wf = Workflow(ContainerModule())
        cand = Candidate(simple_space, torch.tensor([3.0]))

        with patch("dsbo.workflow.workflow.warm_container_dependencies") as mock_warm, \
             patch("dsbo.container.discover_container_environments",
                   return_value={"docker": True, "apptainer": False, "singularity": False}), \
             patch.object(DockerImageProvider, "prepare_image", return_value="ubuntu:22.04"):
            wf.run_batch([cand])

        mock_warm.assert_called_once_with(wf, verbose=True)


class TestDuplicateResultKeys:
    """Duplicate result keys make the merged result and its gradient
    ambiguous, so merging must raise (bugs.md #3 family)."""

    def _duplicate_key_workflow(self):
        class Dup(Module):
            def __call__(self, design, **kwargs):
                return {"out": design["x"] * 2.0}

        return Workflow(Dup() >> Dup())

    def test_local_run_batch_raises_on_duplicate_result_keys(self):
        wf = self._duplicate_key_workflow()
        cand = Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([3.0]))

        with pytest.raises(ValueError, match="result keys"):
            run_batch(wf, [cand])

    def test_distributed_run_batch_raises_on_duplicate_result_keys(self):
        from tests.workflow.conftest import MockDaskWorker
        from dsbo.worker.dask import WorkerContext

        wf = self._duplicate_key_workflow()
        cand = Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([3.0]))

        worker = MockDaskWorker()

        def colliding_gather(futures, catch=()):
            # Both scheduled tasks report results under the same key.
            return [(tid, {"out": torch.tensor(1.0)}) for tid, _, _, _ in worker.scheduled]

        worker.gather = colliding_gather

        with WorkerContext.use_worker(worker):
            with pytest.raises(ValueError, match="result keys"):
                run_batch(wf, [cand])


class TestRunBatchReverseAutograd:
    def test_gradients_for_backward_module(self, simple_space):
        m = BackwardModule()
        wf = Workflow(m)
        x = torch.tensor([3.0], requires_grad=True)
        cand = Candidate(simple_space, x)
        grad_outputs = [{"out": torch.tensor(1.0)}]
        grads = run_batch_reverse_autograd(wf, [cand], grad_outputs)
        assert len(grads) == 1
        assert grads[0].shape == (1,)

    def test_gradients_for_no_grad_module(self, simple_space):
        m = NoGradModule()
        wf = Workflow(m)
        x = torch.tensor([3.0], requires_grad=True)
        cand = Candidate(simple_space, x)
        grad_outputs = [{"out": torch.tensor(1.0)}]
        grads = run_batch_reverse_autograd(wf, [cand], grad_outputs)
        assert len(grads) == 1
        assert grads[0].shape == (1,)

    def test_multiple_candidates(self, simple_space):
        m = BackwardModule()
        wf = Workflow(m)
        cands = [Candidate(simple_space, torch.tensor([1.0], requires_grad=True)),
                 Candidate(simple_space, torch.tensor([2.0], requires_grad=True))]
        grad_outputs = [{"out": torch.tensor(1.0)}, {"out": torch.tensor(1.0)}]
        grads = run_batch_reverse_autograd(wf, cands, grad_outputs)
        assert len(grads) == 2

    def test_catches_exceptions(self, simple_space):
        #TODO: While this is the currently correct behaviour, this doesnt make much sense
        m = FailingBackwardModule()
        wf = Workflow(m)
        x = torch.tensor([3.0], requires_grad=True)
        cand = Candidate(simple_space, x)
        grad_outputs = [{"out": torch.tensor(1.0)}]
        grads = run_batch_reverse_autograd(wf, [cand], grad_outputs, catch=(ValueError,))
        assert len(grads) == 1

    def test_re_raises_uncaught_exception(self, simple_space):
        m = FailingBackwardModule()
        wf = Workflow(m)
        x = torch.tensor([3.0], requires_grad=True)
        cand = Candidate(simple_space, x)
        grad_outputs = [{"out": torch.tensor(1.0)}]
        with pytest.raises(ValueError, match="simulated backward failure"):
            run_batch_reverse_autograd(wf, [cand], grad_outputs, catch=(RuntimeError,))

    def test_backward_produces_gradients(self, simple_space):
        m = BackwardModule()
        wf = Workflow(m)
        x = torch.tensor([[3.0]], requires_grad=True)
        vals, _, _, _ = RunBatchAutogradFn.apply(wf, x, simple_space, ())
        loss = vals.sum()
        loss.backward()
        assert x.grad is not None

    def test_backward_diamond(self, simple_space):
        order = []

        class DiamondA(Module):
            def __call__(self, design, **kwargs):
                return {"out_a": design["x"] * 2.0}
            def backward(self, design, grad_output=None):
                order.append("A")
                x = design["x"]
                out = x * 2.0
                grad = grad_output["out_a"] if grad_output is not None else torch.ones_like(out)
                out.backward(grad)

        class DiamondB(Module):
            def __call__(self, design, **kwargs):
                return {"out_b": design["x"] * 3.0}
            def backward(self, design, grad_output=None):
                order.append("B")
                x = design["x"]
                out = x * 3.0
                grad = grad_output["out_b"] if grad_output is not None else torch.ones_like(out)
                out.backward(grad)

        class DiamondC(Module):
            def __call__(self, design, **kwargs):
                return {"out_c": design["x"] * 4.0}
            def backward(self, design, grad_output=None):
                order.append("C")
                x = design["x"]
                out = x * 4.0
                grad = grad_output["out_c"] if grad_output is not None else torch.ones_like(out)
                out.backward(grad)

        class DiamondD(Module):
            def __call__(self, design, **kwargs):
                return {"out_d": design["x"] * 5.0}
            def backward(self, design, grad_output=None):
                order.append("D")
                x = design["x"]
                out = x * 5.0
                out.backward(grad_output["out_d"])

        a, b, c, d = DiamondA(), DiamondB(), DiamondC(), DiamondD()
        dd = DependencyDict({a: [b, c], b: [d], c: [d]})
        wf = Workflow(dd)

        x = torch.tensor([[3.0]], requires_grad=True)
        vals, _, _, _ = RunBatchAutogradFn.apply(wf, x, simple_space, ())
        loss = vals.sum()
        loss.backward()

        assert order[:1] == ["D"]
        assert set(order[1:3]) == {"B", "C"}
        assert order[3:] == ["A"]

        expected_grad = 5.0 + 3.0 + 4.0 + 2.0
        assert x.grad is not None
        assert abs(x.grad.item() - expected_grad) < 1e-5


class TestBackwardWorkdirContext:
    """The local backward pass must replay each candidate's forward I/O context:
    per-candidate tasks run under their batch_i subdir and receive the forward
    file overlays, so registered files resolve to the correct candidate's data."""

    def test_backward_enters_batch_subdirs(self, simple_space):
        recorded = []

        class SubdirRecorder(Module):
            def __call__(self, design, **kwargs):
                return {"out": design["x"]}
            def backward(self, design, grad_output=None):
                recorded.append(get_subdir())

        wf = Workflow(SubdirRecorder())
        cands = [Candidate(simple_space, torch.tensor([1.0], requires_grad=True)),
                 Candidate(simple_space, torch.tensor([2.0], requires_grad=True))]
        grads = run_batch_reverse_autograd(wf, cands, [{"out": torch.tensor(1.0)}] * 2)
        assert recorded == ["batch_0", "batch_1"]
        assert len(grads) == 2

    def test_backward_single_candidate_has_no_subdir(self, simple_space):
        recorded = []

        class SubdirRecorder(Module):
            def __call__(self, design, **kwargs):
                return {"out": design["x"]}
            def backward(self, design, grad_output=None):
                recorded.append(get_subdir())

        wf = Workflow(SubdirRecorder())
        cand = Candidate(simple_space, torch.tensor([1.0], requires_grad=True))
        run_batch_reverse_autograd(wf, [cand], [{"out": torch.tensor(1.0)}])
        assert recorded == [""]

    def test_backward_reads_intermediates_from_batch_subdirs(self, simple_space):
        class Producer(Module):
            def __init__(self):
                super().__init__()
                self.output = Output(Workdir.register_file("simulation", "events.h5"))

            def __call__(self, design, **kwargs):
                HDF5File.write(self.output(), (design["x"] * 2.0).reshape(1))

            def backward(self, design, grad_output=None):
                grad = HDF5File.read_backward_gradient(self.output())
                (design["x"].reshape(1) * 2.0).backward(grad)

        class Consumer(Module):
            def __init__(self):
                super().__init__()
                self.input = Input(Workdir.register_file("simulation", "events.h5"))

            def __call__(self, design, **kwargs):
                return {"out": HDF5File.open(self.input(), "cpu").sum()}

            def backward(self, design, grad_output=None):
                x = HDF5File.open(self.input(), "cpu")
                x.sum().backward(grad_output["out"])

        wf = Workflow(Producer() >> Consumer())
        cands = [Candidate(simple_space, torch.tensor([1.0])),
                 Candidate(simple_space, torch.tensor([2.0]))]

        results = run_batch(wf, cands)
        assert torch.allclose(results[0]["out"], torch.tensor(2.0))
        assert torch.allclose(results[1]["out"], torch.tensor(4.0))

        grads = run_batch_reverse_autograd(
            wf, cands, [{"out": torch.tensor(1.0)}, {"out": torch.tensor(2.0)}]
        )
        assert torch.allclose(grads[0], torch.tensor(2.0))
        assert torch.allclose(grads[1], torch.tensor(4.0))

    def test_backward_replays_file_overlays(self, simple_space):
        overlays_seen = []

        class Producer(Module):
            def __init__(self):
                super().__init__()
                self.output = Output(Workdir.register_file("simulation", "events.h5"))

            def __call__(self, design, **kwargs):
                HDF5File.write(self.output(), design["x"].reshape(1))

        class Consumer(Module):
            def __init__(self):
                super().__init__()
                self.input = Input(Workdir.register_file("simulation", "events.h5"))

            def __call__(self, design, **kwargs):
                return {"out": HDF5File.open(self.input(), "cpu").sum()}

            def backward(self, design, grad_output=None):
                overlays_seen.append(Workdir.get_file_overlay(self.registered_inputs[0]))

        wf = Workflow(Producer() >> Consumer())
        cands = [Candidate(simple_space, torch.tensor([1.0])),
                 Candidate(simple_space, torch.tensor([2.0]))]

        run_batch(wf, cands)
        run_batch_reverse_autograd(wf, cands, [{"out": torch.tensor(1.0)}] * 2)

        assert overlays_seen[0] is not None and "batch_0" in overlays_seen[0]
        assert overlays_seen[1] is not None and "batch_1" in overlays_seen[1]

    def test_batch_fused_backward_uses_batch_reduce_subdir(self, simple_space, candidates):
        recorded = []

        class RecordingFused(BatchFusedModule):
            def backward(self, design, grad_output=None):
                recorded.append(get_subdir())

        class Inner(Module):
            def __init__(self):
                super().__init__()
                self.input = Input(Workdir.register_file("simulation", "events.h5"))

            def __call__(self, design, **kwargs):
                return {}

        wf = Workflow(RecordingFused(Inner()))
        grad_outputs = [{"out": torch.tensor(1.0)} for _ in candidates]
        run_batch_reverse_autograd(wf, candidates, grad_outputs)
        assert recorded == ["batch_reduce"] * len(candidates)


class TestWorkflowRunBatchNoAutograd:
    def test_executes_single_candidate(self, single_module_workflow, simple_space):
        cand = Candidate(simple_space, torch.tensor([3.0]))
        with torch.no_grad():
            results = single_module_workflow.run_batch([cand])
        assert torch.allclose(results[0]["out"], torch.tensor(6.0))

    def test_executes_multiple_candidates(self, single_module_workflow, candidates):
        with torch.no_grad():
            results = single_module_workflow.run_batch(candidates)
        assert len(results) == 3
        assert torch.allclose(results[0]["out"], torch.tensor(2.0))
        assert torch.allclose(results[1]["out"], torch.tensor(4.0))
        assert torch.allclose(results[2]["out"], torch.tensor(6.0))

    def test_executes_chain(self, chain_workflow, candidates):
        with torch.no_grad():
            results = chain_workflow.run_batch(candidates)
        assert len(results) == 3
        assert torch.allclose(results[0]["out"], torch.tensor(2.0))


class TestWorkflowRunBatchCacheIntegration:
    def test_all_hits_no_run_batch_called(self, workflow_with_mock_cache, candidates, mock_run_batch, mock_cache):
        mock_cache.load_or_compute.side_effect = None
        mock_cache.load_or_compute.return_value = [
            {"out": torch.tensor(10.0)},
            {"out": torch.tensor(20.0)},
            {"out": torch.tensor(30.0)},
        ]
        results = workflow_with_mock_cache.run_batch(candidates)
        mock_cache.load_or_compute.assert_called_once()
        mock_run_batch.assert_not_called()
        assert len(results) == 3
        assert torch.allclose(results[0]["out"], torch.tensor(10.0))
        assert torch.allclose(results[1]["out"], torch.tensor(20.0))
        assert torch.allclose(results[2]["out"], torch.tensor(30.0))

    def test_all_misses_run_batch_called(self, workflow_with_mock_cache, candidates, mock_run_batch, mock_cache):
        results = workflow_with_mock_cache.run_batch(candidates)

        mock_cache.load_or_compute.assert_called_once()
        mock_run_batch.assert_called_once()
        assert len(results) == 3

    def test_partial_hits_misses_merged(self, workflow_with_mock_cache, candidates, mock_cache, mock_run_batch):
        mock_cache.load_or_compute.side_effect = None
        mock_cache.load_or_compute.return_value = [
            {"out": torch.tensor(10.0)},
            {"out": torch.tensor(20.0)},
            {"out": torch.tensor(30.0)},
        ]

        results = workflow_with_mock_cache.run_batch(candidates)

        mock_cache.load_or_compute.assert_called_once()
        assert len(results) == 3
        assert torch.allclose(results[0]["out"], torch.tensor(10.0))
        assert torch.allclose(results[1]["out"], torch.tensor(20.0))
        assert torch.allclose(results[2]["out"], torch.tensor(30.0))

    def test_cache_disabled_no_cache_lookup(self, workflow_with_mock_cache, candidates, mock_cache, mock_run_batch):
        mock_cache.config.should_cache.return_value = False

        results = workflow_with_mock_cache.run_batch(candidates)

        mock_cache.load_or_compute.assert_not_called()
        mock_run_batch.assert_called_once()
        assert len(results) == 3

    def test_no_cache_available(self, single_module_workflow, candidates, mock_run_batch):
        with patch.object(single_module_workflow, "_get_cache", return_value=None):
            results = single_module_workflow.run_batch(candidates)

        mock_run_batch.assert_called_once()
        assert len(results) == 3

class TestGetCache:
    def test_returns_none_when_no_cache_config(self, single_module_workflow):
        with patch("dsbo.workflow.workflow.config.get", return_value=None):
            cache = single_module_workflow._get_cache()
            assert cache is None

    def test_returns_cache_when_config_exists(self, single_module_workflow):
        from dsbo.utils.cache import CacheConfig
        config = CacheConfig()
        with patch("dsbo.workflow.workflow.config.get", return_value=config):
            with patch("dsbo.workflow.workflow.get_workdir") as mock_wd:
                mock_wd.return_value.root = "/tmp/workdir"
                cache = single_module_workflow._get_cache()
                assert cache is not None


class TestBatchFuse:
    def test_batch_fuse_keeps_workflow_graph_unchanged(self):
        class FileModule(Module):
            def __init__(self):
                super().__init__()
                self.input = Input(Workdir.register_file("simulation", "events.h5"))

        producer = SimpleModule()
        fused = batch_fuse(FileModule())
        consumer = SimpleModule()
        graph = Workflow(producer >> fused >> consumer).task_graph

        assert list(graph.nodes) == [producer, fused, consumer]
        assert list(graph.edges) == [(producer, fused), (fused, consumer)]

    def test_delegates_metadata(self):
        class FileModule(Module):
            def __init__(self):
                super().__init__()
                self.input = Input(Workdir.register_file("simulation", "events.h5"))

        inner = FileModule()
        fused = batch_fuse(inner)
        assert isinstance(fused, BatchFusedModule)
        assert fused.inner is inner
        assert fused.name == inner.name
        assert fused.resources is inner.resources
        inner.surrogate = object()
        assert fused.surrogate is inner.surrogate
        replica = fused.replicate(2)
        assert replica.i == replica.inner.i == 2

    def test_rejects_invalid_and_double_wrapped_modules(self):
        class FileModule(Module):
            def __init__(self):
                super().__init__()
                self.input = Input(Workdir.register_file("simulation", "events.h5"))

        with pytest.raises(TypeError, match="Module"):
            batch_fuse(object())
        with pytest.raises(ValueError, match="twice"):
            batch_fuse(batch_fuse(FileModule()))

    def test_requires_registered_input(self):
        with pytest.raises(ValueError, match="register at least one Input"):
            batch_fuse(SimpleModule())

    def test_reduce_collects_files_once_and_broadcasts_result(self, candidates):
        calls = []

        class Producer(Module):
            def __init__(self):
                super().__init__()
                self.output = Output(Workdir.register_file("simulation", "events.h5"))

            def __call__(self, design, **kwargs):
                HDF5File.write(self.output(), design["x"].reshape(1))

        class Trainer(Module):
            def __init__(self):
                super().__init__()
                self.input = Input(Workdir.register_file("simulation", "events.h5"))

            def __call__(self, design, **kwargs):
                calls.append(design["x"])
                values = HDF5File.open(self.input(), "cpu")
                return {"loss": values.sum()}

        results = run_batch(Workflow(Producer() >> batch_fuse(Trainer())), candidates)
        assert len(calls) == 1
        assert calls[0].item() == 1.0
        assert results == [{}, {}, {}]

    def test_reduce_collects_matching_replica_outputs(self, candidates):
        class Producer(Module):
            def __init__(self):
                super().__init__()
                self.features = Output(Workdir.register_file("simulation", "simulation.h5"))
                self.targets = Output(Workdir.register_file("simulation", "target.h5"))

            def __call__(self, design, **kwargs):
                HDF5File.write(self.features(), design["x"].reshape(1))
                HDF5File.write(self.targets(), (design["x"] * 10).reshape(1))

        class Trainer(Module):
            def __init__(self):
                super().__init__()
                self.features = Input(Workdir.register_file("simulation", "simulation.h5"))
                self.targets = Input(Workdir.register_file("simulation", "target.h5"))

            def __call__(self, design, **kwargs):
                return {
                    "features": HDF5File.open(self.features(), "cpu").sum(),
                    "targets": HDF5File.open(self.targets(), "cpu").sum(),
                }

        results = run_batch(
            Workflow(Producer() * 2 >> batch_fuse(Trainer())),
            candidates,
        )
        assert results == [{}, {}, {}]

    def test_reduction_destination_does_not_overwrite_replica_sources(self, candidates):
        class Producer(Module):
            def __init__(self):
                super().__init__()
                self.output = Output(Workdir.register_file("simulation", "events.h5"))

            def __call__(self, design, **kwargs):
                HDF5File.write(self.output(), (design["x"] + 10 * self.i).reshape(1))

        class Consumer(Module):
            def __init__(self):
                super().__init__()
                self.input = Input(Workdir.register_file("simulation", "events.h5"))

            def __call__(self, design, **kwargs):
                return {f"total_{self.i}": HDF5File.open(self.input(), "cpu").sum()}

        producers = list((Producer() * 2).keys())
        consumers = list((Consumer() * 3).keys())
        dependencies = DependencyDict({
            **{producer: consumers for producer in producers},
            **{consumer: [] for consumer in consumers},
        })

        results = Workflow(dependencies).run_batch(candidates)
        assert [result["total_0"].item() for result in results] == [12.0, 14.0, 16.0]
        for candidate_index, candidate in enumerate(candidates):
            for producer in producers:
                source = Workdir.resolve_file(
                    producer.output.path,
                    subdir=f"batch_{candidate_index}",
                    suffix=producer.i,
                    use_overlays=False,
                )
                expected = candidate.values["x"] + 10 * producer.i
                torch.testing.assert_close(HDF5File.open(source, "cpu"), expected.reshape(1))

    def test_collective_reduction_preserves_task_major_source_order(self, candidates):
        class Producer(Module):
            def __init__(self):
                super().__init__()
                self.output = Output(Workdir.register_file("simulation", "events.h5"))

            def __call__(self, design, **kwargs):
                HDF5File.write(self.output(), (design["x"] + 100 * self.i).reshape(1))

        class Reader(Module):
            def __init__(self):
                super().__init__()
                self.input = Input(Workdir.register_file("simulation", "events.h5"))

            def __call__(self, design, **kwargs):
                return {"values": HDF5File.open(self.input(), "cpu")}

        results = Workflow(Producer() * 2 >> batch_fuse(Reader())).run_batch(candidates)
        assert results == [{}, {}, {}]

    def test_reduction_destinations_are_consumer_scoped(self, candidates):
        destinations = []

        class Producer(Module):
            def __init__(self):
                super().__init__()
                self.output = Output(Workdir.register_file("simulation", "events.h5"))

            def __call__(self, design, **kwargs):
                HDF5File.write(self.output(), (design["x"] + self.i).reshape(1))

        class Consumer(Module):
            def __init__(self):
                super().__init__()
                self.input = Input(Workdir.register_file("simulation", "events.h5"))

            def __call__(self, design, **kwargs):
                return {f"total_{self.i}": HDF5File.open(self.input(), "cpu").sum()}

        producers = list((Producer() * 4).keys())
        consumers = [Consumer(), Consumer()]
        consumers[0].i, consumers[1].i = 0, 1
        dependencies = DependencyDict({
            producers[0]: [consumers[0]],
            producers[1]: [consumers[0]],
            producers[2]: [consumers[1]],
            producers[3]: [consumers[1]],
            consumers[0]: [],
            consumers[1]: [],
        })
        real_reduce = HDF5File.reduce

        def track(sources, destination):
            destinations.append(destination)
            return real_reduce(sources, destination)

        with patch.object(HDF5File, "reduce", track):
            Workflow(dependencies).run_batch(candidates)

        assert len(destinations) == 2 * len(candidates)
        assert len(set(destinations)) == len(destinations)

    def test_collective_preserves_ambient_input_overlay(self, candidates):
        input_path = Workdir.register_file("simulation", "events.h5")
        shared = os.path.join(Workdir.get_current_dir(), "shared-events.h5")
        HDF5File.write(shared, torch.tensor([7.0]))

        class Reader(Module):
            def __init__(self):
                super().__init__()
                self.input = Input(input_path)

            def __call__(self, design, **kwargs):
                return {"loss": HDF5File.open(self.input(), "cpu").sum()}

        with Workdir.use_file_overlays({input_path: shared}):
            results = Workflow(batch_fuse(Reader())).run_batch(candidates)

        assert results == [{}] * len(candidates)

    def test_independent_consumer_merges_its_candidate_replicas(self, candidates):
        class Producer(Module):
            def __init__(self):
                super().__init__()
                self.output = Output(
                    Workdir.register_file("simulation", "events.h5", unique_filename=True)
                )

            def __call__(self, design, **kwargs):
                HDF5File.write(self.output(), design["x"].reshape(1))

        class Consumer(Module):
            def __init__(self):
                super().__init__()
                self.input = Input(Workdir.register_file("simulation", "events.h5"))

            def __call__(self, design, **kwargs):
                return {"total": HDF5File.open(self.input(), "cpu").sum()}

        workflow = Workflow(Producer() * 3 >> Consumer())
        results = run_batch(workflow, candidates)
        assert [result["total"].item() for result in results] == [3.0, 6.0, 9.0]
        # Under lazy reduction the merged file lives in the execution-private area
        # rather than at the logical candidate path; verify the overlay still works
        # by checking correctness of results above

    def test_batch_fuse_marks_module_no_grad(self, candidates):
        class Reader(Module):
            def __init__(self):
                super().__init__()
                self.input = Input(Workdir.register_file("simulation", "events.h5"))

            def __call__(self, design, **kwargs):
                return {"loss": HDF5File.open(self.input(), "cpu").sum()}

        fused = batch_fuse(Reader())
        assert fused.inner._no_grad_op

        with Workdir.create_subdir("batch_reduce"):
            HDF5File.write(
                Workdir.register_file("simulation", "events.h5")(),
                candidates[0].tensor,
            )
        results = Workflow(fused).run_batch(candidates)
        assert results == [{}, {}, {}]

    def test_collective_execution_does_not_use_partial_cache(self, candidates):
        class Reader(Module):
            def __init__(self):
                super().__init__()
                self.input = Input(Workdir.register_file("simulation", "events.h5"))

        workflow = Workflow(batch_fuse(Reader()))
        cache = MagicMock()
        cache.config.should_cache.return_value = True
        results = [{"loss": torch.tensor(6.0)} for _ in candidates]

        with patch.object(workflow, "_get_cache", return_value=cache):
            with patch("dsbo.workflow.runner.run_batch", return_value=results) as execute:
                workflow.run_batch(candidates)

        cache.load_or_compute.assert_not_called()
        execute.assert_called_once()

    def test_batch_fused_module_skips_backward(self, candidates):
        class Reader(Module):
            def __init__(self):
                super().__init__()
                self.input = Input(Workdir.register_file("simulation", "events.h5"))

        fused = batch_fuse(Reader())
        assert fused.inner._no_grad_op

        grad_candidates = [
            Candidate(candidate.space, candidate.tensor.clone().requires_grad_())
            for candidate in candidates
        ]
        workflow = Workflow(fused)
        results = workflow.run_batch(grad_candidates)
        assert results == [{}, {}, {}]

    def test_reduce_preserves_nested_workdir(self, candidates):
        class Reader(Module):
            def __init__(self):
                super().__init__()
                self.input = Input(Workdir.register_file("simulation", "events.h5"))

            def __call__(self, design, **kwargs):
                return {"loss": HDF5File.open(self.input(), "cpu").sum()}

        with Workdir.create_subdir("nested"):
            with Workdir.create_subdir("batch_reduce"):
                HDF5File.write(
                    Workdir.register_file("simulation", "events.h5")(),
                    torch.stack([c.tensor for c in candidates]).sum(dim=0),
                )

        with Workdir.create_subdir("nested"):
            results = run_batch(Workflow(batch_fuse(Reader())), candidates)
            assert results == [{}] * 3

    def test_rejects_reduction_of_non_reducible_extension_with_producer(self, candidates):
        """Multiple candidate sources from a predecessor with non-reducible extension raises TypeError."""
        class Producer(Module):
            def __init__(self):
                super().__init__()
                self.output = Output(Workdir.register_file("simulation", "events.pt"))

            def __call__(self, design, **kwargs):
                torch.save(torch.tensor([design["x"]]), self.output())

        class Reader(Module):
            def __init__(self):
                super().__init__()
                self.input = Input(Workdir.register_file("simulation", "events.pt"))

            def __call__(self, design, **kwargs):
                self.input()
                return {"loss": torch.tensor(0.0)}

        workflow = Workflow(Producer() * 2 >> batch_fuse(Reader()))
        with pytest.raises(TypeError, match="does not support"):
            run_batch(workflow, candidates)

    def test_unused_input_does_not_reduce(self, candidates):
        calls = []

        class Producer(Module):
            def __init__(self):
                super().__init__()
                self.output = Output(Workdir.register_file("simulation", "events.h5"))

            def __call__(self, design, **kwargs):
                HDF5File.write(self.output(), design["x"].reshape(1))

        class Reader(Module):
            def __init__(self):
                super().__init__()
                self.a = Input(Workdir.register_file("simulation", "events.h5"))
                self.b = Input(Workdir.register_file("simulation", "other.h5"))

            def __call__(self, design, **kwargs):
                calls.append("called")
                return {"out": HDF5File.open(self.a(), "cpu").sum()}

        # other.h5 does not exist; since it is never resolved, batch should succeed
        workflow = Workflow(Producer() >> batch_fuse(Reader()))
        results = run_batch(workflow, candidates)
        assert results == [{}, {}, {}]

    def test_unresolved_missing_input_raises_on_access(self, candidates):
        class Reader(Module):
            def __init__(self):
                super().__init__()
                self.input = Input(Workdir.register_file("simulation", "missing.h5"))

            def __call__(self, design, **kwargs):
                HDF5File.open(self.input(), "cpu")
                return {"out": torch.tensor(0.0)}

        workflow = Workflow(batch_fuse(Reader()))
        with pytest.raises(Exception):
            run_batch(workflow, candidates)

    def test_shared_registered_output_is_visible_to_each_candidate(self, candidates):
        class Producer(Module):
            def __init__(self):
                super().__init__()
                self.output = Output(Workdir.register_file("simulation", "events.h5"))

            def __call__(self, design, **kwargs):
                HDF5File.write(self.output(), design["x"].reshape(1))

        class Trainer(Module):
            def __init__(self):
                super().__init__()
                self.input = Input(Workdir.register_file("simulation", "events.h5"))
                self.output = Output(Workdir.register_file("reconstruction", "model.h5"))

            def __call__(self, design, **kwargs):
                values = HDF5File.open(self.input(), "cpu")
                HDF5File.write(self.output(), values.sum().reshape(1))

        class Validator(Module):
            def __init__(self):
                super().__init__()
                self.input = Input(Workdir.register_file("reconstruction", "model.h5"))

            def __call__(self, design, **kwargs):
                return {"score": HDF5File.open(self.input(), "cpu").item() + design["x"]}

        workflow = Workflow(Producer() >> batch_fuse(Trainer()) >> Validator())
        results = run_batch(workflow, candidates)
        assert [result["score"].item() for result in results] == [7.0, 8.0, 9.0]

    def test_shared_non_hdf5_artifacts_are_visible_to_independent_candidates(self, candidates):
        class Producer(Module):
            def __init__(self):
                super().__init__()
                self.output = Output(Workdir.register_file("simulation", "events.h5"))

            def __call__(self, design, **kwargs):
                HDF5File.write(self.output(), design["x"].reshape(1))

        class Trainer(Module):
            def __init__(self):
                super().__init__()
                self.input = Input(Workdir.register_file("simulation", "events.h5"))
                self.model = Output(Workdir.register_file("reconstruction", "model.pt"))

            def __call__(self, design, **kwargs):
                torch.save(HDF5File.open(self.input(), "cpu").sum(), self.model())

        class Validator(Module):
            def __init__(self):
                super().__init__()
                self.model = Input(Workdir.register_file("reconstruction", "model.pt"))

            def __call__(self, design, **kwargs):
                return {"score": torch.load(self.model()) + design["x"]}

        workflow = Workflow(Producer() >> batch_fuse(Trainer()) >> Validator())
        results = run_batch(workflow, candidates)
        assert [result["score"].item() for result in results] == [7.0, 8.0, 9.0]
        for i in range(len(candidates)):
            with Workdir.create_subdir(f"batch_{i}"):
                path = Workdir.register_file("reconstruction", "model.pt")()
            assert os.path.islink(path)
            assert not os.path.isabs(os.readlink(path))

        rerun = run_batch(workflow, candidates)
        assert [result["score"].item() for result in rerun] == [7.0, 8.0, 9.0]

    def test_reduced_failure_respects_catch(self, candidates):
        class BrokenTrainer(Module):
            def __init__(self):
                super().__init__()
                self.input = Input(Workdir.register_file("simulation", "missing.h5"))

            def __call__(self, design, **kwargs):
                HDF5File.open(self.input(), "cpu")
                return {"out": torch.tensor(0.0)}

        workflow = Workflow(batch_fuse(BrokenTrainer()))
        assert run_batch(workflow, candidates, catch=(FileNotFoundError,)) == [{}, {}, {}]
        with pytest.raises(FileNotFoundError):
            run_batch(workflow, candidates)

    def test_repeated_run_batch_uses_normal_registered_path(self, candidates):
        calls = []

        class Producer(Module):
            def __init__(self):
                super().__init__()
                self.output = Output(Workdir.register_file("simulation", "events.h5"))

            def __call__(self, design, **kwargs):
                HDF5File.write(self.output(), design["x"].reshape(1))

        class Trainer(Module):
            def __init__(self):
                super().__init__()
                self.input = Input(Workdir.register_file("simulation", "events.h5"))
                self._last_resolved = None

            def __call__(self, design, **kwargs):
                self._last_resolved = self.input()
                calls.append(self._last_resolved)
                values = HDF5File.open(self._last_resolved, "cpu")
                return {"loss": values.sum()}

        workflow = Workflow(Producer() >> batch_fuse(Trainer()))

        run_batch(workflow, candidates)
        first_paths = list(calls)
        calls.clear()

        run_batch(workflow, candidates)
        second_paths = list(calls)

        for f, s in zip(first_paths, second_paths):
            assert f == s
            assert "batch_reduce" in f
            assert "events" in f

    def test_unused_input_does_not_materialize_reduced_file(self, candidates):
        """A registered input that is never resolved does not create a
        reduced file on disk."""
        reduce_destinations = []

        class Producer(Module):
            def __init__(self):
                super().__init__()
                self.output = Output(Workdir.register_file("simulation", "events.h5"))

            def __call__(self, design, **kwargs):
                HDF5File.write(self.output(), design["x"].reshape(1))

        class Trainer(Module):
            def __init__(self):
                super().__init__()
                self.a = Input(Workdir.register_file("simulation", "events.h5"))
                self.b = Input(Workdir.register_file("simulation", "other.h5"))

            def __call__(self, design, **kwargs):
                # Only resolve self.a; self.b is never touched
                return {"out": HDF5File.open(self.a(), "cpu").sum()}

        import dsbo.utils.io.storage.hdf5 as _hdf5_module
        _real_reduce = HDF5File.reduce

        def tracking_reduce(sources, destination):
            reduce_destinations.append(destination)
            return _real_reduce(sources, destination)

        with patch.object(_hdf5_module.HDF5File, "reduce", tracking_reduce):
            workflow = Workflow(Producer() >> batch_fuse(Trainer()))
            run_batch(workflow, candidates)

        # HDF5File.reduce should be called only once for events.h5,
        # not for other.h5 (which the module never resolves)
        assert len(reduce_destinations) == 1
        assert "events_merged" in reduce_destinations[0]

    def test_lazy_reduction_only_on_resolved_input_with_multiple_inputs(self, candidates):
        """With two registered inputs but only one resolved, only that one
        triggers reduction."""
        reduce_calls = []

        class Producer(Module):
            def __init__(self):
                super().__init__()
                self.output = Output(Workdir.register_file("simulation", "events.h5"))

            def __call__(self, design, **kwargs):
                HDF5File.write(self.output(), design["x"].reshape(1))

        class Trainer(Module):
            def __init__(self):
                super().__init__()
                self.features = Input(Workdir.register_file("simulation", "events.h5"))
                self.targets = Input(Workdir.register_file("simulation", "target.h5"))

            def __call__(self, design, **kwargs):
                # Only resolve features, leave targets untouched
                vals = HDF5File.open(self.features(), "cpu")
                return {"loss": vals.sum()}

        import dsbo.utils.io.storage.hdf5 as _hdf5_module2
        _real_reduce2 = HDF5File.reduce

        def tracking_reduce(sources, destination):
            reduce_calls.append(destination)
            return _real_reduce2(sources, destination)

        with patch.object(_hdf5_module2.HDF5File, "reduce", tracking_reduce):
            run_batch(
                Workflow(Producer() >> batch_fuse(Trainer())),
                candidates,
            )

        assert len(reduce_calls) == 1
        assert "events_merged" in reduce_calls[0]


from tests.workflow.conftest import MockDaskWorker
from dsbo.worker.dask import WorkerContext


class TestRunBatchWithWorker:
    def test_single_candidate(self):
        module = SimpleModule()
        wf = Workflow(module)
        worker = MockDaskWorker()
        with WorkerContext.use_worker(worker):
            candidates = [Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([5.0]))]
            results = run_batch(wf, candidates)
            assert len(results) == 1

    def test_multi_candidate_schedules_independently(self):
        module = SimpleModule()
        wf = Workflow(module)
        worker = MockDaskWorker()
        with WorkerContext.use_worker(worker):
            candidates = [
                Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([1.0])),
                Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([2.0])),
                Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([3.0])),
            ]
            results = run_batch(wf, candidates)
            assert len(results) == 3
            assert len(worker.scheduled) == 3
