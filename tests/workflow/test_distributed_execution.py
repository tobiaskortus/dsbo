import pytest
import os
import tempfile
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import UUID

from dsbo.storage.models import JobStatus
from dsbo.storage.sqlmodel import StudyStorage

from dsbo.parameters import ParameterSpace, Continuous, Candidate
from dsbo.modules import Module, batch_fuse
from dsbo.workflow import (
    run_batch, run_batch_reverse_autograd, Workflow,
)
from dsbo.worker.dask import WorkerContext
from dsbo.utils.io import Workdir
from tests.workflow.conftest import (
    MockDaskWorker, LocalDaskWorker, make_mock_worker,
    SimpleModule, BackwardModule, NoGradModule, FailingModule,
    FailingBackwardModule, DataProducer, FileTrainer, DataConsumer,
)
from dsbo.modules.fused import BatchFusedModule


class KeyedModule(Module):
    """SimpleModule variant with a configurable result key."""
    def __init__(self, key):
        super().__init__()
        self.key = key
    def __call__(self, design, **kwargs):
        return {self.key: design["x"] * 2.0}


class TestDistributedRunBatch:
    def test_correct_delayed_dependencies(self):
        m1 = KeyedModule("out_1")
        m2 = KeyedModule("out_2")
        m3 = KeyedModule("out_3")
        wf = Workflow(m1 >> m2 >> m3)           # chain: m1→m2→m3
        cands = [
            Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([1.0])),
            Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([2.0])),
        ]
        worker = MockDaskWorker()

        # Record the (dependencies, priority, delayed_obj) tuple each time schedule is called
        schedule_calls: list[tuple[Module, list, int, MagicMock]] = []

        original_schedule = worker.schedule
        def tracking_schedule(module, design, dependencies=(), design_tensor=None,
                              priority=None, file_overlays=None):
            tid, delayed = original_schedule(
                module, design, dependencies, design_tensor=design_tensor,
                priority=priority, file_overlays=file_overlays,
            )
            schedule_calls.append((module, list(dependencies), priority, delayed))
            return tid, delayed
        worker.schedule = tracking_schedule

        with WorkerContext.use_worker(worker):
            results = run_batch(wf, cands)

        assert len(results) == 2
        # 2 candidates × 3 modules = 6 scheduled tasks
        assert len(schedule_calls) == 6

        # Tasks are ordered by workflow depth, then candidate.
        assert schedule_calls[0][2] == 0                     # m1 at depth 0
        assert schedule_calls[0][1] == []                    # m1 has no dependencies
        assert schedule_calls[1][2] == 0                     # candidate 1 m1
        assert schedule_calls[1][1] == []

        assert schedule_calls[2][2] == 1                     # candidate 0 m2
        assert len(schedule_calls[2][1]) == 1
        assert schedule_calls[2][1][0] is schedule_calls[0][3]
        assert schedule_calls[3][2] == 1                     # candidate 1 m2
        assert schedule_calls[3][1][0] is schedule_calls[1][3]

        assert schedule_calls[4][2] == 2                     # candidate 0 m3
        assert len(schedule_calls[4][1]) == 1
        assert schedule_calls[4][1][0] is schedule_calls[2][3]
        assert schedule_calls[5][2] == 2                     # candidate 1 m3
        assert len(schedule_calls[5][1]) == 1
        assert schedule_calls[5][1][0] is schedule_calls[3][3]

    def test_batch_fuse_depends_on_all_candidate_producers(self):
        workflow = Workflow(DataProducer() * 2 >> batch_fuse(DataConsumer()))
        candidates = [
            Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([1.0])),
            Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([2.0])),
        ]
        worker = MockDaskWorker()

        schedule_calls: list[tuple[Module, list]] = []
        original_schedule = worker.schedule
        def tracking_schedule(module, design, dependencies=(), design_tensor=None,
                              priority=None, file_overlays=None):
            tid, delayed = original_schedule(
                module, design, dependencies, design_tensor=design_tensor,
                priority=priority, file_overlays=file_overlays,
            )
            schedule_calls.append((module, list(dependencies)))
            return tid, delayed
        
        worker.schedule = tracking_schedule

        with WorkerContext.use_worker(worker):
            run_batch(workflow, candidates)

        _, dependencies = schedule_calls[-1]
        assert len(dependencies) == 2 * len(candidates)

    def test_catches_exceptions_in_distributed(self):
        wf = Workflow(SimpleModule())
        cand = Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([3.0]))

        worker = MockDaskWorker()
        worker.gather = MagicMock(side_effect=RuntimeError("computation failed"))

        with WorkerContext.use_worker(worker):
            with pytest.raises(RuntimeError):
                run_batch(wf, [cand], catch=(ValueError,))

    def test_registers_storage_plugin(self):
        wf = Workflow(SimpleModule())
        cand = Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([3.0]))

        mock_storage = MagicMock()
        mock_storage.workflow.add_workflow.return_value = MagicMock()
        mock_storage.workflow.create_task_stream_plugin.return_value = "plugin"

        worker = MockDaskWorker()

        with WorkerContext.use_worker(worker):
            with patch("dsbo.workflow.runner.get_current_storage", return_value=mock_storage):
                run_batch(wf, [cand])

        mock_storage.workflow.add_workflow.assert_called_once()
        mock_storage.workflow.add_runs.assert_called_once()
        assert mock_storage.workflow.add_runs.call_args.kwargs["num_jobs"] == 1
        mock_storage.workflow.create_task_stream_plugin.assert_called_once()
        worker.client.register_plugin.assert_called_once_with("plugin", name="dsbo_task_stream")

    def test_batch_fused_module_warns_on_non_empty_result(self):

        class RetModule(Module):
            def __call__(self, design, **kwargs):
                return {"out": torch.tensor(2.0)}

        fused = BatchFusedModule(RetModule())
        with pytest.warns(UserWarning, match="returned a non-empty dict"):
            fused({"x": torch.tensor([1.0])})
        result = fused({"x": torch.tensor([1.0])})
        assert result == {}


class TestTransitionWriterCompletion:
    def test_execution_completes_when_job_count_matches_scheduled_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.sqlite"
            storage = StudyStorage.create(path)
            try:
                workflow = Workflow(SimpleModule() >> SimpleModule())
                workflow_record = storage.workflow.add_workflow(workflow.task_graph)
                exec_id = "backward-run"
                storage.workflow.add_runs(
                    workflow_record,
                    [exec_id],
                    "batch",
                    num_jobs=1,
                )

                writer = storage.workflow._make_transition_writer()
                writer(
                    "task-key",
                    "task-name",
                    "worker",
                    1.0,
                    2.0,
                    exec_id,
                    "",
                    JobStatus.SUCCESS,
                )

                execution = next(
                    execution
                    for execution in storage.workflow.get_executions()
                    if execution.exec_id == exec_id
                )
                assert execution.num_jobs == 1
                assert execution.num_successful_jobs == 1
                assert execution.status == JobStatus.SUCCESS

                traces = storage.workflow.get_job_traces(exec_id)
                assert len(traces) == 1
                assert traces[0].status == JobStatus.SUCCESS
            finally:
                storage.close()


class TestDistributedRunBatchReverseAutograd:
    def test_schedules_backward_tasks_in_reverse_order(self):
        scheduled_order = []

        class GradTracker(Module):
            def __call__(self, design, **kwargs):
                return {"out": design["x"] * 2.0}
            def backward(self, design, grad_output=None):
                pass

        a = GradTracker()
        a._alias = "A"
        b = GradTracker()
        b._alias = "B"
        c = GradTracker()
        c._alias = "C"

        wf = Workflow(a >> (b | c))
        cand = Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([3.0]))
        grad_outputs = [{"out": torch.tensor(1.0)}]

        worker = MockDaskWorker()
        def mock_schedule_bwd(module, candidate, grad_output=None,
                              dependencies=(), priority=None, file_overlays=None,
                              gradient_overlays=None, pass_id=None, task_id=None):
            scheduled_order.append(module.name)
            tid = UUID(int=len(scheduled_order), version=4)
            delayed = MagicMock()
            delayed.key = f"mock-bwd-{module.name}"
            return tid, delayed

        worker.schedule_backward = mock_schedule_bwd

        with WorkerContext.use_worker(worker):
            grads = run_batch_reverse_autograd(wf, [cand], grad_outputs)

        assert len(grads) == 1
        # Reverse topological: B and C (sinks) first, then A
        assert scheduled_order[:2] == ["B", "C"] or scheduled_order[:2] == ["C", "B"]
        assert scheduled_order[2:] == ["A"]

    def test_skips_no_grad_tasks(self):
        wf = Workflow(NoGradModule())
        cand = Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([3.0]))
        grad_outputs = [{"out": torch.tensor(1.0)}]

        worker = MockDaskWorker()

        with WorkerContext.use_worker(worker):
            grads = run_batch_reverse_autograd(wf, [cand], grad_outputs)

        assert len(worker.backward_scheduled) == 0
        assert len(grads) == 1
        assert torch.allclose(grads[0], torch.zeros(1))

    def test_skips_no_grad_tasks_in_mixed_dag(self):
        a = NoGradModule(); a._alias = "A"
        b = BackwardModule(); b._alias = "B"
        c = BackwardModule(); c._alias = "C"
        d = BackwardModule(); d._alias = "D"
        wf = Workflow(a >> (b | c) >> d)
        cand = Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([3.0]))
        grad_outputs = [{"out": torch.tensor(1.0)}]

        worker = MockDaskWorker()

        with WorkerContext.use_worker(worker):
            grads = run_batch_reverse_autograd(wf, [cand], grad_outputs)

        scheduled_aliases = {module._alias for _, _, module, _ in worker.backward_scheduled}
        assert "A" not in scheduled_aliases
        assert scheduled_aliases == {"B", "C", "D"}

    def test_handles_none_gradients_from_failed_task(self):
        wf = Workflow(BackwardModule())
        cands = [
            Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([1.0])),
            Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([2.0])),
        ]
        grad_outputs = [{"out": torch.tensor(1.0)}, {"out": torch.tensor(1.0)}]

        tid_a = UUID(int=1, version=4)
        tid_b = UUID(int=2, version=4)

        worker = MockDaskWorker()
        worker._gather_results = [
            (tid_a, torch.tensor([2.0])),
            (tid_b, None),
        ]

        with WorkerContext.use_worker(worker):
            grads = run_batch_reverse_autograd(wf, cands, grad_outputs)

        assert len(grads) == 2

    def test_registers_backward_storage_plugin(self):
        wf = Workflow(BackwardModule())
        cand = Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([3.0]))
        grad_outputs = [{"out": torch.tensor(1.0)}]

        mock_storage = MagicMock()
        mock_storage.workflow.add_workflow.return_value = MagicMock()
        mock_storage.workflow.create_task_stream_plugin.return_value = "bwd-plugin"

        worker = MockDaskWorker()

        with WorkerContext.use_worker(worker):
            with patch("dsbo.workflow.runner.get_current_storage", return_value=mock_storage):
                run_batch_reverse_autograd(wf, [cand], grad_outputs)

        mock_storage.workflow.add_workflow.assert_called_once()
        mock_storage.workflow.add_runs.assert_called_once()
        assert mock_storage.workflow.add_runs.call_args.kwargs["num_jobs"] == 1
        mock_storage.workflow.create_task_stream_plugin.assert_called_once()
        worker.client.register_plugin.assert_called_once_with("bwd-plugin",
                                                              name="dsbo_task_stream_bwd")

    def test_registers_only_grad_jobs_for_backward(self):
        a = NoGradModule(); a._alias = "A"
        b = BackwardModule(); b._alias = "B"
        c = BackwardModule(); c._alias = "C"
        d = BackwardModule(); d._alias = "D"
        wf = Workflow(a >> (b | c) >> d)
        cand = Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([3.0]))
        grad_outputs = [{"out": torch.tensor(1.0)}]

        mock_storage = MagicMock()
        mock_storage.workflow.add_workflow.return_value = MagicMock()
        mock_storage.workflow.create_task_stream_plugin.return_value = "bwd-plugin"

        worker = MockDaskWorker()
        with WorkerContext.use_worker(worker):
            with patch("dsbo.workflow.runner.get_current_storage", return_value=mock_storage):
                run_batch_reverse_autograd(wf, [cand], grad_outputs)

        assert wf.task_graph.number_of_nodes() == 4
        assert mock_storage.workflow.add_runs.call_args.kwargs["num_jobs"] == 3




@pytest.mark.integration
class TestDistributedRunBatchRealCluster:
    def test_run_batch_single_candidate(self, local_worker):
        wf = Workflow(SimpleModule())
        cand = Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([3.0]))

        with WorkerContext.use_worker(local_worker):
            results = run_batch(wf, [cand])

        assert len(results) == 1
        assert torch.allclose(results[0]["out"], torch.tensor(6.0))

    def test_run_batch_multiple_candidates(self, local_worker):
        wf = Workflow(SimpleModule())
        cands = [
            Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([1.0])),
            Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([2.0])),
            Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([3.0])),
        ]

        with WorkerContext.use_worker(local_worker):
            results = run_batch(wf, cands)

        assert len(results) == 3
        assert torch.allclose(results[0]["out"], torch.tensor(2.0))
        assert torch.allclose(results[1]["out"], torch.tensor(4.0))
        assert torch.allclose(results[2]["out"], torch.tensor(6.0))

    def test_run_batch_chain(self, local_worker):
        m1 = SimpleModule()
        m2 = KeyedModule("out_2")
        wf = Workflow(m1 >> m2)
        cand = Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([3.0]))

        with WorkerContext.use_worker(local_worker):
            results = run_batch(wf, [cand])

        assert len(results) == 1
        # Each module operates on raw candidate design: 3*2=6
        assert torch.allclose(results[0]["out"], torch.tensor(6.0))
        assert torch.allclose(results[0]["out_2"], torch.tensor(6.0))

    def test_run_batch_catches_exceptions(self, local_worker):
        wf = Workflow(FailingModule())
        cand = Candidate(ParameterSpace(x=Continuous(0, 10)), torch.tensor([3.0]))

        with WorkerContext.use_worker(local_worker):
            results = run_batch(wf, [cand], catch=(ValueError,))

        assert len(results) == 1
        assert results[0] == {}

    def test_run_batch_reduced_files_symlinks(self, local_worker, candidates):
        workflow = Workflow(DataProducer() >> batch_fuse(FileTrainer()))
        with WorkerContext.use_worker(local_worker):
            results = run_batch(workflow, candidates)
        for i in range(len(candidates)):
            link = os.path.join(Workdir.get_current_dir(), f"batch_{i}", "reconstruction", "model.pt")
            assert os.path.islink(link)

    def test_run_batch_reduced_files_result_warning(self, local_worker, candidates):
        workflow = Workflow(DataProducer() >> batch_fuse(FileTrainer()))
        with WorkerContext.use_worker(local_worker):
            results = run_batch(workflow, candidates)
        assert results == [{} for _ in candidates]


@pytest.mark.integration
class TestDistributedRunBatchReverseAutogradRealCluster:
    def test_backward_single_candidate(self, local_worker):
        wf = Workflow(BackwardModule())
        x = torch.tensor([3.0], requires_grad=True)
        cand = Candidate(ParameterSpace(x=Continuous(0, 10)), x)
        grad_outputs = [{"out": torch.tensor(1.0)}]

        with WorkerContext.use_worker(local_worker):
            grads = run_batch_reverse_autograd(wf, [cand], grad_outputs)

        assert len(grads) == 1
        assert grads[0].shape == (1,)

    def test_backward_multiple_candidates(self, local_worker):
        wf = Workflow(BackwardModule())
        cands = [
            Candidate(ParameterSpace(x=Continuous(0, 10)),
                      torch.tensor([1.0], requires_grad=True)),
            Candidate(ParameterSpace(x=Continuous(0, 10)),
                      torch.tensor([2.0], requires_grad=True)),
        ]
        grad_outputs = [{"out": torch.tensor(1.0)}, {"out": torch.tensor(1.0)}]

        with WorkerContext.use_worker(local_worker):
            grads = run_batch_reverse_autograd(wf, cands, grad_outputs)

        assert len(grads) == 2

    def test_backward_no_grad_module_skipped(self, local_worker):
        wf = Workflow(NoGradModule())
        x = torch.tensor([3.0], requires_grad=True)
        cand = Candidate(ParameterSpace(x=Continuous(0, 10)), x)
        grad_outputs = [{"out": torch.tensor(1.0)}]

        with WorkerContext.use_worker(local_worker):
            grads = run_batch_reverse_autograd(wf, [cand], grad_outputs)

        assert len(grads) == 1
        assert torch.allclose(grads[0], torch.zeros(1))
