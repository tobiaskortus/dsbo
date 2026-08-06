import os
from unittest.mock import patch

import networkx as nx

from dsbo.utils.io import Workdir
from dsbo.utils.io.workdir import DefaultFilenames, DefaultPaths
from dsbo.modules.utils import ResultRecorder
from dsbo.surrogates.graph_transformations import SurrogateTrainingPlan
from dsbo.surrogates.lifecycle import _reset_stale_records
from tests.surrogates.conftest import SimpleModule

_ROOT = "/virtual/root"

_RECORD_FILENAMES = ("results.h5", "results_0.h5", "results_1.h5")


def _make_plan(count=2, recorders=None):
    record_path = Workdir.register_file(DefaultPaths.TRAINING_RECORDS, DefaultFilenames.RESULTS)

    if recorders is None:
        inner0, inner1, inner_none = SimpleModule(), SimpleModule(), SimpleModule()
        inner0.i, inner1.i, inner_none.i = 0, 1, None
        recorders = [
            ResultRecorder(inner0, record_path),
            ResultRecorder(inner1, record_path),
            ResultRecorder(inner_none, record_path),
        ]

    graph = nx.DiGraph()
    graph.add_nodes_from(recorders)

    class _Train:
        def binding(self, name):
            assert name == "candidates"
            return [object() for _ in range(count)]

    return SurrogateTrainingPlan(graph, None, _Train(), record_path), record_path


def _make_plan_without_recorders(count=2):
    record_path = Workdir.register_file(DefaultPaths.TRAINING_RECORDS, DefaultFilenames.RESULTS)
    graph = nx.DiGraph()
    graph.add_nodes_from([SimpleModule(), SimpleModule()])

    class _Train:
        def binding(self, name):
            return [object() for _ in range(count)]

    return SurrogateTrainingPlan(graph, None, _Train(), record_path), record_path


def _make_plan_without_record_path(count=2):
    inner = SimpleModule()
    inner.i = 0
    graph = nx.DiGraph()
    graph.add_nodes_from([ResultRecorder(inner, Workdir.register_file(DefaultPaths.TRAINING_RECORDS, DefaultFilenames.RESULTS))])

    class _Train:
        def binding(self, name):
            return [object() for _ in range(count)]

    return SurrogateTrainingPlan(graph, None, _Train(), None)


def _run_reset(plan, *, exists=True, subdir="surrogate_training"):
    with patch("dsbo.surrogates.lifecycle.get_workdir") as get_workdir, \
         patch("dsbo.surrogates.lifecycle.get_subdir", return_value=subdir), \
         patch("os.path.exists", return_value=exists), \
         patch("os.remove") as remove:
        get_workdir.return_value.root = _ROOT
        _reset_stale_records(plan)
        return remove


class TestResetStaleRecords:
    def test_removes_exactly_the_plans_record_files(self):
        plan, record_path = _make_plan(count=2)
        base = os.path.join(_ROOT, "surrogate_training", record_path.pathname)

        expected = [
            os.path.join(base, batch, filename)
            for batch in ("batch_0", "batch_1")
            for filename in _RECORD_FILENAMES
        ]

        remove = _run_reset(plan)

        assert remove.call_count == len(expected)
        assert sorted(call.args[0] for call in remove.call_args_list) == sorted(expected)

    def test_single_candidate_reset(self):
        plan, record_path = _make_plan(count=1)
        base = os.path.join(_ROOT, "surrogate_training", record_path.pathname)
        expected = [os.path.join(base, filename) for filename in _RECORD_FILENAMES]

        remove = _run_reset(plan)

        assert sorted(call.args[0] for call in remove.call_args_list) == sorted(expected)

    def test_does_not_touch_paths_outside_record_area(self):
        plan, record_path = _make_plan(count=2)
        base = os.path.join(_ROOT, "surrogate_training", record_path.pathname)

        remove = _run_reset(plan)

        removed = [call.args[0] for call in remove.call_args_list]
        assert all(path.startswith(base + os.sep) for path in removed)
        assert not any("other" in path or path.endswith("unrelated.txt") for path in removed)

    def test_record_path_none_is_a_noop(self):
        plan = _make_plan_without_record_path(count=2)

        remove = _run_reset(plan)

        assert remove.call_count == 0

    def test_zero_candidates_resets_nothing(self):
        plan, _ = _make_plan(count=0)

        remove = _run_reset(plan)

        assert remove.call_count == 0

    def test_missing_files_are_skipped(self):
        plan, record_path = _make_plan(count=2)

        remove = _run_reset(plan, exists=False)

        assert remove.call_count == 0

    def test_missing_files_skipped_but_existing_removed(self):
        plan, record_path = _make_plan(count=2)
        base = os.path.join(_ROOT, "surrogate_training", record_path.pathname)
        kept = os.path.join(base, "batch_0", "results_1.h5")

        def _exists(path):
            return path != kept

        with patch("dsbo.surrogates.lifecycle.get_workdir") as get_workdir, \
             patch("dsbo.surrogates.lifecycle.get_subdir", return_value="surrogate_training"), \
             patch("os.path.exists", side_effect=_exists), \
             patch("os.remove") as remove:
            get_workdir.return_value.root = _ROOT
            _reset_stale_records(plan)

        removed = [call.args[0] for call in remove.call_args_list]
        assert kept not in removed
        assert len(removed) == 5

    def test_no_recorders_removes_only_base_filename(self):
        plan, record_path = _make_plan_without_recorders(count=2)
        base = os.path.join(_ROOT, "surrogate_training", record_path.pathname)

        expected = [
            os.path.join(base, batch, "results.h5")
            for batch in ("batch_0", "batch_1")
        ]

        remove = _run_reset(plan)

        assert sorted(call.args[0] for call in remove.call_args_list) == sorted(expected)

    def test_empty_subdir_anchors_at_root(self):
        plan, record_path = _make_plan(count=1)
        base = os.path.join(_ROOT, record_path.pathname)
        expected = [os.path.join(base, filename) for filename in _RECORD_FILENAMES]

        remove = _run_reset(plan, subdir="")

        assert sorted(call.args[0] for call in remove.call_args_list) == sorted(expected)

    def test_duplicate_indices_are_deduplicated(self):
        record_path = Workdir.register_file(DefaultPaths.TRAINING_RECORDS, DefaultFilenames.RESULTS)
        inner_a, inner_b = SimpleModule(), SimpleModule()
        inner_a.i, inner_b.i = 0, 0
        recorders = [
            ResultRecorder(inner_a, record_path),
            ResultRecorder(inner_b, record_path),
        ]
        plan, _ = _make_plan(count=1, recorders=recorders)
        base = os.path.join(_ROOT, "surrogate_training", record_path.pathname)

        expected = sorted([os.path.join(base, "results.h5"), os.path.join(base, "results_0.h5")])

        remove = _run_reset(plan)

        assert sorted(call.args[0] for call in remove.call_args_list) == expected
