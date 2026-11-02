import traceback
from unittest.mock import patch

import networkx as nx
import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from dsbo.modules import Module
from dsbo.storage.sqlmodel import ExecutionConflictError, StudyStorage


def test_duplicate_execution_reises_and_rolls_back(tmp_path):
    with StudyStorage.create(tmp_path / "study.db") as study:
        workflow = study.workflow.add_workflow(nx.DiGraph([(Module(), Module())]))
        study.workflow.add_runs(workflow, ["step=0"], "step=0")
        original = study.workflow.get_executions()[0]

        with pytest.raises(ExecutionConflictError) as caught:
            study.workflow.add_runs(workflow, ["step=0/new", "step=0"], "step=0")

        message = str(caught.value)
        assert "step=0" in message
        assert "already registered" in message
        assert "Workdir.create_subdir('evaluation')" in message
        formatted = "".join(traceback.format_exception(caught.value))
        assert "INSERT INTO" not in formatted
        assert "sqlalchemy.exc.IntegrityError" not in formatted
        assert caught.value.__suppress_context__
        assert study.workflow.get_batches() == []
        executions = study.workflow.get_executions()
        assert len(executions) == 1
        assert executions[0].id == original.id
        assert executions[0].status == original.status

        study.workflow.add_runs(workflow, ["step=0/evaluation"], "step=0/evaluation")
        assert len(study.workflow.get_executions()) == 2


def test_unrelated_integrity_errors(tmp_path):
    with StudyStorage.create(tmp_path / "study.db") as study:
        workflow = study.workflow.add_workflow(nx.DiGraph([(Module(), Module())]))
        error = IntegrityError("INSERT", {}, Exception("unrelated constraint failed"))
        with patch.object(Session, "commit", side_effect=error):
            with pytest.raises(IntegrityError) as caught:
                study.workflow.add_runs(workflow, ["step=0"], "step=0")
        assert caught.value is error
