from __future__ import annotations

import torch
import contextvars
import numpy as np
import networkx as nx
from pathlib import Path
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from dsbo.storage.models import Batch, Execution, Intermediate, JobStatus, JobTrace, Workflow, Trial


class TransitionWriter(ABC):
    @abstractmethod
    def __call__(
        self,
        key: str,
        name: str,
        worker: str,
        start: float,
        stop: float,
        exec_id: str,
        log: str,
        status: "JobStatus",
    ) -> None:
        pass

_current_storage: contextvars.ContextVar["StudyStorage | None"] = contextvars.ContextVar(
    "current_storage", default=None)


@contextmanager
def use_storage(storage: "StudyStorage") -> Generator["StudyStorage", None, None]:
    token = _current_storage.set(storage)
    try:
        yield storage
    finally:
        _current_storage.reset(token)


def get_current_storage() -> Optional["StudyStorage"]:
    return _current_storage.get()



class WorkflowStorageBase(ABC):
    def __init__(self, readonly: bool = False):
        super().__init__()
        self._readonly = readonly

    @abstractmethod
    def add_workflow(self, repr: nx.DiGraph) -> "Workflow":
        pass
        
    @abstractmethod
    def add_runs(self, wf: "Workflow", exec_ids: list[str], batch_id: str) -> "Batch":
        pass

    @abstractmethod
    def get_batches(self) -> List["Batch"]:
        pass

    @abstractmethod
    def get_executions_by_batch(self, batch_id: str) -> List["Execution"]:
        pass

    @abstractmethod
    def get_workflows(self) -> List["Workflow"]:
        pass

    @abstractmethod
    def get_workflow_graphs(self, as_nx: bool = True) -> List[nx.DiGraph]:
        pass
    
    @abstractmethod
    def get_workflow_graph(self, exec_id: str, as_nx: bool = True) -> Optional[nx.DiGraph]:
        pass

    @abstractmethod
    def get_job_traces_by_batch(self, batch_id: str) -> List["JobTrace"]:
        pass

    @abstractmethod
    def get_job_traces(self, exec_id: str) -> List["JobTrace"]:
        pass

    def create_task_stream_plugin(self, task_map: Dict[str, str] | None = None, name_map: Dict[str, str] | None = None) -> "TaskStreamPlugin":
        from dsbo.storage.taskstream import TaskStreamPlugin
        if self._readonly:
            raise RuntimeError("Cannot create TaskStreamPlugin from readonly StudyStorage.")
        return TaskStreamPlugin(writer=self._make_transition_writer(), task_map=task_map, name_map=name_map)

    @abstractmethod
    def _make_transition_writer(self) -> TransitionWriter:
        pass


class IntermediatesStorageBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def add_scalar(self, tag: str, step: int, value: float) -> None:
        pass


class TrialStorageBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def add_trial(self, step: int, design_vec: np.array | torch.Tensor, objectives: dict):
        pass

    @abstractmethod
    def get_trials(self) -> list[Trial]:
        pass

class StudyStorageBase:
    def __init__(self, db_path: Path | str):
        self._trial_storage: Optional[TrialStorageBase] = None
        self._workflow_storage: Optional[WorkflowStorageBase] = None
        self._intermediates_storage: Optional[IntermediatesStorageBase] = None
    
    @property
    def workflow(self) -> WorkflowStorageBase:
        return self._workflow_storage
    
    @property
    def intermediates(self) -> IntermediatesStorageBase:
        return self._intermediates_storage
    
    @property
    def trials(self) -> TrialStorageBase:
        return self._trial_storage

    def __enter__(self):
        self._storage_token = _current_storage.set(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        _current_storage.reset(self._storage_token)