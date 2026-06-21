from __future__ import annotations
import functools

import torch
import contextvars
import numpy as np
import networkx as nx
from pathlib import Path
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Callable, Dict, Generator, List, Mapping, Optional, TYPE_CHECKING

from dsbo.utils.functools import auto_wrap
from dsbo.storage.artifacts import LocalModelRegistry

if TYPE_CHECKING:
    from dsbo.storage.taskstream import TaskStreamPlugin
    from dsbo.storage.models import Batch, Execution, JobStatus, JobTrace, Workflow, Trial


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
        status: JobStatus,
    ) -> None:
        pass

_current_storage: contextvars.ContextVar[StudyStorageBase | None] = contextvars.ContextVar(
    "current_storage", default=None)


@contextmanager
def use_storage(storage: StudyStorageBase) -> Generator[StudyStorageBase, None, None]:
    token = _current_storage.set(storage)
    try:
        yield storage
    finally:
        _current_storage.reset(token)
    

def get_current_storage() -> Optional[StudyStorageBase]:
    return _current_storage.get()



class WorkflowStorageBase(ABC):
    def __init__(self, readonly: bool = False):
        super().__init__()
        self._readonly = readonly

    @abstractmethod
    def add_workflow(self, repr: nx.DiGraph) -> Workflow:
        pass
        
    @abstractmethod
    def add_runs(self, wf: Workflow, exec_ids: list[str], batch_id: str) -> "Batch":
        pass

    @abstractmethod
    def get_batches(self) -> List[Batch]:
        pass

    @abstractmethod
    def get_executions(self, batch_id: Optional[str] = None, updated_after: Optional[float] = None) -> List[Execution]:
        pass

    @abstractmethod
    def get_workflows(self) -> List[Workflow]:
        pass

    @abstractmethod
    def get_workflow_graphs(self, as_nx: bool = True) -> List[nx.DiGraph]:
        pass
    
    @abstractmethod
    def get_workflow_graph(self, exec_id: str, as_nx: bool = True) -> Optional[nx.DiGraph]:
        pass

    @abstractmethod
    def get_job_traces_by_batch(self, batch_id: str, after_trace_id: Optional[int] = None) -> List[JobTrace]:
        pass

    @abstractmethod
    def get_job_traces(self, exec_id: str, after_trace_id: Optional[int] = None) -> List[JobTrace]:
        pass

    def create_task_stream_plugin(self, task_map: Dict[str, str] | None = None, name_map: Dict[str, str] | None = None) -> TaskStreamPlugin:
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
        self._trial_objective_keys = set()
        self._trial_constraint_keys = set()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        def validate_objective_and_constraint_keys(func: Callable):
            @functools.wraps(func)
            def validator(self, step: int, design_vec: np.array | torch.Tensor, objectives: dict, 
                    constraints: Optional[dict] = None, feasible: bool = True):
                if len(self._trial_objective_keys) == 0:
                    self._trial_objective_keys = set(objectives.keys())
                else:
                    assert set(objectives.keys()) == self._trial_objective_keys, \
                        f"Objective keys {set(objectives.keys())} do not match existing keys {self._trial_objective_keys}."
                    
                if len(self._trial_constraint_keys) == 0 and constraints is not None:
                    self._trial_constraint_keys = set(constraints.keys())
                elif constraints is not None:
                    assert set(constraints.keys()) == self._trial_constraint_keys, \
                        f"Constraint keys {set(constraints.keys())} do not match existing keys {self._trial_constraint_keys}."
                    
                return func(self, step, design_vec, objectives, constraints, feasible)
                    
            return validator

        auto_wrap(cls, "add_trial", validate_objective_and_constraint_keys)

        
    @abstractmethod
    def add_trial(self, step: int, design_vec: np.array | torch.Tensor, objectives: dict, 
                  constraints: Optional[dict] = None, feasible: bool = True) -> Trial:
        pass

    @abstractmethod
    def get_trials(self, since_step: Optional[int] = None) -> list[Trial]:
        pass

    @abstractmethod
    def get_similar_trials(self, query_vec: np.ndarray | torch.Tensor, k: int = 10, metric: str = "l2") -> list[Trial]:
        pass


class StudyStorageBase:
    def __init__(self, db_path: Path | str):
        self._trial_storage: Optional[TrialStorageBase] = None
        self._workflow_storage: Optional[WorkflowStorageBase] = None
        self._artifact_storage: Optional[ArtifactStorageBase] = None
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
    
    @property
    def artifacts(self) -> ArtifactStorageBase:
        return self._artifact_storage

    def __enter__(self):
        self._storage_token = _current_storage.set(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        _current_storage.reset(self._storage_token)


class ArtifactStorageBase(ABC):
    def __init__(self, storage_name: str):
        self.model_registry = LocalModelRegistry(f"{storage_name}.artifacts")

    def write_to_model_registry(self, artifact: torch.nn.Module, tag: str, step: int) -> None:
        fpath = self.model_registry.save(artifact, tag, step)
        self.write_artifact_metadata(tag, step, {"artifact_type": type(artifact).__name__})

    def load_from_model_registry(self, tag: str, step: int | None = None) -> Mapping[str, torch.Tensor]:
        metadata = self.load_artifact_metadata(tag, step)
        return self.model_registry.load(tag, step)

    @abstractmethod
    def write_artifact_metadata(self, tag: str, step: int, metadata: dict) -> None:
        pass

    @abstractmethod
    def load_artifact_metadata(self, tag: str, step: int | None = None) -> Optional[dict]:
        pass