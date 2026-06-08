from __future__ import annotations

import enum
import uuid
from typing import TYPE_CHECKING, Optional, List, Dict

import time
from sqlalchemy import Double, Column, LargeBinary, JSON
from sqlmodel import Field, SQLModel

if TYPE_CHECKING:
    import numpy as np


def _new_id() -> str:
    return str(uuid.uuid4())


class JobStatus(str, enum.Enum):
    PENDING = "PENDING"
    QUEUED  = "QUEUED"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED  = "FAILED"


class Workflow(SQLModel, table=True):
    id:        str = Field(default_factory=_new_id, primary_key=True)
    graph:     str
    num_tasks: int = 0


class Batch(SQLModel, table=True):
    id:          str = Field(default_factory=_new_id, primary_key=True)
    workflow_id: str = Field(foreign_key="workflow.id")
    batch_id:    str = Field(index=True)


class Execution(SQLModel, table=True):
    id:                  str           = Field(default_factory=_new_id, primary_key=True)
    workflow_id:         str           = Field(foreign_key="workflow.id")
    batch_id:            Optional[str] = Field(default=None, foreign_key="batch.id")
    exec_id:             str           = Field(unique=True, index=True)
    status:              JobStatus     = JobStatus.PENDING
    num_jobs:            int           = 0
    num_successful_jobs: int           = 0
    start:               float         = Field(sa_column=Column(Double), default=-1.0)
    stop:                float         = Field(sa_column=Column(Double), default=-1.0)
    updated_at:          float         = Field(sa_column=Column(Double, index=True), default_factory=time.time)


class JobTrace(SQLModel, table=True):
    id:           Optional[int] = Field(default=None, primary_key=True)
    execution_id: str           = Field(foreign_key="execution.id")
    task_key:     str
    name:         str
    worker:       str       = Field(default="")
    start:        float     = Field(sa_column=Column(Double), default=-1.0)
    stop:         float     = Field(sa_column=Column(Double), default=-1.0)
    log:          str       = Field(default="")
    status:       JobStatus


class Intermediate(SQLModel, table=True):
    id:       str   = Field(default_factory=_new_id, primary_key=True)
    tag:      str   = Field(index=True)
    step:     int
    value:    float
    walltime: float = Field(sa_column=Column(Double))


class Trial(SQLModel, table=True):
    id:         str     = Field(default_factory=_new_id, primary_key=True)
    step:       int
    feasible:   bool    = Field(index=True)
    design:     bytes   = Field(sa_column=Column(LargeBinary))
    
    objectives: Dict[str, float]  = Field(sa_column=Column(JSON))
    constraints: Optional[Dict[str, float]] = Field(default=None, sa_column=Column(JSON, nullable=True))

    @property
    def design_vector(self) -> np.ndarray:
        import numpy as np
        return np.frombuffer(self.design, dtype=np.float32).copy()