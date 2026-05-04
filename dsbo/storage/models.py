from __future__ import annotations

import enum
import uuid
from typing import Optional

from sqlalchemy import Column
from sqlalchemy import Double
from sqlmodel import Field, SQLModel


def _new_id() -> str:
    return str(uuid.uuid4())


class JobStatus(str, enum.Enum):
    PENDING = "PENDING"
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


class JobTrace(SQLModel, table=True):
    id:           str       = Field(default_factory=_new_id, primary_key=True)
    execution_id: str       = Field(foreign_key="execution.id")
    task_key:     str
    name:         str
    worker:       str
    start:        float     = Field(sa_column=Column(Double))
    stop:         float     = Field(sa_column=Column(Double))
    log:          str
    status:       JobStatus


class Intermediate(SQLModel, table=True):
    id:       str   = Field(default_factory=_new_id, primary_key=True)
    tag:      str   = Field(index=True)
    step:     int
    value:    float
    walltime: float = Field(sa_column=Column(Double))
