from enum import Enum
from peewee import (IntegerField, Model, AutoField, CharField,
                    FloatField, ForeignKeyField, TextField)

from dsbo.storage.fields import EnumField

class JobStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED  = "FAILED"

class WorkflowRecord(Model):
    id        = AutoField(index=True)
    graph     = CharField() #Store the workflow graph as a JSON string
    num_tasks = IntegerField(default=0)

class ExecutionRecord(Model):
    id       = AutoField(index=True)
    workflow = ForeignKeyField(WorkflowRecord, backref="executions", on_delete="CASCADE")
    exec_id  = CharField(index=True, unique=True)
    status              = EnumField(JobStatus, default=JobStatus.PENDING)
    num_jobs            = IntegerField(default=0)
    num_successful_jobs = IntegerField(default=0)

class JobTraceRecord(Model):
    id        = AutoField(index=True)
    execution = ForeignKeyField(ExecutionRecord, backref="job_traces", on_delete="CASCADE")
    task_key  = CharField()
    name      = CharField()
    worker    = CharField()
    start     = FloatField()
    stop      = FloatField()
    log       = TextField()
    status    = EnumField(JobStatus)
