from enum import Enum
from peewee import (IntegerField, Model, AutoField, CharField,
                    FloatField, ForeignKeyField, TextField)

from dsbo.storage.fields import EnumField

class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED  = "FAILED"

class WorkflowRecord(Model):
    id        = AutoField(index=True)
    graph     = CharField() #Store the workflow graph as a JSON string
    num_tasks = IntegerField(default=0)

class BatchRecord(Model):
    id       = AutoField(index=True)
    workflow = ForeignKeyField(WorkflowRecord, backref="batches", on_delete="CASCADE")
    batch_id = CharField(index=True)  # Path to the directory containing all runs in this batch

class ExecutionRecord(Model):
    id       = AutoField(index=True)
    workflow = ForeignKeyField(WorkflowRecord, backref="executions", on_delete="CASCADE")
    batch    = ForeignKeyField(BatchRecord, backref="executions", on_delete="CASCADE", null=True)
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
