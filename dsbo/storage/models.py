from peewee import Model, AutoField, CharField, FloatField, ForeignKeyField

class WorkflowRecord(Model):
    id    = AutoField(index=True)
    graph = CharField()

class ExecutionRecord(Model):
    id       = AutoField(index=True)
    workflow = ForeignKeyField(WorkflowRecord, backref="executions", on_delete="CASCADE")
    exec_id  = CharField(index=True, unique=True)

class JobTraceRecord(Model):
    id        = AutoField(index=True)
    execution = ForeignKeyField(ExecutionRecord, backref="job_traces", on_delete="CASCADE")
    task_key  = CharField()
    name      = CharField()
    worker    = CharField()
    start     = FloatField()
    stop      = FloatField()
