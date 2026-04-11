from peewee import Model, AutoField, CharField, FloatField, ForeignKeyField


class WorkflowRecord(Model):
    id      = AutoField(index=True)
    exec_id = CharField(index=True, unique=True)


class JobTraceRecord(Model):
    id       = AutoField(index=True)
    workflow = ForeignKeyField(WorkflowRecord, backref="job_traces", on_delete="CASCADE")
    task_key = CharField()
    worker   = CharField()
    start    = FloatField()
    stop     = FloatField()
