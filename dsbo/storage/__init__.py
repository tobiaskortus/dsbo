import os
import contextvars
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional
from peewee import SqliteDatabase

from dsbo.storage.taskstream import TaskStreamPlugin
from dsbo.storage.models import WorkflowRecord, JobTraceRecord


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


class StudyStorage:

    def __init__(self, db_path: Path | str):
        if os.path.exists(db_path):
            raise FileExistsError(f"Database file {db_path} already exists.")
        
        self._readonly = False
        self._db_path = str(db_path)
        self._db: Optional[SqliteDatabase] = None
        self._open_db()

    
    @classmethod
    def open(cls, db_path: Path | str) -> "StudyStorage":
        self = object.__new__(cls)
        self._db_path = str(db_path)
        self._db = None
        self._open_db()
        self._readonly = True
        return self


    def _open_db(self):
        self._db = SqliteDatabase(self._db_path, pragmas={"journal_mode": "wal"})
        self._db.connect(reuse_if_open=True)
        with self._db.bind_ctx([WorkflowRecord, JobTraceRecord]):
            self._db.create_tables([WorkflowRecord, JobTraceRecord], safe=True)

    def __enter__(self):
        self._storage_token = _current_storage.set(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        _current_storage.reset(self._storage_token)
        self._db.close()

    def create_task_stream_plugin(self, task_map: Dict[str, str] | None = None) -> TaskStreamPlugin:
        if self._readonly:
            raise RuntimeError("Cannot create TaskStreamPlugin from readonly StudyStorage.")
        
        db_path = self._db_path

        def on_transition(key: str, worker: str, start: float, stop: float, exec_id: str) -> None:
            db = SqliteDatabase(db_path, pragmas={"journal_mode": "wal"})
            db.connect()
            with db.bind_ctx([WorkflowRecord, JobTraceRecord]):
                wf, _ = WorkflowRecord.get_or_create(exec_id=exec_id)
                JobTraceRecord.create(
                    workflow=wf, task_key=key, worker=worker, start=start, stop=stop,
                )
            db.close()

        return TaskStreamPlugin(on_transition=on_transition, task_map=task_map)

    def get_workflows(self) -> List[WorkflowRecord]:
        with self._db.bind_ctx([WorkflowRecord]):
            return list(WorkflowRecord.select().order_by(WorkflowRecord.id))

    def get_job_traces(self, exec_id: str) -> List[JobTraceRecord]:
        with self._db.bind_ctx([WorkflowRecord, JobTraceRecord]):
            return list(
                JobTraceRecord
                .select()
                .join(WorkflowRecord)
                .where(WorkflowRecord.exec_id == exec_id)
                .order_by(JobTraceRecord.start)
            )
