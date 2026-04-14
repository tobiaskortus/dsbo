import json
import os
import contextvars
import networkx as nx
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional
from peewee import SqliteDatabase

from dsbo.storage.models import WorkflowRecord, ExecutionRecord, JobTraceRecord


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
        with self._db.bind_ctx([WorkflowRecord, ExecutionRecord, JobTraceRecord]):
            self._db.create_tables([WorkflowRecord, ExecutionRecord, JobTraceRecord], safe=True)


    def __enter__(self):
        self._storage_token = _current_storage.set(self)
        return self


    def __exit__(self, exc_type, exc, tb):
        _current_storage.reset(self._storage_token)
        self._db.close()


    def save_workflow(self, repr: nx.DiGraph) -> WorkflowRecord:
        if self._readonly:
            raise RuntimeError("Cannot save workflow to readonly StudyStorage.")

        node_index = {n: i for i, n in enumerate(repr.nodes())}
        G_serializable = nx.relabel_nodes(repr, node_index)
        for node, i in node_index.items():
            G_serializable.nodes[i]['name'] = node.name
            G_serializable.nodes[i]['type'] = type(node).__name__
            if node.resources is not None:
                G_serializable.nodes[i]['resources'] = node.resources._asdict()

        graph_json = json.dumps(nx.node_link_data(G_serializable))
        with self._db.bind_ctx([WorkflowRecord]):
            return WorkflowRecord.create(graph=graph_json)


    def register_runs(self, wf_record: WorkflowRecord, exec_ids: list[str]) -> None:
        if self._readonly:
            raise RuntimeError("Cannot register runs to readonly StudyStorage.")

        with self._db.bind_ctx([ExecutionRecord]):
            for exec_id in exec_ids:
                ExecutionRecord.create(workflow=wf_record, exec_id=exec_id)

    def create_task_stream_plugin(self, task_map: Dict[str, str] | None = None, name_map: Dict[str, str] | None = None) -> "TaskStreamPlugin":
        from dsbo.storage.taskstream import TaskStreamPlugin
        if self._readonly:
            raise RuntimeError("Cannot create TaskStreamPlugin from readonly StudyStorage.")

        db_path = self._db_path

        def on_transition(key: str, name: str, worker: str, start: float, stop: float, exec_id: str, log: str) -> None:
            db = SqliteDatabase(db_path, pragmas={"journal_mode": "wal"})
            db.connect()
            with db.bind_ctx([ExecutionRecord, JobTraceRecord]):
                execution = ExecutionRecord.get(ExecutionRecord.exec_id == exec_id)
                JobTraceRecord.create(
                    execution=execution, task_key=key, name=name, worker=worker, start=start, stop=stop, log=log,
                )
            db.close()
        return TaskStreamPlugin(on_transition=on_transition, task_map=task_map, name_map=name_map)


    def get_workflows(self) -> List[ExecutionRecord]:
        with self._db.bind_ctx([WorkflowRecord, ExecutionRecord]):
            return list(ExecutionRecord.select().order_by(ExecutionRecord.id))


    def get_workflow_graphs(self, as_nx: bool = True) -> List[nx.DiGraph]:
        with self._db.bind_ctx([WorkflowRecord]):
            records = WorkflowRecord.select().order_by(WorkflowRecord.id)
            if as_nx:
                graphs = [nx.node_link_graph(json.loads(record.graph)) for record in records]
            else:
                graphs = [json.loads(record.graph) for record in records]
            return graphs
    

    def get_workflow_graph(self, exec_id: str, as_nx: bool = True) -> Optional[nx.DiGraph]:
        with self._db.bind_ctx([WorkflowRecord, ExecutionRecord]):
            execution = ExecutionRecord.get_or_none(ExecutionRecord.exec_id == exec_id)
            if execution is None:
                raise ValueError(f"No workflow found for exec_id {exec_id}")
            record = WorkflowRecord.get_by_id(execution.workflow_id)
            
            graph_data = json.loads(record.graph)
            graph = nx.node_link_graph(graph_data) if as_nx else graph_data
            return graph


    def get_job_traces(self, exec_id: str) -> List[JobTraceRecord]:
        with self._db.bind_ctx([ExecutionRecord, JobTraceRecord]):
            return list(
                JobTraceRecord
                .select()
                .join(ExecutionRecord)
                .where(ExecutionRecord.exec_id == exec_id)
                .order_by(JobTraceRecord.start)
            )
