from __future__ import annotations

import time
import json
import dataclasses
from pathlib import Path
from typing import  List, Optional

import networkx as nx
from sqlalchemy import Engine
from sqlalchemy.pool import NullPool
from sqlmodel import Session, SQLModel, create_engine, select

from dsbo.storage import IntermediatesStorageBase, StudyStorageBase, TransitionWriter, WorkflowStorageBase
from dsbo.storage.models import Batch, Execution, Intermediate, JobStatus, JobTrace, Workflow


@dataclasses.dataclass
class SQLModelTransitionWriter(TransitionWriter):
    url: str

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
        engine = create_engine(self.url, poolclass=NullPool)
        with Session(engine) as session:
            execution = session.exec(
                select(Execution).where(Execution.exec_id == exec_id)
            ).one()
            wf = session.get(Workflow, execution.workflow_id)

            session.add(
                JobTrace(
                    execution_id=execution.id,
                    task_key=key,
                    name=name,
                    worker=worker,
                    start=start,
                    stop=stop,
                    log=log,
                    status=status,
                )
            )

            execution.num_jobs += 1
            if status == JobStatus.SUCCESS:
                execution.num_successful_jobs += 1

            if status == JobStatus.FAILED:
                execution.status = JobStatus.FAILED
            elif execution.status == JobStatus.PENDING:
                execution.status = JobStatus.RUNNING

            if execution.num_jobs >= wf.num_tasks and execution.status != JobStatus.FAILED:
                execution.status = JobStatus.SUCCESS

            session.add(execution)
            session.commit()


class WorkflowStorage(WorkflowStorageBase):
    def __init__(self, engine: Engine, readonly: bool = False):
        super().__init__(readonly=readonly)
        self._engine = engine


    def _make_transition_writer(self) -> SQLModelTransitionWriter:
        return SQLModelTransitionWriter(url=str(self._engine.url))


    def add_workflow(self, repr: nx.DiGraph) -> Workflow:
        node_index = {n: i for i, n in enumerate(repr.nodes())}
        G_s = nx.relabel_nodes(repr, node_index)
        for node, i in node_index.items():
            G_s.nodes[i]["name"] = node.name
            G_s.nodes[i]["type"] = type(node).__name__
            if node.resources is not None:
                G_s.nodes[i]["resources"] = node.resources._asdict()
        graph_json = json.dumps(nx.node_link_data(G_s, edges="edges"))
        wf = Workflow(graph=graph_json, num_tasks=repr.number_of_nodes())
        with Session(self._engine) as session:
            session.add(wf)
            session.commit()
            session.refresh(wf)
        return wf


    def add_runs(self, wf: Workflow, exec_ids: list[str], batch_id: str) -> Batch:
        if self._readonly:
            raise RuntimeError("Cannot register runs to readonly StudyStorage.")
        batch = Batch(workflow_id=wf.id, batch_id=batch_id)
        with Session(self._engine) as session:
            session.add(batch)
            session.flush()
            for exec_id in exec_ids:
                session.add(Execution(workflow_id=wf.id, batch_id=batch.id, exec_id=exec_id))
            session.commit()
            session.refresh(batch)
        return batch


    def get_batches(self) -> List[Batch]:
        with Session(self._engine) as session:
            return session.exec(select(Batch)).all()


    def get_executions_by_batch(self, batch_id: str) -> List[Execution]:
        with Session(self._engine) as session:
            return session.exec(
                select(Execution).where(Execution.batch_id == batch_id)
            ).all()


    def get_workflows(self) -> List[Workflow]:
        with Session(self._engine) as session:
            return session.exec(select(Workflow)).all()


    def get_workflow_graphs(self, as_nx: bool = True) -> List[nx.DiGraph]:
        with Session(self._engine) as session:
            records = session.exec(select(Workflow)).all()
        if as_nx:
            return [nx.node_link_graph(json.loads(r.graph)) for r in records]
        return [json.loads(r.graph) for r in records]


    def get_workflow_graph(self, exec_id: str, as_nx: bool = True) -> Optional[nx.DiGraph]:
        with Session(self._engine) as session:
            execution = session.exec(
                select(Execution).where(Execution.exec_id == exec_id)
            ).one_or_none()
            if execution is None:
                raise ValueError(f"No execution found for exec_id '{exec_id}'")
            wf = session.get(Workflow, execution.workflow_id)
        data = json.loads(wf.graph)
        return nx.node_link_graph(data) if as_nx else data


    def get_job_traces_by_batch(self, batch_id: str) -> List[JobTrace]:
        with Session(self._engine) as session:
            return session.exec(
                select(JobTrace)
                .select_from(JobTrace)
                .join(Execution, JobTrace.execution_id == Execution.id)
                .where(Execution.batch_id == batch_id)
                .order_by(JobTrace.start)
            ).all()

    def get_job_traces(self, exec_id: str) -> List[JobTrace]:
        with Session(self._engine) as session:
            return session.exec(
                select(JobTrace)
                .select_from(JobTrace)
                .join(Execution, JobTrace.execution_id == Execution.id)
                .where(Execution.exec_id == exec_id)
                .order_by(JobTrace.start)
            ).all()


class IntermediatesStorage(IntermediatesStorageBase):
    def __init__(self, engine: Engine, readonly: bool = False):
        super().__init__()
        self._engine = engine
        self._readonly = readonly

    def add_scalar(self, tag: str, step: int, value: float) -> None:
        if self._readonly:
            raise RuntimeError("Cannot write to readonly StudyStorage.")
        
        with Session(self._engine) as session:
            session.add(Intermediate(tag=tag, step=step, value=value, walltime=time.time()))
            session.commit()


class StudyStorage(StudyStorageBase):

    def __init__(self, engine: Engine, *, readonly: bool = False):
        super().__init__(db_path="")
        self._engine = engine
        self._readonly = readonly
        SQLModel.metadata.create_all(engine)
        self._workflow_storage = WorkflowStorage(engine, readonly)
        self._intermediates_storage = IntermediatesStorage(engine, readonly)


    @classmethod
    def create(cls, db_path: Path | str, *, readonly: bool = False, backend: str = "duckdb") -> "StudyStorage":
        if backend == "duckdb":
            engine = create_engine(f"duckdb:///{db_path}", poolclass=NullPool)
        elif backend == "sqlite":
            engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False})
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        return cls(engine, readonly=readonly)


    @classmethod
    def open(cls, db_path: Path | str, backend: str = "duckdb") -> "StudyStorage":
        return cls.create(db_path, readonly=True, backend=backend)

    def __exit__(self, exc_type, exc, tb):
        super().__exit__(exc_type, exc, tb)
        self._engine.dispose()
