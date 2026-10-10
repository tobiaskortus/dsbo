from __future__ import annotations

import os
import time
import json
import shutil
import uuid
import torch
import numpy as np
import dataclasses
from pathlib import Path
from typing import List, Optional

import networkx as nx
from sqlalchemy import Engine, event as sa_event, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import NullPool
from sqlmodel import Session, SQLModel, create_engine, select

from dsbo.storage import IntermediatesStorageBase, StudyStorageBase, TransitionWriter, TrialStorageBase, WorkflowStorageBase, ArtifactStorageBase, Artifact
from dsbo.storage.models import Batch, Execution, Intermediate, JobStatus, JobTrace, Trial, Workflow, ArtifactEntry, StudyMetadata
from dsbo.utils.io.workdir import Workdir


class StudyStorageExistsError(FileExistsError):
    pass


def _configure_sqlite_vec(engine: Engine, vector_dim: int) -> None:
    import sqlite_vec

    @sa_event.listens_for(engine, "connect")
    def _load_vec(conn, _):
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

    with engine.connect() as conn:
        conn.exec_driver_sql(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_trials "
            f"USING vec0(design float[{vector_dim}])"
        )
        conn.exec_driver_sql(
            "CREATE TABLE IF NOT EXISTS vec_trial_map "
            "(rowid INTEGER PRIMARY KEY, trial_id TEXT NOT NULL)"
        )
        conn.commit()


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
        tag: str = "",
    ) -> None:
        engine = create_engine(self.url, poolclass=NullPool)
        with Session(engine) as session:
            execution = session.exec(select(Execution).where(Execution.exec_id == exec_id)).one()
            wf = session.get(Workflow, execution.workflow_id)

            if status in (JobStatus.RUNNING, JobStatus.QUEUED):
                query = select(JobTrace).where(
                    JobTrace.execution_id == execution.id,
                    JobTrace.task_key == key
                )
                job_trace = session.exec(query).first()
                
                if job_trace is None:
                    job_trace = JobTrace(
                        execution_id=execution.id,
                        task_key=key,
                        name=name,
                        worker=worker,
                        start=start,
                        stop=stop,
                        log=log,
                        status=status,
                        tag=tag,
                    )
                else:
                    job_trace.status = status
                    if status == JobStatus.RUNNING:
                        job_trace.start = start
                        job_trace.worker = worker
            else:
                query = select(JobTrace).where(
                    JobTrace.execution_id == execution.id,
                    JobTrace.task_key == key
                )
                job_trace = session.exec(query).one()
                job_trace.worker = worker
                job_trace.start = start
                job_trace.stop = stop
                job_trace.log = log
                job_trace.status = status
                job_trace.tag = tag

            session.add(job_trace)

            if execution.start < 0: execution.start = start

            match status:
                case JobStatus.SUCCESS:
                    execution.num_successful_jobs += 1
                case JobStatus.FAILED:
                    execution.status = JobStatus.FAILED
                case JobStatus.RUNNING:
                    execution.status = JobStatus.RUNNING 
                case JobStatus.QUEUED:
                    execution.status = JobStatus.QUEUED
                case _:
                    execution.status = JobStatus.PENDING
            
            #Mark execution as SUCCESS if all jobs are done and none have failed
            if execution.num_successful_jobs >= wf.num_tasks and execution.status != JobStatus.FAILED:
                execution.status = JobStatus.SUCCESS
                execution.stop = stop

            execution.updated_at = stop
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
                G_s.nodes[i]["resources"] = dataclasses.asdict(node.resources)
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
        
        batch = None
        if len(exec_ids) > 1:
            batch = Batch(workflow_id=wf.id, batch_id=batch_id)

        with Session(self._engine) as session:
            if batch is not None:
                session.add(batch)
                session.flush()

            for exec_id in exec_ids:
                session.add(Execution(workflow_id=wf.id, batch_id=batch.id if batch is not None else None, 
                                      exec_id=exec_id, num_jobs=wf.num_tasks))
            
            session.commit()
            if batch is not None:
                session.refresh(batch)
        return batch



    def get_batches(self) -> List[Batch]:
        with Session(self._engine) as session:
            return session.exec(select(Batch)).all()


    def get_executions(self, batch_id: Optional[str] = None, updated_after: Optional[float] = None) -> List[Execution]:
        with Session(self._engine) as session:
            query = select(Execution)
            if batch_id is not None:
                query = query.where(Execution.batch_id == batch_id)
            if updated_after is not None:
                query = query.where(Execution.updated_at > updated_after)
            return session.exec(query).all()


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


    def get_job_traces_by_batch(self, batch_id: str, after_trace_id: Optional[int] = None) -> List[JobTrace]:
        with Session(self._engine) as session:
            query = (
                select(JobTrace)
                .select_from(JobTrace)
                .join(Execution, JobTrace.execution_id == Execution.id)
                .where(Execution.batch_id == batch_id)
            )
            
            if after_trace_id is not None:
                query = query.where(JobTrace.id > after_trace_id)
            
            return session.exec(query.order_by(JobTrace.id)).all()

    def get_job_traces(self, exec_id: str, after_trace_id: Optional[int] = None) -> List[JobTrace]:
        with Session(self._engine) as session:
            query = (
                select(JobTrace)
                .select_from(JobTrace)
                .join(Execution, JobTrace.execution_id == Execution.id)
                .where(Execution.exec_id == exec_id)
            )
            
            if after_trace_id is not None:
                query = query.where(JobTrace.id > after_trace_id)
            
            return session.exec(query.order_by(JobTrace.id)).all()


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

    def get_scalars(self, tag: str | None = None) -> list[float] | dict[str, list[float]]:
        with Session(self._engine) as session:
            query = select(Intermediate).order_by(Intermediate.step)
            if tag is not None:
                query = query.where(Intermediate.tag == tag)
            rows = session.exec(query).all()

        if tag is not None:
            return [r.value for r in rows]

        result: dict[str, list[float]] = {}
        for r in rows:
            result.setdefault(r.tag, []).append(r.value)
        return result

    def get_tags(self) -> list[str]:
        with Session(self._engine) as session:
            rows = session.exec(
                select(Intermediate.tag).distinct().order_by(Intermediate.tag)
            ).all()
        return list(rows)


class TrialStorage(TrialStorageBase):
    def __init__(self, engine: Engine, readonly: bool = False, vector_dim: int | None = None):
        super().__init__()
        self._engine = engine
        self._readonly = readonly
        self._vector_dim = vector_dim

    def add_trial(self, step: int, design_vec: np.ndarray | torch.Tensor, objectives: dict,
                  constraints: Optional[dict] = None, feasible: bool = True) -> Trial:
        
        if self._readonly:
            raise RuntimeError("Cannot write to readonly TrialStorage.")

        if isinstance(design_vec, torch.Tensor):
            design_vec = design_vec.detach().cpu().numpy()

        objectives = self._json_scalars(objectives, "objectives")
        if constraints is not None:
            constraints = self._json_scalars(constraints, "constraints")

        arr = np.asarray(design_vec, dtype=np.float32)
        with Session(self._engine) as session:
            trial = Trial(step=int(step), design=arr.tobytes(), 
                          objectives=objectives, constraints=constraints, 
                          feasible=feasible)
            
            session.add(trial)
            session.commit()
            session.refresh(trial)
            trial_id = trial.id

        if self._vector_dim is not None:
            self._index_design(trial_id, arr)

        return trial

    @staticmethod
    def _json_scalars(values: dict, name: str) -> dict:
        result = {}
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError(f"{name}['{key}'] must be a scalar tensor.")
                value = value.detach().cpu().item()
            elif isinstance(value, np.generic):
                value = value.item()
            result[key] = value
        return result


    def _index_design(self, trial_id: str, arr: np.ndarray) -> None:
        with self._engine.connect() as conn:
            result = conn.exec_driver_sql(
                "SELECT COALESCE(MAX(rowid), 0) + 1 FROM vec_trial_map"
            )
            next_rowid = result.scalar()
            conn.exec_driver_sql(
                "INSERT INTO vec_trials(rowid, design) VALUES (?, ?)",
                (next_rowid, arr.tobytes()),
            )
            conn.exec_driver_sql(
                "INSERT INTO vec_trial_map(rowid, trial_id) VALUES (?, ?)",
                (next_rowid, trial_id),
            )
            conn.commit()


    def get_trials(self, since_step: Optional[int] = None) -> list[Trial]:
        with Session(self._engine) as session:
            query = select(Trial)
            if since_step is not None:
                query = query.where(Trial.step > since_step)
            return session.exec(query.order_by(Trial.step)).all()

    def get_similar_trials(self, query_vec: np.ndarray | torch.Tensor,
                              k: int = 10, metric: str = "l2") -> list[Trial]:
        if self._vector_dim is None:
            raise NotImplementedError(
                "Vector search is not enabled. Pass vector_dim to StudyStorage.create() to enable it."
            )

        if metric != "l2":
            raise ValueError("sqlite-vec only supports the 'l2' metric.")

        if isinstance(query_vec, torch.Tensor):
            query_vec = query_vec.detach().cpu().numpy()

        arr = np.asarray(query_vec, dtype=np.float32)

        with self._engine.connect() as conn:
            result = conn.exec_driver_sql(
                "SELECT vec_trial_map.trial_id "
                "FROM vec_trials "
                "JOIN vec_trial_map ON vec_trials.rowid = vec_trial_map.rowid "
                "WHERE vec_trials.design MATCH ? AND k = ? "
                "ORDER BY vec_trials.distance",
                (arr.tobytes(), k),
            )
            trial_ids = [row[0] for row in result]

        if not trial_ids: return []
        
        with Session(self._engine) as session:
            trials = session.exec(select(Trial).where(Trial.id.in_(trial_ids))).all()

        id_to_trial = {t.id: t for t in trials}
        return [id_to_trial[tid] for tid in trial_ids if tid in id_to_trial]


class ArtifactStorage(ArtifactStorageBase):
    def __init__(self, storage_name: str, engine: Engine):
        super().__init__(storage_name)
        self._engine = engine

    def create_artifact(self, name: str, type: str, metadata: dict | None = None) -> Artifact:
        return Artifact(name, type, metadata)

    def save_artifact(self, artifact: Artifact, stage: str | None = None,
                      step: int | None = None) -> Artifact:
        root = Path(Workdir.get_current_dir()).as_posix()
        artifacts_dir = Path(root) / "artifacts" / artifact.name
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # 1. Stage files atomically so a crash never leaves a torn version dir.
        staging = artifacts_dir / f".staging-{uuid.uuid4().hex}"
        try:
            artifact._save_to_dir(staging)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        # 2. Reserve a unique version atomically (unique constraint + retry).
        with Session(self._engine) as session:
            while True:
                version = self._next_version(session, artifact.name, root)
                entry = ArtifactEntry(
                    name=artifact.name,
                    type=artifact.type,
                    version=version,
                    step=step,
                    stage=stage,
                    meta=artifact._metadata,
                    root=root,
                )
                session.add(entry)
                entry_id = entry.id
                try:
                    session.commit()
                    break
                except IntegrityError:
                    session.rollback()
                    continue

        # 3. Move the staged dir into the final versioned location.
        final = artifacts_dir / f"v{version}"
        if final.exists():
            shutil.rmtree(final, ignore_errors=True)
        try:
            os.replace(staging, final)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            with Session(self._engine) as session:
                entry = session.get(ArtifactEntry, entry_id)
                if entry is not None:
                    session.delete(entry)
                    session.commit()
            raise

        artifact._version = version
        artifact._stage = stage
        artifact._step = step
        artifact._write_manifest(final)

        with Session(self._engine) as session:
            entry = session.get(ArtifactEntry, entry_id)
            entry.artifact_dir = str(final)
            entry.filepath = str(final / "artifact.json")
            session.add(entry)
            session.commit()

        return artifact

    def _next_version(self, session: Session, name: str, root: str) -> int:
        latest = session.exec(
            select(ArtifactEntry.version)
            .where(ArtifactEntry.name == name, ArtifactEntry.root == root)
            .order_by(ArtifactEntry.version.desc())
            .limit(1)
        ).first()
        return (latest + 1) if latest is not None else 1

    def load_artifact(self, name: str, version: int | None = None,
                      stage: str | None = None) -> Artifact:
        with Session(self._engine) as session:
            query = select(ArtifactEntry).where(ArtifactEntry.name == name)
            if version is not None:
                query = query.where(ArtifactEntry.version == version)
            if stage is not None:
                query = query.where(ArtifactEntry.stage == stage)
            query = query.order_by(ArtifactEntry.version.desc())
            entry = session.exec(query).first()

            if entry is None:
                raise FileNotFoundError(
                    f"No artifact found for '{name}'"
                    + (f" (version={version})" if version else "")
                    + (f" (stage={stage})" if stage else "")
                )

        return Artifact._load_from_dir(
            Path(entry.artifact_dir),
            entry.name,
            entry.type,
            entry.version,
            entry.stage,
            entry.step,
            dict(entry.meta),
        )

    def latest_version(self, name: str, stage: str | None = None,
                       subdir: str | None = None) -> int | None:
        with Session(self._engine) as session:
            query = select(ArtifactEntry).where(ArtifactEntry.name == name)
            if stage is not None:
                query = query.where(ArtifactEntry.stage == stage)
            if subdir is not None:
                query = query.where(ArtifactEntry.root == subdir)
            query = query.order_by(ArtifactEntry.version.desc())
            latest = session.exec(query).first()
            return latest.version if latest is not None else None

    def list_artifacts(self, type: str | None = None) -> list[dict]:
        with Session(self._engine) as session:
            query = select(ArtifactEntry).order_by(ArtifactEntry.name, ArtifactEntry.version)
            if type is not None:
                query = query.where(ArtifactEntry.type == type)
            entries = session.exec(query).all()

        return [
            {
                "name": e.name,
                "type": e.type,
                "version": e.version,
                "stage": e.stage,
                "step": e.step,
                "metadata": dict(e.meta),
                "created_at": e.created_at,
            }
            for e in entries
        ]

class StudyStorage(StudyStorageBase):

    def __init__(self, engine: Engine, *, readonly: bool = False, vector_dim: int | None = None):
        super().__init__(db_path="")
        self._engine = engine
        self._readonly = readonly
        self._vector_dim = vector_dim
        inspector = inspect(engine)
        if not inspector.get_table_names():  # DB is empty
            SQLModel.metadata.create_all(engine)

        self._trial_storage = TrialStorage(engine, readonly, vector_dim=vector_dim)
        self._workflow_storage = WorkflowStorage(engine, readonly)
        self._intermediates_storage = IntermediatesStorage(engine, readonly)
        self._artifact_storage = ArtifactStorage("artifacts", engine)

    @property
    def created_at(self) -> float | None:
        with Session(self._engine) as session:
            metadata = session.get(StudyMetadata, 1)
            return metadata.created_at if metadata is not None else None

    def __getstate__(self) -> dict:
        url = self._engine.url.render_as_string(hide_password=False)
        if url == "sqlite:///:memory:":
            raise TypeError("In-memory study storage cannot be shared with worker processes.")
        return {
            "url": url,
            "readonly": self._readonly,
            "vector_dim": self._vector_dim,
        }

    def __setstate__(self, state: dict) -> None:
        url = state["url"]
        connect_args = {"check_same_thread": False} if url.startswith("sqlite:") else {}
        engine = create_engine(url, connect_args=connect_args)
        vector_dim = state["vector_dim"]
        if vector_dim is not None:
            _configure_sqlite_vec(engine, vector_dim)
        self.__init__(
            engine,
            readonly=state["readonly"],
            vector_dim=vector_dim,
        )


    @classmethod
    def create(cls, db_path: Path | str, *, readonly: bool = False,
               vector_dim: int | None = None) -> "StudyStorage":
        db_path = Path(db_path).resolve()
        if db_path.exists():
            raise StudyStorageExistsError(
                f"Study storage '{db_path}' already exists. Use "
                "StudyStorage.open(...) to access the existing study."
            )
        
        created_at = time.time()
        engine = create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        if vector_dim is not None:
            _configure_sqlite_vec(engine, vector_dim)
        storage = cls(engine, readonly=readonly, vector_dim=vector_dim)
        StudyMetadata.__table__.create(engine, checkfirst=True)
        with Session(engine) as session:
            session.add(StudyMetadata(created_at=created_at))
            session.commit()
        return storage

    @classmethod
    def open(cls, db_path: Path | str, vector_dim: int | None = None) -> "StudyStorage":
        db_path = Path(db_path).resolve()
        if not db_path.is_file():
            raise FileNotFoundError(f"Study storage '{db_path}' does not exist.")
        engine = create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        if vector_dim is not None:
            _configure_sqlite_vec(engine, vector_dim)
        return cls(engine, readonly=True, vector_dim=vector_dim)
    
    def close(self):
        self._engine.dispose()

    def __exit__(self, exc_type, exc, tb):
        super().__exit__(exc_type, exc, tb)
        self.close()
