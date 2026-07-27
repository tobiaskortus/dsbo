from __future__ import annotations

import time
import json
import torch
import numpy as np
import dataclasses
from pathlib import Path
from typing import List, Optional

import networkx as nx
from sqlalchemy import Engine, event as sa_event, inspect
from sqlalchemy.pool import NullPool
from sqlmodel import Session, SQLModel, case, create_engine, select

from dsbo.storage import IntermediatesStorageBase, StudyStorageBase, TransitionWriter, TrialStorageBase, WorkflowStorageBase, ArtifactStorageBase, Artifact
from dsbo.storage.models import Batch, Execution, Intermediate, JobStatus, JobTrace, Trial, Workflow, ArtifactEntry
from dsbo.utils.io.workdir import Workdir


DUCKDB_VECTOR_METRICS = {
        "l2":     "array_distance",
        "cosine": "array_cosine_distance",
        "ip":     "array_negative_inner_product",
    }


def _configure_duckdb_vss(engine: Engine, vector_dim: int) -> None:
    @sa_event.listens_for(engine, "connect")
    def _load_vss(conn, _):
        try:
            conn.execute("LOAD vss")
        except Exception:
            conn.execute("INSTALL vss")
            conn.execute("LOAD vss")
        conn.execute("SET hnsw_enable_experimental_persistence = true")

    with engine.connect() as conn:
        conn.exec_driver_sql(
            f"CREATE TABLE IF NOT EXISTS trial_vectors "
            f"(trial_id VARCHAR, vec FLOAT[{vector_dim}])"
        )
        conn.commit()

    with engine.connect() as conn:
        try:
            conn.exec_driver_sql(
                f"CREATE INDEX IF NOT EXISTS trial_vectors_hnsw "
                f"ON trial_vectors USING HNSW(vec)"
            )
            conn.commit()
        except Exception:
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
    def __init__(self, engine: Engine, readonly: bool = False, vector_dim: int | None = None, backend: str = "duckdb"):
        super().__init__()
        self._engine = engine
        self._backend = backend
        self._readonly = readonly
        self._vector_dim = vector_dim

    def add_trial(self, step: int, design_vec: np.ndarray | torch.Tensor, objectives: dict,
                  constraints: Optional[dict] = None, feasible: bool = True) -> Trial:
        
        if self._readonly:
            raise RuntimeError("Cannot write to readonly TrialStorage.")

        if isinstance(design_vec, torch.Tensor):
            design_vec = design_vec.detach().cpu().numpy()

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


    def _index_design(self, trial_id: str, arr: np.ndarray) -> None:
        def _index_design_duckdb():
            with self._engine.connect() as conn:
                conn.exec_driver_sql(
                    f"INSERT INTO trial_vectors VALUES (?, ?::FLOAT[{self._vector_dim}])",
                    (trial_id, arr.tolist()),
                )
                conn.commit()

        def _index_design_sqlite():
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

        match self._backend:
            case "duckdb":
                _index_design_duckdb()
            case "sqlite":
                _index_design_sqlite()
            case _:
                raise ValueError(f"Unsupported backend: {self._backend}")
            

    def get_trials(self, since_step: Optional[int] = None) -> list[Trial]:
        with Session(self._engine) as session:
            query = select(Trial)
            if since_step is not None:
                query = query.where(Trial.step > since_step)
            return session.exec(query.order_by(Trial.step)).all()

    def get_similar_trials(self, query_vec: np.ndarray | torch.Tensor, 
                              k: int = 10, metric: str = "l2") -> list[Trial]:
        
        def _search_duckdb(arr: np.ndarray, k: int, metric: str) -> list[Trial]:
            fn = DUCKDB_VECTOR_METRICS.get(metric)
            if fn is None:
                raise ValueError(
                    f"Unsupported metric '{metric}' for DuckDB. "
                    f"Choose one of: {list(DUCKDB_VECTOR_METRICS)}")
            
            with self._engine.connect() as conn:
                result = conn.exec_driver_sql(
                    f"SELECT trial_id FROM trial_vectors "
                    f"ORDER BY {fn}(vec, ?::FLOAT[{self._vector_dim}]) LIMIT {k}",
                    (arr.tolist(),),
                )
                trial_ids = [row[0] for row in result]

            return trial_ids

    
        def _search_sqlite(arr: np.ndarray, k: int, metric: str) -> list[Trial]:
            if metric != "l2":
                raise ValueError(
                    "sqlite-vec only supports 'l2' metric. "
                    "Use the DuckDB backend for cosine or inner-product search.")
            
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
            
            return trial_ids


        if self._vector_dim is None:
            raise NotImplementedError(
                "Vector search is not enabled. Pass vector_dim to StudyStorage.create() to enable it."
            )

        if isinstance(query_vec, torch.Tensor):
            query_vec = query_vec.detach().cpu().numpy()

        arr = np.asarray(query_vec, dtype=np.float32)

        match self._backend:
            case "duckdb":
                trial_ids =  _search_duckdb(arr, k, metric)
            case "sqlite":
                trial_ids =  _search_sqlite(arr, k, metric)
            case _:
                raise ValueError(f"Unsupported backend: {self._backend}")
            
        if not trial_ids: return []
        
        with Session(self._engine) as session:
            trials = session.exec(select(Trial).where(Trial.id.in_(trial_ids))).all()

        id_to_trial = {t.id: t for t in trials}
        return [id_to_trial[tid] for tid in trial_ids if tid in id_to_trial]


class ArtifactStorage(ArtifactStorageBase):
    def __init__(self, storage_name: str, engine: Engine):
        super().__init__(storage_name)
        self._engine = engine

    def write_artifact_metadata(self, tag: str, step: int, metadata: dict) -> None:
        pass

    def load_artifact_metadata(self, tag: str, step: int | None = None) -> Optional[dict]:
        pass

    def create_artifact(self, name: str, type: str, metadata: dict | None = None) -> Artifact:
        return Artifact(name, type, metadata)

    def save_artifact(self, artifact: Artifact, stage: str | None = None,
                      step: int | None = None) -> Artifact:
        current_dir = Workdir.get_current_dir()
       
        prefix = Path(current_dir).as_posix() + "/artifacts/" + artifact.name + "/v"
        with Session(self._engine) as session:
            latest = session.exec(
                select(ArtifactEntry)
                .where(ArtifactEntry.name == artifact.name)
                .where(ArtifactEntry.artifact_dir.startswith(prefix))
                .order_by(ArtifactEntry.version.desc())
            ).first()
            version = (latest.version + 1) if latest is not None else 1

        artifact_dir = Path(current_dir) / "artifacts" / artifact.name / f"v{version}"
        artifact._version = version
        artifact._stage = stage
        artifact._step = step
        artifact._save_to_dir(artifact_dir)

        with Session(self._engine) as session:
            entry = ArtifactEntry(
                name=artifact.name,
                type=artifact.type,
                version=version,
                step=step,
                stage=stage,
                meta=artifact._metadata,
                artifact_dir=str(artifact_dir),
                filepath=str(artifact_dir / "artifact.json"),
            )
            session.add(entry)
            session.commit()

        return artifact

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
                prefix = subdir + "/artifacts/" + name + "/v"
                query = query.where(ArtifactEntry.artifact_dir.startswith(prefix))
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

    def __init__(self, engine: Engine, *, readonly: bool = False, vector_dim: int | None = None,
                 backend: str = "duckdb"):
        super().__init__(db_path="")
        self._engine = engine
        self._readonly = readonly
        inspector = inspect(engine)
        if not inspector.get_table_names():  # DB is empty
            SQLModel.metadata.create_all(engine)

        self._trial_storage = TrialStorage(engine, readonly, vector_dim=vector_dim, backend=backend)
        self._workflow_storage = WorkflowStorage(engine, readonly)
        self._intermediates_storage = IntermediatesStorage(engine, readonly)
        self._artifact_storage = ArtifactStorage("artifacts", engine)


    @classmethod
    def create(cls, db_path: Path | str, *, readonly: bool = False, backend: str = "duckdb",
               vector_dim: int | None = None) -> "StudyStorage":
        match backend:
            case "duckdb":
                engine = create_engine(f"duckdb:///{db_path}", poolclass=NullPool)
                if vector_dim is not None:
                    _configure_duckdb_vss(engine, vector_dim)
            case "sqlite":
                engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
                if vector_dim is not None:
                    _configure_sqlite_vec(engine, vector_dim)
            case _:
                raise ValueError(f"Unsupported backend: {backend}")

        return cls(engine, readonly=readonly, vector_dim=vector_dim, backend=backend)

    @classmethod
    def open(cls, db_path: Path | str, backend: str = "duckdb", vector_dim: int | None = None) -> "StudyStorage":
        return cls.create(db_path, readonly=True, backend=backend, vector_dim=vector_dim)
    
    def close(self):
        self._engine.dispose()

    def __exit__(self, exc_type, exc, tb):
        super().__exit__(exc_type, exc, tb)
        self.close()
