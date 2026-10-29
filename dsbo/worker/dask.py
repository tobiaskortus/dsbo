from __future__ import annotations

import dask
import uuid
import torch
import warnings
import contextvars
import torch.autograd.forward_ad as fw

from collections import ChainMap
from dask.delayed import Delayed, delayed
from dask.distributed import Client, Future

import dsbo
from contextlib import contextmanager
from dsbo.storage import get_current_storage, use_storage
from dsbo.autograd import _pack, _unpack
from dsbo_heterogeneous.hybrid import HybridCluster
from dsbo.utils.io import get_workdir, get_file_suffix, get_subdir
from typing import Any, Generator, Optional, Iterable, TYPE_CHECKING


if TYPE_CHECKING:
    from dsbo.worker.spec import WorkerSpec
    from dsbo.workflow import Workflow
    from dsbo.modules import Module

dask.config.set({
    "distributed.worker.memory.target": 0.85,
    "distributed.worker.memory.spill": 0.90,
    "distributed.worker.memory.pause": 0.95,
    "distributed.worker.memory.terminate": 0.98,
})


def _pack_tensor(t: torch.Tensor | None):
    return None if t is None else _pack({"value": t})["value"]


def _unpack_tensor(t):
    return None if t is None else _unpack({"value": t})["value"]


class DaskWorkerBase:
    job_specs: tuple[WorkerSpec, ...] = ()

    def __init__(self, cluster):
        self.client = Client(cluster)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()
        self.cluster.close()

    def close(self):
        self.client.close()
        self.cluster.close()

    def compute(self, objs, **kwargs):
        return self.client.compute(objs, **kwargs)

    def gather(self, futures: Iterable[Future], catch: tuple = ()) -> list:
        """Should be okay since client.gather implements

        ```
        if isinstance(futures, Iterator):
            return (self.gather(f, errors=errors, direct=direct) for f in futures)
        ```
        
        internally
        """
        if not catch:
            return [(tid, _unpack(p)) for tid, p in self.client.gather(futures)]

        results = []
        for f in futures:
            try:
                tid, p = self.client.gather(f)
                results.append((tid, _unpack(p)))
            except Exception as e:
                if isinstance(e, catch):
                    continue
                raise
        return results

    def schedule(
        self,
        module: Module,
        design: dict,
        dependencies: Iterable = (),
        design_tensor: torch.Tensor | None = None,
        priority: int | None = None,
        file_overlays=None,
    ) -> tuple[uuid.UUID, Delayed]:
        suffix = get_file_suffix()
        subdir = get_subdir()
        workdir = getattr(get_workdir(), "workdir_root", None)
        resources = getattr(module.resources, "resources", None)
        packed_design = _pack(design)
        packed_design_tensor = _pack_tensor(design_tensor)

        is_forward_ad = fw._current_level > -1
        is_grad_enabled = torch.is_grad_enabled()

        config_state = dsbo.config.state()
        storage = get_current_storage(create=False)

        task_id = uuid.uuid4()

        def _module_wrapper(module, design_arg, design_tensor_arg, *deps):
            dsbo.config._set(config_state)
            task_kwargs = dict(
                design=design_arg,
                workdir=workdir,
                suffix=suffix,
                subdir=subdir,
                file_overlays=file_overlays,
                is_forward_ad=is_forward_ad,
                is_grad_enabled=is_grad_enabled,
                pack_result=is_forward_ad,
                unpack_design=True,
            )
            design_tensor = _unpack_tensor(design_tensor_arg)
            if design_tensor is not None:
                task_kwargs["design_tensor"] = design_tensor
            with use_storage(storage):
                packed_result = module.run(**task_kwargs)
            return task_id, packed_result

        with dask.annotate(resources=(resources or {}), priority=priority):
            delayed_obj = delayed(_module_wrapper, name=module.name)(
                module, packed_design, packed_design_tensor, *dependencies
            )

        return task_id, delayed_obj

    def schedule_backward(
        self, module: Module, candidate, grad_output=None,
        dependencies: Iterable = (), priority: int | None = None,
        file_overlays=None, gradient_overlays=None, 
        pass_id: uuid.UUID | None = None,
        task_id: uuid.UUID | None = None,
    ) -> tuple[uuid.UUID, Delayed]:
        suffix = get_file_suffix()
        subdir = get_subdir()
        workdir = getattr(get_workdir(), "workdir_root", None)
        resources = getattr(module.resources, "resources", None)

        config_state = dsbo.config.state()
        storage = get_current_storage(create=False)
        task_id = task_id or uuid.uuid4()
        pass_id = pass_id or uuid.uuid4()

        def _backward_wrapper(module, candidate, *deps):
            dsbo.config._set(config_state)
            grad_candidate = candidate.detach().requires_grad_()
            design_tensor = grad_candidate.tensor if getattr(module, "requires_design_tensor", False) else None
            with use_storage(storage), torch.enable_grad():
                module.run_backward(
                    grad_candidate.values, grad_output,
                    design_tensor=design_tensor,
                    workdir=workdir, suffix=suffix, subdir=subdir,
                    file_overlays=file_overlays,
                    gradient_overlays=gradient_overlays,
                    pass_id=pass_id,
                    gradient_writer_id=task_id.hex,
                )
            gradient = grad_candidate.grad.clone() if grad_candidate.grad is not None else None
            return task_id, gradient

        with dask.annotate(resources=(resources or {}), priority=priority):
            delayed_obj = delayed(_backward_wrapper, name=f"{module.name}_bwd")(module, candidate, *dependencies)

        return task_id, delayed_obj


def check_worker_available(worker: Optional[DaskWorkerBase]) -> bool:
    """Return whether every job queue backend required by ``worker`` is available."""
    if worker is None:
        return True
    return all(spec.is_available() for spec in worker.job_specs)


class WorkerContext:
    _current_worker = contextvars.ContextVar("current_worker", default=None)

    @classmethod
    @contextmanager
    def use_worker(cls, worker: Optional[DaskWorkerBase]) -> Generator[None, Any, None]:
        assert worker == None or isinstance(worker, DaskWorkerBase), "Worker must be an instance of DaskWorkerBase"
        if worker is not None and not check_worker_available(worker):
            unavailable = [type(spec).__name__ for spec in worker.job_specs if not spec.is_available()]
            warnings.warn(
                f"Job queue system required by the worker is not available on this machine "
                f"({', '.join(unavailable)}). Falling back to local execution.",
                UserWarning,
                stacklevel=2,
            )
            worker = None
        token = cls._current_worker.set(worker)
        try:
            yield
        finally:
            cls._current_worker.reset(token)

    @classmethod
    def get_current_worker(cls) -> Optional[DaskWorkerBase]:
        return cls._current_worker.get()


class HeterogeneousDaskWorker(DaskWorkerBase):
    def __init__(self, job_specs: Iterable[WorkerSpec], logdir: str = "logs"):
        self.logdir = logdir
        self.job_specs = tuple(job_specs)
        cluster_spec = dict(ChainMap(*[s.get_spec_dict() for s in self.job_specs]))

        adapt_config = dict()

        for spec in self.job_specs:
            assert (spec.min_worker is not None and 
                    spec.max_worker is not None), "Define static or adaptive scaling with .scale() or .adapt()"

            #TODO: enable constant scaling of workers + combinations of adapt and scale
            adapt_config[spec.spec_key] = (spec.min_worker, spec.max_worker)

        self.cluster = HybridCluster(spec=cluster_spec, log_directory=logdir)
        if check_worker_available(self):
            if all(minimum == maximum for minimum, maximum in adapt_config.values()):
                self.cluster.scale({key: minimum for key, (minimum, _) in adapt_config.items()})
            else:
                self.cluster.adapt(adapt_config)
        super().__init__(self.cluster)

    @classmethod
    def from_workflow(cls, workflow: Workflow, logdir: str = "logs") -> "HeterogeneousDaskWorker":
        job_specs = workflow.get_unique_workers()
        return cls(job_specs=job_specs, logdir=logdir)
