from __future__ import annotations

import dask
import uuid
import time
import torch
import contextvars
import torch.distributed as dist
import torch.autograd.forward_ad as fw

from collections import ChainMap
from dask.delayed import Delayed, delayed
from dask.distributed import Client, Future

import dsbo
from contextlib import contextmanager
from dsbo.storage import get_current_storage, use_storage
from dsbo.autograd import _pack, _unpack
from dsbo.worker.ddp import (
    get_worker_info,
    configure_rank_env,
    assert_ddp_supported,
    init_rendezvous,
    DDPDispatchError,
    DDP_RENDEZVOUS_TIMEOUT_SECONDS,
)
from dask_heterogeneous.hybrid import HybridCluster
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


class DaemonicWorkerError(RuntimeError):
    """A task attempted nested multiprocessing from a daemonic Dask worker."""


@contextmanager
def translate_daemonic_worker_error():
    try:
        yield
    except Exception as exc:
        current = exc
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if "daemonic processes are not allowed to have children" in str(current):
                raise DaemonicWorkerError(
                    "Task attempted to start a multiprocessing child from a "
                    "daemonic Dask worker. Configure WorkerSpec(daemon=False) "
                    "or disable child workers for this task."
                ) from None
            current = current.__cause__ or current.__context__
        raise


def _pack_tensor(t: torch.Tensor | None):
    return None if t is None else _pack({"value": t})["value"]


def _unpack_tensor(t):
    return None if t is None else _unpack({"value": t})["value"]


def _capture_schedule_state(design, design_tensor):
    """Capture scheduling context once so worker wrappers stay picklable."""
    return dict(
        suffix=get_file_suffix(),
        subdir=get_subdir(),
        workdir=getattr(get_workdir(), "workdir_root", None),
        packed_design=_pack(design),
        packed_design_tensor=_pack_tensor(design_tensor),
        is_forward_ad=fw._current_level > -1,
        is_grad_enabled=torch.is_grad_enabled(),
        config_state=dsbo.config.state(),
        storage=get_current_storage(create=False),
    )


def _task_kwargs(state, design_arg, design_tensor_arg, *, file_overlays):
    task_kwargs = dict(
        design=design_arg,
        workdir=state["workdir"],
        suffix=state["suffix"],
        subdir=state["subdir"],
        file_overlays=file_overlays,
        is_forward_ad=state["is_forward_ad"],
        is_grad_enabled=state["is_grad_enabled"],
        pack_result=state["is_forward_ad"],
        unpack_design=True,
    )
    design_tensor = _unpack_tensor(design_tensor_arg)
    if design_tensor is not None:
        task_kwargs["design_tensor"] = design_tensor
    return task_kwargs


def _run_module(module, design_arg, design_tensor_arg, state, file_overlays, task_id):
    dsbo.config._set(state["config_state"])
    task_kwargs = _task_kwargs(
        state, design_arg, design_tensor_arg, file_overlays=file_overlays
    )
    with translate_daemonic_worker_error(), use_storage(state["storage"]):
        return task_id, module.run(**task_kwargs)


class DaskWorkerBase:
    def __init__(self, cluster):
        self.cluster = cluster
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

    def _scale_rank_workers(self, resource_key: str, world_size: int, accelerator: str = "auto") -> list[dict[str, Any]]:
        # TODO: This is a bit messy and questionable at least, replace with elastic dddp scaling 
        # and the automatic adapt provided by dask.
        self.cluster.scale({resource_key: world_size})
        deadline = time.monotonic() + DDP_RENDEZVOUS_TIMEOUT_SECONDS
        while True:
            available = sum(
                1
                for metadata in self.client.scheduler_info()
                .get("workers", {})
                .values()
                if metadata.get("resources", {}).get(resource_key, 0) >= 1
            )
            if available >= world_size:
                break
            if time.monotonic() >= deadline:
                raise DDPDispatchError(
                    f"DDP dispatch requested {world_size} rank slots but only "
                    f"{available} workers with resource '{resource_key}' appeared "
                    f"within {DDP_RENDEZVOUS_TIMEOUT_SECONDS}s of scaling."
                )
            time.sleep(0.5)
        return get_worker_info(
            self.client, world_size, accelerator=accelerator, resource_key=resource_key
        )

    def schedule(
        self,
        module: Module,
        design: dict,
        dependencies: Iterable = (),
        design_tensor: torch.Tensor | None = None,
        priority: int | None = None,
        file_overlays=None,
        world_size: int | None = None,
        accelerator: str = "auto",
    ) -> list[tuple[uuid.UUID, Delayed]]:
        """Schedule ``module``'s task on a worker.

        Returns a list of ``(task_id, delayed)`` pairs: a single pair for normal
        execution, or one pair per rank when the module carries ``n_ranks > 1``
        (DDP). DDP rank tasks are each pinned to a distinct matching worker
        address and initialize ``torch.distributed`` around ``module.run``.
        """
        spec = module.resources
        resource_key = getattr(spec, "resource_key", None)
        world_size = world_size or getattr(spec, "n_ranks", 1) or 1

        state = _capture_schedule_state(design, design_tensor)

        if world_size <= 1:
            #default execution without ddp scaling‚
            resources = getattr(module.resources, "resources", None)
            task_id = uuid.uuid4()

            def _module_wrapper(module, design_arg, design_tensor_arg, *dependencies):
                return _run_module(module, design_arg, design_tensor_arg, state, file_overlays, task_id)

            with dask.annotate(resources=(resources or {}), priority=priority):
                delayed_obj = delayed(_module_wrapper, name=module.name)(
                    module, 
                    state["packed_design"], 
                    state["packed_design_tensor"],
                    *dependencies,
                )
            return [(task_id, delayed_obj)]


        assert_ddp_supported(spec, world_size)

        worker_infos = self._scale_rank_workers(resource_key, world_size, accelerator=accelerator)
        scheduled: list[tuple[uuid.UUID, Delayed]] = []

        for info in worker_infos:
            task_id = uuid.uuid4()

            def _ddp_rank_module_wrapper(module, design_arg, design_tensor_arg, info, rank_task_id, *dependencies):
                configure_rank_env(info)
                init_rendezvous(info)
                try:
                    return _run_module(module, design_arg, design_tensor_arg,
                                       state, file_overlays, rank_task_id)
                finally:
                    if dist.is_initialized():
                        dist.destroy_process_group()

            annotations = {
                "resources": {resource_key: 1} if resource_key else None,
                "workers": [info["worker"]],
                **({"priority": priority} if priority is not None else {}),
            }

            with dask.annotate(**annotations):
                delayed_fn = delayed(_ddp_rank_module_wrapper, name=f"{module.name}_rank{info['rank']}")
                delayed_obj = delayed_fn(
                    module, 
                    state["packed_design"], 
                    state["packed_design_tensor"],
                    info, task_id, 
                    *dependencies,
                )
            scheduled.append((task_id, delayed_obj))

        return scheduled


    def schedule_backward(self, module: Module, candidate, grad_output=None, dependencies: Iterable = (), priority: int | None = None) -> tuple[uuid.UUID, Delayed]:
        suffix = get_file_suffix()
        subdir = get_subdir()
        workdir = getattr(get_workdir(), "workdir_root", None)
        resources = getattr(module.resources, "resources", None)

        config_state = dsbo.config.state()
        storage = get_current_storage(create=False)
        task_id = uuid.uuid4()

        def _backward_wrapper(module, candidate, *deps):
            dsbo.config._set(config_state)
            grad_candidate = candidate.detach().requires_grad_()
            design_tensor = grad_candidate.tensor if getattr(module, "requires_design_tensor", False) else None
            with translate_daemonic_worker_error(), use_storage(storage), torch.enable_grad():
                module.run_backward(
                    grad_candidate.values, grad_output,
                    design_tensor=design_tensor,
                    workdir=workdir, suffix=suffix, subdir=subdir,
                )
            return task_id, grad_candidate.grad.clone() if grad_candidate.grad is not None else None

        with dask.annotate(resources=(resources or {}), priority=priority):
            delayed_obj = delayed(_backward_wrapper, name=f"{module.name}_bwd")(module, candidate, *dependencies)

        return task_id, delayed_obj

class WorkerContext:
    _current_worker = contextvars.ContextVar("current_worker", default=None)

    @classmethod
    @contextmanager
    def use_worker(cls, worker: Optional[DaskWorkerBase]) -> Generator[None, Any, None]:
        assert worker == None or isinstance(worker, DaskWorkerBase), "Worker must be an instance of DaskWorkerBase"
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
        job_specs = tuple(job_specs)
        cluster_spec = dict(ChainMap(*[s.get_spec_dict() for s in job_specs]))

        adapt_config = dict()

        for spec in job_specs:
            assert (spec.min_worker is not None and 
                    spec.max_worker is not None), "Define static or adaptive scaling with .scale() or .adapt()"

            #TODO: enable constant scaling of workers + combinations of adapt and scale
            minimum, maximum = adapt_config.get(spec.spec_key, (0, 0))
            adapt_config[spec.spec_key] = (
                max(minimum, spec.min_worker),
                max(maximum, spec.max_worker),
            )

        self.cluster = HybridCluster(spec=cluster_spec, log_directory=logdir)
        if all(minimum == maximum for minimum, maximum in adapt_config.values()):
            self.cluster.scale({key: minimum for key, (minimum, _) in adapt_config.items()})
        else:
            self.cluster.adapt(adapt_config)
        super().__init__(self.cluster)

    @classmethod
    def from_workflow(cls, workflow: Workflow, logdir: str = "logs") -> "HeterogeneousDaskWorker":
        job_specs = workflow.get_unique_workers()
        return cls(job_specs=job_specs, logdir=logdir)
