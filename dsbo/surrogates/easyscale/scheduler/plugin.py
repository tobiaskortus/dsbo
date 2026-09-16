import time
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any
from distributed.core import Status
from distributed.diagnostics.plugin import SchedulerPlugin, WorkerPlugin

from dsbo.surrogates.easyscale.protocol import Proposal
from dsbo.surrogates.easyscale.scheduler.allocation import Allocation
from dsbo.surrogates.easyscale.scheduler.arbiter import RoundArbiter
from dsbo.surrogates.easyscale.scheduler.registry import JobRegistry
from dsbo.surrogates.easyscale.scheduler.state import EasyScaleJobState


EASYSCALE_PLUGIN_NAME = "dsbo-easyscale"
EASYSCALE_RUNTIME_PLUGIN_NAME = "dsbo-easyscale-runtime"

_HANDLERS = {
    "dsbo_easyscale_register": "_register",
    "dsbo_easyscale_submit_proposals": "_submit_proposals",
    "dsbo_easyscale_decision": "_decision",
    "dsbo_easyscale_complete_generation": "_complete_generation",
    "dsbo_easyscale_fail_generation": "_fail_generation",
    "dsbo_easyscale_fallback": "_fallback",
    "dsbo_easyscale_release_allocation": "_release_allocation",
    "dsbo_easyscale_capacity": "_capacity",
    "dsbo_easyscale_worker_ready": "_worker_ready",
    "dsbo_easyscale_close": "_close_job",
}


class EasyScaleRuntimePlugin(WorkerPlugin):
    def setup(self, worker) -> None:
        import dsbo.surrogates.easyscale.runner  # noqa: F401

        self._dataloaders = {}

        async def report_ready() -> None:
            await worker.scheduler.dsbo_easyscale_worker_ready(address=worker.address)

        worker.loop.add_callback(report_ready)

    def take_dataloader(self, job_id: str):
        return self._dataloaders.get(job_id)

    def retain_dataloader(self, job_id: str, loader) -> None:
        previous = self._dataloaders.pop(job_id, None)
        if previous is not None and previous is not loader:
            previous.close_workers()
        self._dataloaders[job_id] = loader

    def close_job(self, job_id: str) -> None:
        loader = self._dataloaders.pop(job_id, None)
        if loader is not None:
            loader.close_workers()

    def teardown(self, worker) -> None:
        for loader in self._dataloaders.values():
            loader.close_workers()
        self._dataloaders.clear()


class EasyScaleSchedulerPlugin(SchedulerPlugin):
    idempotent = True

    def __init__(
        self,
        provision_limits: Mapping[str, int],
        require_runtime_ready: bool = False,
    ) -> None:
        self.provision_limits = dict(sorted(provision_limits.items()))
        self.require_runtime_ready = require_runtime_ready
        self.registry = JobRegistry()
        self.arbiter = RoundArbiter(self.provision_limits, self.registry)

        self._scheduler = None
        self._ready_workers: set[str] = set()

    async def start(self, scheduler: Any) -> None:
        self._scheduler = scheduler
        scheduler.handlers.update(
            {name: getattr(self, method) for name, method in _HANDLERS.items()}
        )

    async def close(self) -> None:
        if self._scheduler is not None:
            for name in _HANDLERS:
                self._scheduler.handlers.pop(name, None)

    def restart(self, scheduler: Any) -> None:
        for job_id in list(self.registry.jobs):
            if self.registry.close(job_id):
                self.arbiter.discard(job_id)
        self.arbiter.reset()
        self._ready_workers.intersection_update(scheduler.workers)

    def remove_worker(self, scheduler: Any, worker: Any, **kwargs: Any) -> None:
        self._ready_workers.discard(getattr(worker, "address", str(worker)))

    def remove_client(self, scheduler: Any, client: str, **kwargs: Any) -> None:
        for job in self.registry.values():
            if not job.closed and job.client_id == client:
                self._close_job(job_id=job.job_id)

    def valid_workers_downscaling(self, scheduler: Any, workers: Sequence[Any]) -> list[Any]:
        total_approved = self.registry.total_approved(self.provision_limits).to_dict()
        total_live = [self._worker_resource_type(worker) for worker in scheduler.workers.values()]

        removable = Counter(total_live) - Counter(total_approved)
        if not removable:
            return []

        valid = []
        for worker in workers:
            if getattr(worker, "long_running", None):
                continue
            resource_type = self._worker_resource_type(worker)
            if removable[resource_type] > 0:
                valid.append(worker)
                removable[resource_type] -= 1
        return valid

    def _register(
        self,
        comm: Any = None,
        *,
        job_id: str,
        priority: int,
        initial_allocation: Mapping[str, int],
        client_id: str | None = None,
    ) -> dict[str, Any]:

        allocation = Allocation.parse(self.provision_limits, initial_allocation)
        if len(allocation.nonzero()) != 1:
            raise ValueError("EasyScale jobs require one resource type.")
        approved = self.registry.total_approved(self.provision_limits)
        for resource_type, count in allocation.counts.items():
            if approved.get(resource_type) + count > self.provision_limits[resource_type]:
                raise ValueError(f"initial allocation exceeds {resource_type} provision limit")

        self.registry.register(job_id, priority, allocation, client_id, time.monotonic())

        return {
            "job_id": job_id,
            "generation": 0,
            "allocation": allocation.to_dict(),
            "provision_limits": dict(self.provision_limits),
            "allocation_revision": 0,
        }

    def _submit_proposals(
        self,
        comm: Any = None,
        *,
        job_id: str,
        proposals: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        job = self.registry.get(job_id)
        if job.closed:
            raise RuntimeError(f"EasyScale job is closed: {job_id}")

        job_proposals: dict[str, Proposal] = {}
        for serialized_proposal in proposals:
            proposal = Proposal.from_dict(serialized_proposal)
            self._validate_proposal(job, proposal)
            job_proposals[proposal.proposal_id] = proposal

        job.proposals = job_proposals
        job.decision = None

        self.arbiter.notify(job_id)
        round_id = self.arbiter.round
        decisions = self.arbiter.decisions_if_ready()
        return {
            "round": round_id,
            "pending": decisions is None,
            "decisions": [] if decisions is None else decisions
        }

    def _validate_proposal(self, job: EasyScaleJobState, proposal: Proposal) -> None:
        resources = Allocation.parse(self.provision_limits, dict(proposal.resources))
        incremental = Allocation.parse(self.provision_limits, dict(proposal.incremental_resources))
        if sum(incremental.counts.values()) < 1:
            raise ValueError("a proposal must request at least one incremental resource")

        for resource_type in self.provision_limits:
            if (resources.get(resource_type) != job.approved_allocation[resource_type] + incremental.get(resource_type)):
                raise ValueError("proposal resources must equal the current allocation plus its increment")

    def _decision(
        self,
        comm: Any = None,
        *,
        job_id: str,
        round_id: int | None = None,
    ) -> dict[str, Any]:
        job = self.registry.get(job_id)
        if round_id is not None:
            if job.decision_round < round_id:
                return {"job_id": job_id, "round": round_id, "pending": True}

        decision = job.decision
        return {
            "job_id": job_id,
            "round": job.decision_round,
            "pending": False,
            "proposal_id": decision.proposal_id if decision else None,
            "allocation": (dict(decision.allocation) if decision else dict(job.approved_allocation)),
            "previous_allocation": (dict(decision.previous_allocation)
                                    if decision
                                    else dict(job.previous_allocation)),
            "score": decision.score if decision else None,
            "allocation_revision": (
                decision.allocation_revision if decision else job.allocation_revision
            ),
        }

    def _complete_generation(
        self,
        comm: Any = None,
        *,
        job_id: str,
        generation: int,
        throughput: float,
    ) -> dict[str, Any]:
        job = self.registry.get(job_id)
        if job.closed or generation != job.generation:
            raise RuntimeError(f"stale EasyScale generation: {generation}")
        
        job.generations[generation] = {"status": "success", "throughput": throughput,}
        job.generation += 1
        return {"generation": generation, "throughput": throughput}

    def _fail_generation(
        self,
        comm: Any = None,
        *,
        job_id: str,
        generation: int,
    ) -> dict[str, Any]:
        job = self.registry.get(job_id)
        if job.closed or generation != job.generation:
            raise RuntimeError(f"stale EasyScale generation: {generation}")
        
        job.generations[generation] = {"status": "failed", "throughput": None}
        job.generation += 1
        return {"generation": generation, "failed": True}

    def _fallback(
        self,
        comm: Any = None,
        *,
        job_id: str,
        allocation: Mapping[str, int],
        expected_revision: int,
    ) -> dict[str, Any]:
        restored = Allocation.parse(self.provision_limits, allocation)
        job = self.registry.revise_downscale(job_id, restored, expected_revision)
        self._notify_downscale_transition(job_id)
        
        return {
            "job_id": job_id,
            "allocation": dict(job.approved_allocation),
            "allocation_revision": job.allocation_revision,
            "reason": "fallback",
        }

    def _release_allocation(
        self,
        comm=None,
        *,
        job_id: str,
        allocation: Mapping[str, int],
        expected_revision: int,
        reason: str,
    ) -> dict:
        if reason not in {"scale_down", "final_tail", "revocation"}:
            raise ValueError(f"invalid allocation release reason: {reason}")
        
        released = Allocation.parse(self.provision_limits, allocation)
        job = self.registry.revise_downscale(job_id, released, expected_revision)
        self._notify_downscale_transition(job_id)
        return {
            "job_id": job_id,
            "allocation": dict(job.approved_allocation),
            "allocation_revision": job.allocation_revision,
            "reason": reason,
        }

    def _notify_downscale_transition(self, job_id: str) -> None:
        self.arbiter.notify(job_id)
        self.arbiter.decisions_if_ready()

    def _worker_resource_type(self, worker: Any) -> str:
        for resource_type in self.provision_limits:
            if worker.resources.get(resource_type, 0) == 1:
                return resource_type
        raise RuntimeError(f"worker {worker.address} has no EasyScale resource")

    def _capacity(self, comm: Any = None, *, job_id: str) -> dict[str, Any]:
        job = self.registry.get(job_id)
        live = {resource_type: 0 for resource_type in self.provision_limits}
        ready_live = {resource_type: 0 for resource_type in self.provision_limits}

        if self._scheduler is not None:
            for worker in self._scheduler.workers.values():
                resource_type = self._worker_resource_type(worker)
                live[resource_type] += 1
                if worker.address in self._ready_workers and worker.status == Status.running:
                    ready_live[resource_type] += 1

        approved = self.registry.total_approved(self.provision_limits).to_dict()
        return {
            "job_id": job_id,
            "live": live,
            "ready_live": ready_live,
            "approved_total": approved,
            "approved_allocation": dict(job.approved_allocation),
            "allocation_revision": job.allocation_revision,
            "ready": all(
                (ready_live if self.require_runtime_ready else live)[t] >= approved[t]
                for t in self.provision_limits
            ),
        }

    def _worker_ready(self, comm: Any = None, *, address: str) -> dict[str, Any]:
        if self._scheduler is None or address not in self._scheduler.workers:
            raise RuntimeError(f"unknown EasyScale worker: {address}")
        self._ready_workers.add(address)
        return {"address": address, "ready": True}

    def _close_job(self, comm: Any = None, *, job_id: str) -> dict[str, Any]:
        if not self.registry.close(job_id):
            job = self.registry.get(job_id)
            return {
                "job_id": job_id,
                "closed": True,
                "allocation_revision": job.allocation_revision,
            }
        
        self.arbiter.discard(job_id)
        self.arbiter.decisions_if_ready()
        job = self.registry.get(job_id)
        return {
            "job_id": job_id,
            "closed": True,
            "allocation_revision": job.allocation_revision,
        }
