import time
import math
from collections.abc import Mapping, Sequence
from typing import Any
from distributed.diagnostics.plugin import SchedulerPlugin

from dsbo.surrogates.easyscale.protocol import LeaseResponse, Proposal
from dsbo.surrogates.easyscale.scheduler.allocation import Allocation
from dsbo.surrogates.easyscale.scheduler.arbiter import RoundArbiter
from dsbo.surrogates.easyscale.scheduler.registry import JobRegistry
from dsbo.surrogates.easyscale.scheduler.state import (
    GenerationLease,
    has_active_generation,
    lease_response,
)
from dsbo.surrogates.easyscale.scheduler.worker_pool import WorkerPool


EASYSCALE_PLUGIN_NAME = "dsbo-easyscale"

_HANDLERS = {
    "dsbo_easyscale_register": "_register",
    "dsbo_easyscale_submit_proposals": "_submit_proposals",
    "dsbo_easyscale_decision": "_decision",
    "dsbo_easyscale_lease": "_lease",
    "dsbo_easyscale_complete_generation": "_complete_generation",
    "dsbo_easyscale_fail_generation": "_fail_generation",
    "dsbo_easyscale_fallback": "_fallback",
    "dsbo_easyscale_close": "_close_job",
}


class EasyScaleSchedulerPlugin(SchedulerPlugin):
    idempotent = True

    def __init__(self, provision_limits: Mapping[str, int]) -> None:
        self.provision_limits = dict(sorted(provision_limits.items()))
        self.registry = JobRegistry()
        self.pool = WorkerPool(self.provision_limits)
        self.arbiter = RoundArbiter(self.provision_limits, self.registry)
        self.jobs = self.registry.jobs
        self._scheduler = None

    async def start(self, scheduler) -> None:
        self._scheduler = scheduler
        self._sync_workers(scheduler)
        scheduler.handlers.update(
            {name: getattr(self, method) for name, method in _HANDLERS.items()}
        )

    async def close(self) -> None:
        if self._scheduler is not None:
            for name in _HANDLERS:
                self._scheduler.handlers.pop(name, None)

    def restart(self, scheduler) -> None:
        for job_id, job in list(self.registry.items()):
            if self.registry.close(job_id):
                self.arbiter.discard(job_id)
                self._release_job_leases(job)
        self._sync_workers(scheduler)
        self.arbiter.reset()

    def add_worker(self, scheduler, worker: str, **kwargs) -> None:
        state = scheduler.workers.get(worker)
        if state is not None:
            self.pool.record(worker, state)

    def remove_worker(self, scheduler, worker: str, **kwargs) -> None:
        assignment = self.pool.remove(worker)
        if assignment is None:
            return
        job_id, generation = assignment
        job = self.registry.maybe_get(job_id)
        lease = job.leases.get(generation) if job is not None else None
        if lease is not None and not lease.released:
            lease.failed = True

    def remove_client(self, scheduler, client: str, **kwargs) -> None:
        for job in self.registry.values():
            if not job.closed and job.client_id == client:
                self._close_job(job_id=job.job_id)

    def valid_workers_downscaling(self, scheduler, workers):
        return [worker for worker in workers if not self.pool.is_leased(worker.address)]

    def resource_targets(self) -> dict[str, int]:
        return self.registry.total_approved(self.provision_limits).nonzero()
    
    def adaptive_targets(self, ordinary: Mapping[str, int]) -> dict[str, int]:
        targets = dict(ordinary)
        for resource_type, count in self.registry.total_approved(self.provision_limits).counts.items():
            if count:
                targets[resource_type] = targets.get(resource_type, 0) + count
        return targets

    def lease_summary(self) -> dict[str, tuple[str, int]]:
        return self.pool.summary()

    def _register(
        self,
        comm=None,
        *,
        job_id: str,
        priority: int,
        initial_allocation: Mapping[str, int],
        client_id: str | None = None,
    ) -> dict:

        allocation = Allocation.parse(self.provision_limits, initial_allocation)
        approved = self.registry.total_approved(self.provision_limits)
        for resource_type, count in allocation.counts.items():
            if approved.get(resource_type) + count > self.provision_limits[resource_type]:
                raise ValueError(f"initial allocation exceeds {resource_type} provision limit")

        self.registry.register(job_id, priority, allocation, client_id, time.monotonic())

        return {
            "job_id": job_id,
            "allocation": allocation.to_dict(),
            "provision_limits": dict(self.provision_limits),
        }

    def _submit_proposals(
        self,
        comm=None,
        *,
        job_id: str,
        proposals: Sequence[dict[str, Any]],
    ) -> dict:
        job = self.registry.get(job_id)

        job_proposals: dict[str, Proposal] = {}
        for payload in proposals:
            proposal = Proposal.from_dict(payload)
            if proposal.proposal_id in job_proposals:
                raise ValueError(f"duplicate proposal_id: {proposal.proposal_id}")
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

    def _validate_proposal(self, job, proposal: Proposal) -> None:
        resources = Allocation.parse(self.provision_limits, dict(proposal.resources))
        incremental = Allocation.parse(self.provision_limits, dict(proposal.incremental_resources))
        if sum(incremental.counts.values()) < 1:
            raise ValueError("a proposal must request at least one incremental resource")

        for resource_type in self.provision_limits:
            if (resources.get(resource_type) != job.approved_allocation[resource_type] + incremental.get(resource_type)):
                raise ValueError("proposal resources must equal the current allocation plus its increment")

    def _decision(
        self,
        comm=None,
        *,
        job_id: str,
        round_id: int | None = None,
    ) -> dict:
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
        }

    def _lease(
        self,
        comm=None,
        *,
        job_id: str,
        generation: int,
        allocation: Mapping[str, int],
        minimum_allocation: Mapping[str, int] | None = None,
    ) -> LeaseResponse:
        job = self.registry.get(job_id)
        if job.closed:
            raise RuntimeError(f"EasyScale job is closed: {job_id}")

        preferred = Allocation.parse(self.provision_limits, allocation)
        minimum = Allocation.parse(self.provision_limits,
            allocation if minimum_allocation is None else minimum_allocation,
        )

        for resource_type in self.provision_limits:
            if preferred.get(resource_type) > job.approved_allocation[resource_type]:
                raise ValueError("lease allocation exceeds the approved job allocation")
            if minimum.get(resource_type) > preferred.get(resource_type):
                raise ValueError("minimum allocation exceeds the preferred allocation")

        lease = job.leases.get(generation)
        if lease is not None and (lease.released or lease.failed):
            raise RuntimeError(f"stale EasyScale generation: {generation}")
        if lease is not None:
            return lease_response(lease)
        if generation != job.generation + 1:
            raise RuntimeError(f"stale EasyScale generation: {generation}")
        if has_active_generation(job):
            raise RuntimeError("the previous EasyScale generation is still active")

        selected: list[tuple[str, str]] = []
        selected_allocation: dict[str, int] = {}
        for resource_type, count in sorted(preferred.counts.items()):
            available = self.pool.available(resource_type)
            selected_count = min(count, len(available))
            if selected_count < minimum.get(resource_type):
                return {"phase": "SIZING", "workers": []}
            if selected_count:
                selected_allocation[resource_type] = selected_count
                selected.extend(
                    (address, resource_type)
                    for address in available[:selected_count]
                )

        lease = GenerationLease(
            generation=generation,
            allocation=selected_allocation,
            workers=tuple(selected),
            created_at=time.monotonic(),
        )
        job.generation = generation
        job.leases[generation] = lease
        self.pool.grant(job_id, generation, selected)
        return lease_response(lease)

    def _complete_generation(
        self,
        comm=None,
        *,
        job_id: str,
        generation: int,
        throughput: float,
    ) -> dict:
        job = self.registry.get(job_id)
        lease = job.leases.get(generation)
        if (lease is None or lease.released or lease.failed):
            raise RuntimeError(f"stale or failed EasyScale generation: {generation}")
        
        lease.throughput = throughput
        self._release_lease(lease)
        return {"generation": generation, "throughput": throughput}

    def _fail_generation(
        self,
        comm=None,
        *,
        job_id: str,
        generation: int,
    ) -> dict:
        job = self.registry.get(job_id)
        lease = job.leases.get(generation)
        if job.closed or generation != job.generation or lease is None:
            raise RuntimeError(f"stale EasyScale generation: {generation}")
        if lease.released and not lease.failed:
            raise RuntimeError(f"EasyScale generation is already closed: {generation}")
        lease.failed = True
        if not lease.released:
            self._release_lease(lease)
        return {"generation": generation, "failed": True}

    def _fallback(
        self,
        comm=None,
        *,
        job_id: str,
        allocation: Mapping[str, int],
    ) -> dict:
        job = self.registry.get(job_id)
        if job.closed:
            raise RuntimeError(f"EasyScale job is closed: {job_id}")
        
        restored = Allocation.parse(self.provision_limits, allocation)
        for resource_type, count in restored.counts.items():
            if count > job.approved_allocation[resource_type]:
                raise ValueError("fallback allocation exceeds the approved allocation")
            
        previous = dict(job.approved_allocation)
        job.approved_allocation = restored.to_dict()
        job.previous_allocation = previous
        job.proposals.clear()
        job.decision = None
        self.arbiter.notify(job_id)
        self.arbiter.decisions_if_ready()

        return {"job_id": job_id, "allocation": restored.to_dict()}

    def _close_job(self, comm=None, *, job_id: str) -> dict:
        if not self.registry.close(job_id):
            return {"job_id": job_id, "closed": True}
        
        self.arbiter.discard(job_id)
        self._release_job_leases(self.registry.get(job_id))
        self.arbiter.decisions_if_ready()
        return {"job_id": job_id, "closed": True}

    def _sync_workers(self, scheduler) -> None:
        self.pool.clear()
        for address, worker in scheduler.workers.items():
            self.pool.record(address, worker)

    def _release_job_leases(self, job) -> None:
        for lease in job.leases.values():
            if not lease.released:
                self._release_lease(lease)

    def _release_lease(self, lease: GenerationLease) -> None:
        lease.released = True
        self.pool.release(lease.workers)
