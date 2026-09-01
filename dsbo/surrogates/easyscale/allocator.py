from __future__ import annotations

from distributed import Client
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from dsbo.surrogates.easyscale.protocol import Proposal


def scheduler_rpc(client: Client, name: str, **kwargs) -> Any:
    method = getattr(client.scheduler, name)
    return client.sync(method, **kwargs)


@dataclass(frozen=True)
class Registration:
    job_id: str
    generation: int
    allocation: dict[str, int]
    provision_limits: dict[str, int]
    allocation_revision: int


@dataclass(frozen=True)
class AllocationDecision:
    allocation: dict[str, int]
    allocation_revision: int


@dataclass(frozen=True)
class Capacity:
    live: dict[str, int]
    approved_total: dict[str, int]
    approved_allocation: dict[str, int]
    allocation_revision: int
    ready: bool
    ready_live: dict[str, int] = field(default_factory=dict, kw_only=True)


class ClientResourceAllocator:
    def __init__(self, client: Client) -> None:
        if client is None:
            raise TypeError("client is required.")
        self.client = client

    def register_job(self, job_id: str, priority: int, initial_allocation: Mapping[str, int]) -> Registration:
        response = scheduler_rpc(
            self.client,
            "dsbo_easyscale_register",
            job_id=job_id,
            priority=priority,
            initial_allocation=dict(initial_allocation),
            client_id=getattr(self.client, "id", None),
        )
        return Registration(
            job_id=job_id,
            generation=int(response.get("generation", 0)),
            allocation=dict(response["allocation"]),
            provision_limits=dict(response["provision_limits"]),
            allocation_revision=int(response["allocation_revision"]),
        )

    def submit_proposals(self, job_id: str, proposals: Sequence[Proposal]) -> int:
        submission = scheduler_rpc(
            self.client,
            "dsbo_easyscale_submit_proposals",
            job_id=job_id,
            proposals=[proposal.to_dict() for proposal in proposals],
        )
        return submission["round"]

    def poll_allocation(self, job_id: str, round_id: int) -> AllocationDecision | None:
        decision = scheduler_rpc(
            self.client,
            "dsbo_easyscale_decision",
            job_id=job_id,
            round_id=round_id,
        )
        if decision.get("pending", False):
            return None
        return AllocationDecision(
            allocation=dict(decision["allocation"]),
            allocation_revision=int(decision["allocation_revision"]),
        )

    def complete_generation(self, job_id: str, generation: int, throughput: float) -> None:
        scheduler_rpc(
            self.client,
            "dsbo_easyscale_complete_generation",
            job_id=job_id,
            generation=generation,
            throughput=throughput,
        )

    def fail_generation(self, job_id: str, generation: int) -> None:
        scheduler_rpc(
            self.client,
            "dsbo_easyscale_fail_generation",
            job_id=job_id,
            generation=generation,
        )

    def trigger_fallback(
        self,
        job_id: str,
        allocation: Mapping[str, int],
        expected_revision: int,
    ) -> AllocationDecision:
        response = scheduler_rpc(
            self.client,
            "dsbo_easyscale_fallback",
            job_id=job_id,
            allocation=dict(allocation),
            expected_revision=expected_revision,
        )
        return AllocationDecision(
            allocation=dict(response["allocation"]),
            allocation_revision=int(response["allocation_revision"]),
        )

    def release_allocation(
        self,
        job_id: str,
        allocation: Mapping[str, int],
        expected_revision: int,
        reason: str,
    ) -> AllocationDecision:
        response = scheduler_rpc(
            self.client,
            "dsbo_easyscale_release_allocation",
            job_id=job_id,
            allocation=dict(allocation),
            expected_revision=expected_revision,
            reason=reason,
        )
        return AllocationDecision(
            allocation=dict(response["allocation"]),
            allocation_revision=int(response["allocation_revision"]),
        )

    def capacity(self, job_id: str) -> Capacity:
        response = scheduler_rpc(
            self.client,
            "dsbo_easyscale_capacity",
            job_id=job_id,
        )
        return Capacity(
            live=dict(response["live"]),
            ready_live=dict(response["ready_live"]),
            approved_total=dict(response["approved_total"]),
            approved_allocation=dict(response["approved_allocation"]),
            allocation_revision=int(response["allocation_revision"]),
            ready=bool(response["ready"]),
        )

    def close_job(self, job_id: str) -> None:
        scheduler_rpc(self.client, "dsbo_easyscale_close", job_id=job_id)
