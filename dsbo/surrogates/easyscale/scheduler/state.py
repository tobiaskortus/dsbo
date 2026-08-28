import time
from dataclasses import dataclass, field
from typing import Any

from dsbo.surrogates.easyscale.protocol import LeaseResponse, Proposal


@dataclass(frozen=True)
class EasyScaleDecision:
    proposal_id: str
    allocation: dict[str, int]
    previous_allocation: dict[str, int]
    score: float
    decided_at: float


@dataclass
class GenerationLease:
    generation: int
    allocation: dict[str, int]
    workers: tuple[tuple[str, str], ...] = ()
    throughput: float | None = None
    failed: bool = False
    released: bool = False
    created_at: float = 0.0


@dataclass
class EasyScaleJobState:
    job_id: str
    priority: int
    approved_allocation: dict[str, int]
    previous_allocation: dict[str, int]
    proposals: dict[str, dict] = field(default_factory=dict)
    decision: EasyScaleDecision | None = None
    generation: int = -1
    leases: dict[int, GenerationLease] = field(default_factory=dict)
    closed: bool = False
    registered_at: float = 0.0
    client_id: str | None = None
    decision_round: int = -1


def grant_allocation(candidate_job: EasyScaleJobState, proposal: Proposal) -> EasyScaleDecision:
    previous = dict(candidate_job.approved_allocation)
    candidate_job.previous_allocation = previous
    candidate_job.approved_allocation = dict(proposal.resources)
    decision = EasyScaleDecision(
        proposal_id=proposal.proposal_id,
        allocation=dict(proposal.resources),
        previous_allocation=previous,
        score=proposal.proposal_score,
        decided_at=time.monotonic(),
    )
    candidate_job.decision = decision
    return decision


def decision_reply(candidate_job: EasyScaleJobState, decision: EasyScaleDecision) -> dict[str, Any]:
    return {
        "job_id": candidate_job.job_id,
        "proposal_id": decision.proposal_id,
        "allocation": dict(decision.allocation),
        "previous_allocation": dict(decision.previous_allocation),
        "score": decision.score,
    }


def has_active_generation(job: EasyScaleJobState) -> bool:
    return any(
        lease.generation == job.generation
        and not lease.released
        and not lease.failed
        for lease in job.leases.values()
    )


def lease_response(lease: GenerationLease) -> LeaseResponse:
    return {
        "phase": "LEASED",
        "allocation": dict(lease.allocation),
        "workers": [
            {"address": address, "rank": rank, "resource_type": resource_type}
            for rank, (address, resource_type) in enumerate(lease.workers)
        ],
    }
