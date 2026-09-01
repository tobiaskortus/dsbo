import time
from dataclasses import dataclass, field
from typing import Any

from dsbo.surrogates.easyscale.protocol import Proposal


@dataclass(frozen=True)
class EasyScaleDecision:
    proposal_id: str
    allocation: dict[str, int]
    previous_allocation: dict[str, int]
    score: float
    decided_at: float
    allocation_revision: int


@dataclass
class EasyScaleJobState:
    job_id: str
    priority: int
    approved_allocation: dict[str, int]
    previous_allocation: dict[str, int]
    proposals: dict[str, dict] = field(default_factory=dict)
    decision: EasyScaleDecision | None = None
    generation: int = 0
    generations: dict[int, dict[str, Any]] = field(default_factory=dict)
    closed: bool = False
    registered_at: float = 0.0
    client_id: str | None = None
    decision_round: int = -1
    allocation_revision: int = 0


def grant_allocation(candidate_job: EasyScaleJobState, proposal: Proposal) -> EasyScaleDecision:
    previous = dict(candidate_job.approved_allocation)
    candidate_job.previous_allocation = previous
    candidate_job.approved_allocation = dict(proposal.resources)
    candidate_job.allocation_revision += 1
    decision = EasyScaleDecision(
        proposal_id=proposal.proposal_id,
        allocation=dict(proposal.resources),
        previous_allocation=previous,
        score=proposal.proposal_score,
        decided_at=time.monotonic(),
        allocation_revision=candidate_job.allocation_revision,
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
        "allocation_revision": decision.allocation_revision,
    }
