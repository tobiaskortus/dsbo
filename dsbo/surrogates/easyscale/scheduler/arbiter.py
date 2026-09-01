from collections.abc import Mapping
from typing import Any

from dsbo.surrogates.easyscale.protocol import Proposal
from dsbo.surrogates.easyscale.scheduler.allocation import Allocation
from dsbo.surrogates.easyscale.scheduler.registry import JobRegistry
from dsbo.surrogates.easyscale.scheduler.state import (
    EasyScaleDecision,
    EasyScaleJobState,
    decision_reply,
    grant_allocation,
)


def ranked_candidates(
    jobs: Mapping[str, EasyScaleJobState],
) -> list[tuple[EasyScaleJobState, Proposal]]:
    candidates = [
        (candidate_job, proposal)
        for candidate_job in jobs.values()
        if not candidate_job.closed
        for proposal in candidate_job.proposals.values()
    ]
    return sorted(
        candidates,
        key=lambda item: (
            -item[1].proposal_score,
            -item[1].incremental_resource_count,
            -item[0].priority,
            item[0].job_id,
            item[1].proposal_id,
        ),
    )


class RoundArbiter:
    def __init__(self, provision_limits: Mapping[str, int], registry: JobRegistry) -> None:
        self.provision_limits = provision_limits
        self.registry = registry
        self.round = 0
        self.submitted: set[str] = set()

    def notify(self, job_id: str) -> None:
        self.submitted.add(job_id)

    def discard(self, job_id: str) -> None:
        self.submitted.discard(job_id)

    def reset(self) -> None:
        self.submitted.clear()

    def decisions_if_ready(self) -> list[dict[str, Any]] | None:
        """Arbitrate once every open job has submitted; otherwise signal pending."""
        open_jobs = {
            candidate_job.job_id
            for candidate_job in self.registry.values()
            if not candidate_job.closed
        }
        if open_jobs and open_jobs.issubset(self.submitted):
            return self._arbitrate_round()
        return None

    def _free_resources(self) -> Allocation:
        return Allocation(self.provision_limits, self.provision_limits).subtract(
            self.registry.total_approved(self.provision_limits)
        )

    def _arbitrate_round(self) -> list[dict[str, Any]]:
        free = self._free_resources()
        decided: list[tuple[EasyScaleJobState, EasyScaleDecision]] = []
        granted_jobs: set[str] = set()
        for candidate_job, proposal in ranked_candidates(self.registry.jobs):
            if candidate_job.job_id in granted_jobs:
                continue
            incremental = Allocation.parse(
                self.provision_limits, dict(proposal.incremental_resources)
            )
            if not free.covers(incremental.counts):
                continue
            free = free.subtract(incremental)
            decision = grant_allocation(candidate_job, proposal)
            decided.append((candidate_job, decision))
            granted_jobs.add(candidate_job.job_id)

        for candidate_job in self.registry.values():
            if candidate_job.closed:
                continue
            candidate_job.proposals.clear()
            candidate_job.decision_round = self.round
        self.submitted.clear()
        self.round += 1
        return [decision_reply(job, decision) for job, decision in decided]
