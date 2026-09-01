from collections.abc import Mapping

from dsbo.surrogates.easyscale.scheduler.allocation import Allocation
from dsbo.surrogates.easyscale.scheduler.state import EasyScaleJobState


class JobRegistry:
    def __init__(self) -> None:
        self.jobs: dict[str, EasyScaleJobState] = {}

    def get(self, job_id: str) -> EasyScaleJobState:
        try:
            return self.jobs[job_id]
        except KeyError:
            raise KeyError(f"unknown EasyScale job: {job_id}") from None

    def register(self, job_id: str, priority: int, allocation: Allocation, client_id: str | None, now: float) -> None:
        self.jobs[job_id] = EasyScaleJobState(
            job_id=job_id,
            priority=priority,
            approved_allocation=allocation.to_dict(),
            previous_allocation=allocation.to_dict(),
            registered_at=now,
            client_id=client_id,
        )

    def close(self, job_id: str) -> bool:
        """Reset a job's state and mark it closed; return True if newly closed."""
        job = self.jobs[job_id]
        if job.closed:
            return False
        job.proposals.clear()
        job.decision = None
        job.approved_allocation = {
            resource_type: 0 for resource_type in job.approved_allocation
        }
        job.allocation_revision += 1
        job.closed = True
        return True

    def revise_downscale(self, job_id: str, allocation: Allocation, expected_revision: int) -> EasyScaleJobState:
        job = self.get(job_id)
        if job.closed:
            raise RuntimeError(f"EasyScale job is closed: {job_id}")
        
        if expected_revision != job.allocation_revision:
            raise RuntimeError(
                f"stale EasyScale allocation revision: {expected_revision}; "
                f"current revision is {job.allocation_revision}"
            )
        
        for resource_type, count in allocation.counts.items():
            if count > job.approved_allocation[resource_type]:
                raise ValueError("new allocation exceeds the approved allocation")

        previous = dict(job.approved_allocation)
        job.approved_allocation = allocation.to_dict()
        job.previous_allocation = previous
        job.allocation_revision += 1
        job.proposals.clear()
        job.decision = None
        return job

    def values(self):
        return self.jobs.values()

    def total_approved(self, provision_limits: Mapping[str, int]) -> Allocation:
        total = Allocation.zero(provision_limits)
        for job in self.jobs.values():
            if job.closed:
                continue
            total = total.add(Allocation(provision_limits, job.approved_allocation))
        return total
