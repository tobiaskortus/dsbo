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

    def maybe_get(self, job_id: str) -> EasyScaleJobState | None:
        return self.jobs.get(job_id)

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
        job.closed = True
        return True

    def values(self):
        return self.jobs.values()

    def items(self):
        return self.jobs.items()

    def total_approved(self, provision_limits: Mapping[str, int]) -> Allocation:
        total = Allocation.zero(provision_limits)
        for job in self.jobs.values():
            if job.closed:
                continue
            total = total.add(Allocation(provision_limits, job.approved_allocation))
        return total
