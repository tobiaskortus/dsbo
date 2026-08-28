from __future__ import annotations

import time
from distributed import Client
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from dsbo.surrogates.easyscale.protocol import LeaseResponse, Proposal


def scheduler_rpc(client: Client, name: str, **kwargs) -> Any:
    method = getattr(client.scheduler, name)
    return client.sync(method, **kwargs)


@dataclass(frozen=True)
class Registration:
    job_id: str
    generation: int
    allocation: dict[str, int]
    provision_limits: dict[str, int]


@dataclass(frozen=True)
class Leases:
    allocation: dict[str, int]
    workers_by_type: dict[str, tuple[str, ...]]
    worker_physical_ranks: dict[str, int]


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
        )

    def submit_proposals(self, job_id: str, proposals: Sequence[Proposal]) -> int:
        submission = scheduler_rpc(
            self.client,
            "dsbo_easyscale_submit_proposals",
            job_id=job_id,
            proposals=[proposal.to_dict() for proposal in proposals],
        )
        return submission["round"]

    def poll_allocation(self, job_id: str, round_id: int) -> dict[str, int] | None:
        decision = scheduler_rpc(
            self.client,
            "dsbo_easyscale_decision",
            job_id=job_id,
            round_id=round_id,
        )
        if decision.get("pending", False):
            return None
        return dict(decision["allocation"])

    def wait_for_minimum_worker_lease(
        self,
        job_id: str,
        generation: int,
        preferred_allocation: Mapping[str, int],
        minimum_allocation: Mapping[str, int],
        timeout: float,
    ) -> Leases:
        deadline = time.monotonic() + timeout
        while True:
            response = scheduler_rpc(
                self.client,
                "dsbo_easyscale_lease",
                job_id=job_id,
                generation=generation,
                allocation=dict(preferred_allocation),
                minimum_allocation=dict(minimum_allocation),
            )
            leases = self._parse_leases(response)
            if leases is not None and all(
                leases.allocation.get(resource_type, 0) >= count
                for resource_type, count in minimum_allocation.items()
            ):
                return leases
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for EasyScale worker lease.")
            time.sleep(0.05)

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

    def trigger_fallback(self, job_id: str, allocation: Mapping[str, int]) -> None:
        scheduler_rpc(
            self.client,
            "dsbo_easyscale_fallback",
            job_id=job_id,
            allocation=dict(allocation),
        )

    def close_job(self, job_id: str) -> None:
        scheduler_rpc(self.client, "dsbo_easyscale_close", job_id=job_id)

    @staticmethod
    def _parse_leases(lease: LeaseResponse | None) -> Leases | None:
        if lease is None or lease["phase"] == "SIZING":
            return None
        
        grouped: dict[str, list[str]] = {}
        ranks: dict[str, int] = {}

        for worker in lease["workers"]:
            address = worker["address"]
            worker_type = worker["resource_type"]
            grouped.setdefault(worker_type, []).append(address)
            ranks[address] = worker["rank"]

        return Leases(
            allocation=dict(lease["allocation"]),
            workers_by_type={
                name: tuple(addresses) for name, addresses in grouped.items()
            },
            worker_physical_ranks=ranks,
        )
