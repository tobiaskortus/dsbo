from __future__ import annotations

import math
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TypedDict


ResourceVector = tuple[tuple[str, int], ...]
AllocationVector = tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class EasyScaleSchedulerConfig:
    proposal_count: int = 3
    profile_bias_threshold: float = 0.2
    allocation_timeout: float = 300.0
    proposal_increment: int = 1
    capabilities: Mapping[str, float] = field(default_factory=dict)
    allow_cpu: bool = False


class LeaseWorkerPayload(TypedDict):
    address: str
    rank: int
    resource_type: str


class SizingLeaseResponse(TypedDict):
    phase: Literal["SIZING"]
    workers: list[LeaseWorkerPayload]


class ActiveLeaseResponse(TypedDict):
    phase: Literal["LEASED"]
    allocation: dict[str, int]
    workers: list[LeaseWorkerPayload]


LeaseResponse = SizingLeaseResponse | ActiveLeaseResponse


def freeze(vector: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(vector.items()))


def calculate_est_count(resources: ResourceVector, allocation: AllocationVector,) -> int:
    resource_counts = dict(resources)
    return sum(
        resource_counts[resource_type] * assigned_count
        for resource_type, assigned_count in allocation
    )

def calculate_total_estimate_count(plan: ESTPlan) -> int:
    return sum(count for _, count in plan.resource_vector)

def is_same_config(a: ESTPlan, b: ESTPlan) -> bool:
        return a.resource_vector == b.resource_vector and a.allocation == b.allocation


def calculate_overload(allocation: AllocationVector, capabilities: Mapping[str, float]) -> float:
    return max(
        assigned_count / capabilities[resource_type]
        for resource_type, assigned_count in allocation
    )


def calculate_waste(
    allocation: AllocationVector,
    capabilities: Mapping[str, float],
    n_est: int,
    max_p: int,
    f_overload: float,
) -> float:
    underused_capability = sum(
        capabilities[resource_type] - assigned_count / f_overload
        for resource_type, assigned_count in allocation
    )
    overprovisioned_est = (n_est - max_p) / f_overload
    return underused_capability + overprovisioned_est


def calculate_throughput(
    resources: ResourceVector,
    capabilities: Mapping[str, float],
    waste: float,
) -> float:
    raw_throughput = sum(
        resource_count * capabilities[resource_type]
        for resource_type, resource_count in resources
    )
    return raw_throughput - waste


@dataclass(frozen=True)
class ESTPlan:
    resource_vector: ResourceVector
    allocation: AllocationVector
    n_est: int
    f_overload: float
    waste: float
    estimated_throughput: float
    measured_throughput: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "resource_vector", freeze(self.resource_vector))
        object.__setattr__(self, "allocation", freeze(self.allocation))

    @property
    def resources(self) -> dict[str, int]:
        return dict(self.resource_vector)

    @property
    def throughput(self) -> float:
        return self.estimated_throughput


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    resource_vector: ResourceVector
    incremental_resources: ResourceVector
    allocation: AllocationVector
    estimated_throughput: float
    proposal_score: float
    estimated_speedup: float

    def __post_init__(self) -> None:
        if not isinstance(self.proposal_id, str) or not self.proposal_id:
            raise TypeError("proposal_id must be a non-empty string.")
        
        object.__setattr__(self, "resource_vector", freeze(self.resource_vector))
        object.__setattr__(self, "incremental_resources", freeze(self.incremental_resources))
        object.__setattr__(self, "allocation", freeze(self.allocation))

    @property
    def incremental_gpu_count(self) -> int:
        return sum(count for _, count in self.incremental_resources)

    @property
    def resources(self) -> dict[str, int]:
        return dict(self.resource_vector)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "resources": dict(self.resource_vector),
            "incremental_resources": dict(self.incremental_resources),
            "allocation": dict(self.allocation),
            "estimated_throughput": self.estimated_throughput,
            "estimated_speedup": self.estimated_speedup,
            "score": self.proposal_score,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Proposal":
        return cls(
            proposal_id=payload["proposal_id"],
            resource_vector=payload["resources"],
            incremental_resources=payload["incremental_resources"],
            allocation=payload["allocation"],
            estimated_throughput=payload["estimated_throughput"],
            proposal_score=payload["score"],
            estimated_speedup=payload["estimated_speedup"],
        )


class PlanDatabase:
    def __init__(self, max_p: int, capabilities: Mapping[str, float], config: EasyScaleSchedulerConfig) -> None:
        self.max_p = max_p
        self.config = config
        self.capabilities = dict(capabilities)
        self.current_plan: ESTPlan | None = None
        self.previous_plan: ESTPlan | None = None
        self._profiled_throughputs: dict[tuple[ResourceVector, AllocationVector], float] = {}

    def set_capability(self, resource_type: str, capability: float) -> None:
        self.capabilities[resource_type] = capability

    def enumerate_plans(self, resource_vector: Mapping[str, int]) -> tuple[ESTPlan, ...]:
        resources = freeze(resource_vector)
        missing = [name for name, _ in resources if name not in self.capabilities]
        if missing:
            raise ValueError(
                "No EasyScale capability profile for resource type(s): "
                + ", ".join(missing)
            )

        plans = []
        resource_types = tuple(name for name, _ in resources)

        for assigned_counts in itertools.product(range(1, self.max_p + 1), repeat=len(resources)):
            allocation = tuple(zip(resource_types, assigned_counts))
            n_est = calculate_est_count(resources, allocation)
            if n_est < self.max_p:
                continue

            f_overload = calculate_overload(allocation, self.capabilities)
            waste = calculate_waste(allocation, self.capabilities, n_est, self.max_p, f_overload)
            analytical_throughput = calculate_throughput(resources, self.capabilities, waste)

            # A profiled throughput corrects the analytical estimate when model bias is too high.
            profiled_throughput = self._profiled_throughputs.get((resources, allocation))
            
            estimated_throughput = (
                analytical_throughput
                if profiled_throughput is None
                else profiled_throughput
            )

            plans.append(
                ESTPlan(
                    resource_vector=dict(resources),
                    allocation=dict(allocation),
                    n_est=n_est,
                    f_overload=f_overload,
                    waste=waste,
                    estimated_throughput=estimated_throughput,
                )
            )

        return tuple(plans)

    def select_plan(self, resource_vector: Mapping[str, int]) -> ESTPlan:
        plans = self.enumerate_plans(resource_vector)

        if not plans:
            raise ValueError("resource_vector cannot host max_p ESTs.")

        #pick the plan with the highest estimated_throughput
        return min(plans, key=lambda plan: (-plan.estimated_throughput, plan.allocation))

    def propose(self, current_plan: ESTPlan, max_resource_vector: Mapping[str, int]) -> tuple[Proposal, ...]:
        maximum = dict(max_resource_vector)
        current = dict(current_plan.resource_vector)
        current_throughput = current_plan.estimated_throughput
        if current_throughput <= 0:
            raise ValueError("current plan throughput must be > 0 for proposals.")
        
        proposals = []
        for resource_type in sorted(maximum):
            if resource_type not in self.capabilities:
                continue

            available = maximum[resource_type] - current.get(resource_type, 0)
            for increment in range(self.config.proposal_increment, available + 1, self.config.proposal_increment):
                candidate_resources = dict(current)
                candidate_resources[resource_type] = (candidate_resources.get(resource_type, 0) + increment)
                if sum(candidate_resources.values()) > self.max_p:
                    continue

                candidate = self.select_plan(candidate_resources)
                score = (candidate.estimated_throughput - current_throughput) / increment
                if score <= 0:
                    continue

                proposals.append(
                    Proposal(
                        proposal_id=f"{resource_type}+{increment}",
                        resource_vector=dict(candidate.resource_vector),
                        incremental_resources={resource_type: increment},
                        allocation=dict(candidate.allocation),
                        estimated_throughput=candidate.estimated_throughput,
                        proposal_score=score,
                        estimated_speedup=(
                            candidate.estimated_throughput / current_throughput
                        ),
                    )
                )

        proposals.sort(
            key=lambda proposal: (
                -proposal.proposal_score,
                -proposal.incremental_gpu_count,
                proposal.resource_vector,
                proposal.allocation,
                proposal.proposal_id,
            )
        )
        
        return tuple(proposals[: self.config.proposal_count])

    def adopt_plan(self, plan: ESTPlan) -> None:
        if self.current_plan and is_same_config(self.current_plan, plan):
            return
        self.previous_plan = self.current_plan
        self.current_plan = plan

    def update_profile(self, plan: ESTPlan, measured_throughput: float) -> ESTPlan:
        if not math.isfinite(measured_throughput) or measured_throughput <= 0:
            raise ValueError("measured_throughput must be finite and > 0.")

        estimated = plan.estimated_throughput
        bias = math.inf if estimated == 0 else abs(measured_throughput - estimated) / estimated
        if bias > self.config.profile_bias_threshold:
            self._profiled_throughputs[(plan.resource_vector, plan.allocation)] = measured_throughput
            estimated = measured_throughput

        updated = replace(
            plan,
            resource_vector=dict(plan.resource_vector),
            allocation=dict(plan.allocation),
            estimated_throughput=estimated,
            measured_throughput=measured_throughput,
        )
        self._refresh_stored(plan, updated)
        return updated

    def _refresh_stored(self, plan: ESTPlan, updated: ESTPlan) -> None:
        if self.current_plan is plan:
            self.current_plan = updated
        if self.previous_plan is plan:
            self.previous_plan = updated

    def fallback_if_slowdown(self, measured_throughput: float) -> ESTPlan | None:
        if self.current_plan is None:
            raise RuntimeError("No current EasyScale plan is committed.")
        
        current = self.update_profile(self.current_plan, measured_throughput)
        previous = self.previous_plan
        if previous is None:
            return None
        
        previous_throughput = previous.measured_throughput or previous.estimated_throughput
        scaled_out = calculate_total_estimate_count(current) > calculate_total_estimate_count(previous)

        if not scaled_out or measured_throughput >= previous_throughput:
            return None
        
        self.current_plan = previous
        self.previous_plan = current
        return previous


def assign_virtual_ranks(
    max_p: int,
    workers_by_type: Mapping[str, Sequence[str]],
    plan: ESTPlan | Mapping[str, int],
) -> dict[str, tuple[int, ...]]:
    
    allocation = dict(plan.allocation if isinstance(plan, ESTPlan) else plan)

    workers: list[tuple[str, str, int]] = []
    for resource_type in sorted(workers_by_type):
        for address in sorted(workers_by_type[resource_type]):
            workers.append((resource_type, address, allocation[resource_type]))

    if len(workers) > max_p:
        raise ValueError("physical worker count must not exceed max_p.")
    if sum(capacity for _, _, capacity in workers) < max_p:
        raise ValueError("plan allocation does not have capacity for every EST.")

    assigned: dict[str, list[int]] = {address: [] for _, address, _ in workers}
    rank = 0
    for _, address, capacity in workers:
        assigned[address].append(rank)
        rank += 1
    while rank < max_p:
        for _, address, capacity in workers:
            if len(assigned[address]) < capacity:
                assigned[address].append(rank)
                rank += 1
                if rank == max_p:
                    break

    return {address: tuple(values) for address, values in assigned.items()}
