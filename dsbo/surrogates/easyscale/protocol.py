from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

@dataclass(frozen=True)
class EasyScaleSchedulerConfig:
    proposal_count: int = 3
    profile_bias_threshold: float = 0.2
    allocation_timeout: float = 300.0
    proposal_increment: int = 1
    control_interval_steps: int = 25
    capabilities: Mapping[str, float] = field(default_factory=dict)


def calculate_est_count(resource_count: int, assigned_count: int) -> int:
    return resource_count * assigned_count

def calculate_total_estimate_count(plan: ESTPlan) -> int:
    return plan.resource_count

def is_same_config(a: ESTPlan, b: ESTPlan) -> bool:
    return (
        a.resource_type == b.resource_type
        and a.resource_count == b.resource_count
        and a.assigned_count == b.assigned_count
    )


def calculate_overload(assigned_count: int, capability: float) -> float:
    return assigned_count / capability


def calculate_waste(
    assigned_count: int,
    capability: float,
    n_est: int,
    max_p: int,
    f_overload: float,
) -> float:
    underused_capability = capability - assigned_count / f_overload
    overprovisioned_est = (n_est - max_p) / f_overload
    return underused_capability + overprovisioned_est


def calculate_throughput(
    resource_count: int,
    capability: float,
    waste: float,
) -> float:
    return resource_count * capability - waste


@dataclass(frozen=True)
class ESTPlan:
    resource_type: str
    resource_count: int
    assigned_count: int
    n_est: int
    f_overload: float
    waste: float
    estimated_throughput: float
    measured_throughput: float | None = None

@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    resource_type: str
    resource_count: int
    incremental_resource_count: int
    assigned_count: int
    estimated_throughput: float
    proposal_score: float
    estimated_speedup: float

    def __post_init__(self) -> None:
        if not isinstance(self.proposal_id, str) or not self.proposal_id:
            raise TypeError("proposal_id must be a non-empty string.")
        
    @property
    def resources(self) -> dict[str, int]:
        return {self.resource_type: self.resource_count}

    @property
    def incremental_resources(self) -> dict[str, int]:
        return {self.resource_type: self.incremental_resource_count}

    @property
    def allocation(self) -> dict[str, int]:
        return {self.resource_type: self.assigned_count}

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "resources": self.resources,
            "incremental_resources": self.incremental_resources,
            "allocation": self.allocation,
            "estimated_throughput": self.estimated_throughput,
            "estimated_speedup": self.estimated_speedup,
            "score": self.proposal_score,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Proposal":
        resources = {name: count for name, count in payload["resources"].items() if count}
        incremental = {
            name: count for name, count in payload["incremental_resources"].items() if count
        }
        allocation = {name: count for name, count in payload["allocation"].items() if count}
        if not (len(resources) == len(incremental) == len(allocation) == 1):
            raise ValueError("EasyScale proposals require one resource type.")
        resource_type, resource_count = next(iter(resources.items()))
        if set(incremental) != {resource_type} or set(allocation) != {resource_type}:
            raise ValueError("EasyScale proposal resource types must match.")
        return cls(
            proposal_id=payload["proposal_id"],
            resource_type=resource_type,
            resource_count=resource_count,
            incremental_resource_count=incremental[resource_type],
            assigned_count=allocation[resource_type],
            estimated_throughput=payload["estimated_throughput"],
            proposal_score=payload["score"],
            estimated_speedup=payload["estimated_speedup"],
        )


class PlanDatabase:
    def __init__(self, max_p: int, resource_type: str, config: EasyScaleSchedulerConfig) -> None:
        self.max_p = max_p
        self.resource_type = resource_type
        self.config = config
        self.capability = config.capabilities.get(resource_type)
        self.current_plan: ESTPlan | None = None
        self.previous_plan: ESTPlan | None = None
        self._profiled_throughputs: dict[tuple[int, int], float] = {}

    def set_capability(self, capability: float) -> None:
        self.capability = capability

    def enumerate_plans(self, resource_count: int) -> tuple[ESTPlan, ...]:
        if resource_count < 1:
            raise ValueError("resource_count must be >= 1.")
        if self.capability is None:
            raise ValueError(
                f"No EasyScale capability profile for resource type: {self.resource_type}"
            )

        plans = []
        for assigned_count in range(1, self.max_p + 1):
            n_est = calculate_est_count(resource_count, assigned_count)
            if n_est < self.max_p:
                continue

            f_overload = calculate_overload(assigned_count, self.capability)
            waste = calculate_waste(
                assigned_count, self.capability, n_est, self.max_p, f_overload
            )
            analytical_throughput = calculate_throughput(
                resource_count, self.capability, waste
            )

            # A profiled throughput corrects the analytical estimate when model bias is too high.
            profiled_throughput = self._profiled_throughputs.get(
                (resource_count, assigned_count)
            )
            
            estimated_throughput = (
                analytical_throughput
                if profiled_throughput is None
                else profiled_throughput
            )

            plans.append(
                ESTPlan(
                    resource_type=self.resource_type,
                    resource_count=resource_count,
                    assigned_count=assigned_count,
                    n_est=n_est,
                    f_overload=f_overload,
                    waste=waste,
                    estimated_throughput=estimated_throughput,
                )
            )

        return tuple(plans)

    def select_plan(self, resource_count: int) -> ESTPlan:
        plans = self.enumerate_plans(resource_count)

        if not plans:
            raise ValueError("resource_count cannot host max_p ESTs.")

        return min(
            plans,
            key=lambda plan: (-plan.estimated_throughput, plan.assigned_count),
        )

    def initial_plan(self, resource_count: int) -> ESTPlan:
        assigned_count = math.ceil(self.max_p / resource_count)
        n_est = calculate_est_count(resource_count, assigned_count)
        f_overload = calculate_overload(assigned_count, 1.0)
        waste = calculate_waste(assigned_count, 1.0, n_est, self.max_p, f_overload)
        estimated_throughput = calculate_throughput(resource_count, 1.0, waste)
        return ESTPlan(
            resource_type=self.resource_type,
            resource_count=resource_count,
            assigned_count=assigned_count,
            n_est=n_est,
            f_overload=f_overload,
            waste=waste,
            estimated_throughput=estimated_throughput,
        )

    def propose(self, current_plan: ESTPlan, max_resource_count: int) -> tuple[Proposal, ...]:
        current_throughput = current_plan.estimated_throughput
        if current_throughput <= 0:
            raise ValueError("current plan throughput must be > 0 for proposals.")

        proposals = []
        available = max_resource_count - current_plan.resource_count
        for increment in range(
            self.config.proposal_increment,
            available + 1,
            self.config.proposal_increment,
        ):
            resource_count = current_plan.resource_count + increment
            if resource_count > self.max_p:
                continue

            candidate = self.select_plan(resource_count)
            score = (candidate.estimated_throughput - current_throughput) / increment
            if score <= 0:
                continue

            proposals.append(
                Proposal(
                    proposal_id=f"{self.resource_type}+{increment}",
                    resource_type=self.resource_type,
                    resource_count=candidate.resource_count,
                    incremental_resource_count=increment,
                    assigned_count=candidate.assigned_count,
                    estimated_throughput=candidate.estimated_throughput,
                    proposal_score=score,
                    estimated_speedup=(candidate.estimated_throughput / current_throughput),
                )
            )

        proposals.sort(
            key=lambda proposal: (
                -proposal.proposal_score,
                -proposal.incremental_resource_count,
                proposal.resource_count,
                proposal.assigned_count,
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
            self._profiled_throughputs[
                (plan.resource_count, plan.assigned_count)
            ] = measured_throughput
            estimated = measured_throughput

        updated = replace(
            plan,
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
    workers: Sequence[str],
    plan: ESTPlan | int,
) -> dict[str, tuple[int, ...]]:
    assigned_count = plan.assigned_count if isinstance(plan, ESTPlan) else plan
    addresses = sorted(workers)

    if len(addresses) > max_p:
        raise ValueError("physical worker count must not exceed max_p.")
    if len(addresses) * assigned_count < max_p:
        raise ValueError("plan allocation does not have capacity for every EST.")

    assigned: dict[str, list[int]] = {address: [] for address in addresses}
    rank = 0
    for address in addresses:
        assigned[address].append(rank)
        rank += 1
    while rank < max_p:
        for address in addresses:
            if len(assigned[address]) < assigned_count:
                assigned[address].append(rank)
                rank += 1
                if rank == max_p:
                    break

    return {address: tuple(values) for address, values in assigned.items()}
