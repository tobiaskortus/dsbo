from collections.abc import Mapping


class WorkerPool:
    def __init__(self, provision_limits: Mapping[str, int]) -> None:
        self.provision_limits = dict(provision_limits)
        self.arrived: dict[str, str] = {}
        self.leases: dict[str, tuple[str, int]] = {}

    def record(self, address: str, worker) -> None:
        positive_types = sorted(
            resource_type
            for resource_type, value in (getattr(worker, "resources", None) or {}).items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 0
        )
        if len(positive_types) == 1 and positive_types[0] in self.provision_limits:
            self.arrived[address] = positive_types[0]
        else:
            self.arrived.pop(address, None)

    def remove(self, address: str) -> tuple[str, int] | None:
        self.arrived.pop(address, None)
        return self.leases.pop(address, None)

    def available(self, resource_type: str) -> list[str]:
        return sorted(
            address
            for address, arrived_type in self.arrived.items()
            if arrived_type == resource_type and address not in self.leases
        )

    def is_leased(self, address: str) -> bool:
        return address in self.leases

    def grant(self, job_id: str, generation: int, workers) -> None:
        for address, _ in workers:
            self.leases[address] = (job_id, generation)

    def release(self, workers) -> None:
        for address, _ in workers:
            self.leases.pop(address, None)

    def summary(self) -> dict[str, tuple[str, int]]:
        return dict(self.leases)

    def clear(self) -> None:
        self.arrived.clear()
        self.leases.clear()
