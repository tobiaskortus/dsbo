from collections.abc import Mapping


class Allocation:
    """A validated resource-count vector over fixed provision limits."""
    def __init__(self, limits: Mapping[str, int], counts: Mapping[str, int] | None = None) -> None:
        self.limits = dict(sorted(limits.items()))
        self.counts = {
            resource_type: (dict(counts or {}).get(resource_type, 0))
            for resource_type in self.limits
        }

    @classmethod
    def zero(cls, limits: Mapping[str, int]) -> "Allocation":
        return cls(limits)

    @classmethod
    def parse(cls, limits: Mapping[str, int], mapping: Mapping[str, int]) -> "Allocation":
        for resource_type in mapping:
            if resource_type not in limits:
                raise ValueError(f"unknown resource type: {resource_type}")
        return cls(limits, mapping)

    def get(self, resource_type: str, default: int = 0) -> int:
        return self.counts.get(resource_type, default)

    def add(self, other: "Allocation") -> "Allocation":
        return Allocation(
            self.limits,
            {t: self.counts[t] + other.counts[t] for t in self.limits},
        )

    def subtract(self, other: "Allocation") -> "Allocation":
        return Allocation(
            self.limits,
            {t: self.counts[t] - other.counts[t] for t in self.limits},
        )

    def covers(self, other: Mapping[str, int]) -> bool:
        return all(self.counts[t] >= other.get(t, 0) for t in self.limits)

    def to_dict(self) -> dict[str, int]:
        return dict(self.counts)

    def nonzero(self) -> dict[str, int]:
        return {t: c for t, c in self.counts.items() if c}
