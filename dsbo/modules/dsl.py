from __future__ import annotations
from copy import deepcopy
import math
from typing import Iterable, Mapping



class DependencyDict(dict):
    def __init__(self, mapping: dict = {}):
        super().__init__(mapping)

    def __or__(self, other):
        match other:
            case dict() as d:
                out = DependencyDict({k: list(v) for k, v in self.items()})
                for k, vs in d.items():
                    out[k] = list(dict.fromkeys(out.get(k, []) + list(vs)))
                return out
            case _:
                raise TypeError(f"Unsupported operand type(s) for |: 'DependencyDict' and '{type(other).__name__}'")

    def __rshift__(self, other):
        from collections.abc import Iterable
        from collections.abc import Mapping

        def merge(a, b):
            return list(dict.fromkeys(a + list(b)))

        result = {k: list(v) for k, v in self.items()}

        def _leaves(mapping) -> set:
            keys = set(mapping.keys())
            vals = set(x for vs in mapping.values() for x in vs)
            return vals - keys or keys

        self_leaves = _leaves(self)

        def _normalize(rhs):
            if hasattr(rhs, '__rrshift__') and not isinstance(rhs, DependencyDict):
                cand = rhs.__rrshift__(list(self_leaves))
                if isinstance(cand, Mapping):
                    return dict(cand)

            match rhs:
                case Mapping() as m:
                    return dict(m)
                case Iterable() as seq if not isinstance(seq, (str, bytes)):
                    return {None: list(seq)}
                case _:
                    return {None: [rhs]}

        rhs_map = _normalize(other)

        def _has_replicas(m: Mapping) -> bool:
            return any(isinstance(getattr(k, 'iter', None), int) for k in m.keys())

        if _has_replicas(self) and _has_replicas(rhs_map):
            rhs_keys = {k for k in rhs_map.keys() if isinstance(getattr(k, 'iter', None), int)}
            if not rhs_keys.issubset(self_leaves):
                raise TypeError(
                    "Ambiguous replicated-to-replicated operation: use dynamic() for the "
                    "second expression (e.g. `a * 2 >> b * dynamic()`) to indicate a 1:1 mapping."
                )

        # if rhs_map uses the None key it's an explicit list-of-targets
        if None in rhs_map:
            targets = rhs_map.pop(None)
            for leaf in self_leaves:
                result[leaf] = merge(result.get(leaf, []), targets)

        # If resolver returned explicit left->targets mapping (keys overlap our leaves),
        # merge per-leaf targets 
        if rhs_map and any(k in self_leaves for k in rhs_map.keys()):
            for leaf in self_leaves:
                if leaf in rhs_map:
                    result[leaf] = merge(result.get(leaf, []), rhs_map[leaf])
            for k, vs in rhs_map.items():
                if k not in self_leaves:
                    result[k] = merge(result.get(k, []), vs)
            return DependencyDict(result)

        if rhs_map:
            other_vals = set(x for vs in rhs_map.values() for x in vs)
            other_roots = set(rhs_map.keys()) - other_vals
            if not other_roots:
                other_roots = set(rhs_map.keys())

            for leaf in self_leaves:
                result[leaf] = merge(result.get(leaf, []), other_roots)

            for k, vs in rhs_map.items():
                result[k] = merge(result.get(k, []), vs)

        return DependencyDict(result)


class Dynamic:
    def __init__(self):
        self.module: "Module" | None = None

    def __rrshift__(self, left_leaves):
        left_list = sorted(left_leaves, key=lambda x: getattr(x, 'i', 0))
        mapping = {}
        for i, left in enumerate(left_list):
            idx = getattr(left, 'i', i)
            if self.module is None:
                raise TypeError("Dynamic not bound to a module")
            mapping[left] = [self.module.iter_copy(idx)]
        return mapping


def dynamic():
    return Dynamic()


class Module:
    def __init__(self, name: str = None):
        self.name = name

    def __call__(self, *args, **kwds):
        pass

    def iter_copy(self, i: int) -> Module:
        module = deepcopy(self)
        module.iter = i
        return module
    
    def __or__(self: Module, other: Module | DependencyDict) -> DependencyDict:
        match other:
            case Module():
                return DependencyDict({self: [], other: []})
            case dict() as d:  # includes DependencyDict
                merged = {k: list(dict.fromkeys(vs)) for k, vs in d.items()}
                merged.setdefault(self, [])
                return DependencyDict(merged)
            case _:
                raise TypeError(f"Unsupported operand type(s) for |: 'Module' and '{type(other).__name__}'")

    def __rshift__(self: "Module", other) -> DependencyDict:
        match other:
            case Module():
                return DependencyDict({self: [other]})
            case dict() as d:
                vals = {v for vs in d.values() for v in vs}
                roots = [k for k in d.keys() if k not in vals and k is not self]
                if not roots:
                    roots = [k for k in d.keys() if k is not self]

                merged = {k: list(dict.fromkeys(vs)) for k, vs in d.items()}
                merged.setdefault(self, [])
                merged[self] = list(dict.fromkeys(merged[self] + roots))
                return DependencyDict(merged)
            case _:
                raise TypeError(f"Unsupported operand type(s) for >>: 'Module' and '{type(other).__name__}'")
            
    def __mul__(self: Module, other: int | Dynamic) -> object:
        match other:
            case int():
                reps = [self.iter_copy(i) for i in range(other)]
                return DependencyDict({r: [] for r in reps})
            case Dynamic():
                other.module = self
                return other
            case _:
                raise TypeError(f"Unsupported operand type(s) for *: 'Module' and '{type(other).__name__}'")
            