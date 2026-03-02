from __future__ import annotations
from copy import deepcopy
from typing import Iterable, Mapping


class ModuleDslMixin:
    i: int | None

    def replicate(self, i: int) -> ModuleDslMixin:
        """Return a deep copy of this module tagged with replication index *i*."""
        copy = deepcopy(self)
        copy.i = i
        return copy

    def __or__(self, other):
        match other:
            case ModuleDslMixin():
                return DependencyDict({self: [], other: []})
            case dict() as d:
                merged = {k: list(dict.fromkeys(vs)) for k, vs in d.items()}
                merged.setdefault(self, [])
                return DependencyDict(merged)
            case _:
                raise TypeError(f"Unsupported operand type(s) for |: '{type(self).__name__}' and '{type(other).__name__}'")

    def __rshift__(self, other):
        match other:
            case ModuleDslMixin():
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
                raise TypeError(f"Unsupported operand type(s) for >>: '{type(self).__name__}' and '{type(other).__name__}'")

    def __mul__(self, other):
        match other:
            case int():
                reps = [self.replicate(i) for i in range(other)]
                return DependencyDict({r: [] for r in reps})
            case Dynamic():
                other.module = self
                return other
            case _:
                raise TypeError(f"Unsupported operand type(s) for *: '{type(self).__name__}' and '{type(other).__name__}'")


class DependencyDict(dict):
    def __init__(self, mapping: dict = {}):
        assert all(
            isinstance(k, ModuleDslMixin)
            and isinstance(v, list)
            and all(isinstance(item, ModuleDslMixin) for item in v)
            for k, v in mapping.items()
        ), "DependencyDict keys must be Modules and values must be lists of Modules"

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

    def __mul__(self, other):
        if isinstance(other, Dynamic):
            other.subgraph = self
            return other

        if isinstance(other, int):
            all_nodes = list(set(self.keys()) | set(x for vs in self.values() for x in vs))
            out = {}
            for i in range(other):
                copied = {orig: orig.replicate(i) for orig in all_nodes}
                for orig_k, orig_vs in self.items():
                    out[copied[orig_k]] = [copied[v] for v in orig_vs]
            return DependencyDict(out)

        raise TypeError(f"Unsupported operand type(s) for *: 'DependencyDict' and '{type(other).__name__}'")

    def __rshift__(self, other):
        from collections.abc import Iterable
        from collections.abc import Mapping

        def merge(a, b):
            return list(dict.fromkeys(a + list(b)))

        result = {k: list(v) for k, v in self.items()}

        def _leaves(mapping) -> set:
            all_nodes = set(mapping.keys()) | set(x for vs in mapping.values() for x in vs)
            has_out = {k for k, v in mapping.items() if v}
            return (all_nodes - has_out) or all_nodes

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
            return any(isinstance(k.i, int) for k in m.keys() if k is not None)

        if _has_replicas(self) and _has_replicas(rhs_map):
            is_subgraph_dynamic = isinstance(other, Dynamic) and getattr(other, 'subgraph', None) is not None
            if not is_subgraph_dynamic and isinstance(other, DependencyDict):
                raise TypeError(
                    "Ambiguous replicated-to-replicated operation: both sides use integer replication. "
                    "Use `dynamic()` on the RHS (e.g. `a * n >> b * dynamic()`) to indicate a 1:1 mapping, "
                    "or replicate a subgraph (e.g. `(subgraph) * n`)."
                )
            if not is_subgraph_dynamic:
                rhs_keys = {k for k in rhs_map.keys() if isinstance(k.i, int)}
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
        self.module: ModuleDslMixin | None = None
        self.subgraph: DependencyDict | None = None

    def __rrshift__(self, left_leaves):
        left_list = sorted(left_leaves, key=lambda x: x.i or 0)

        # If bound to a single Module (Module * dynamic()), produce
        # left -> [replica] mapping.
        if self.module is not None and self.subgraph is None:
            mapping = {}
            for i, left in enumerate(left_list):
                idx = left.i if left.i is not None else i
                if self.module is None:
                    raise TypeError("Dynamic not bound to a module")
                mapping[left] = [self.module.replicate(idx)]
            return mapping

        elif self.subgraph is not None and self.module is None:
            mapping = {}

            # get roots of the subgraph
            sub_keys = list(self.subgraph.keys())
            sub_vals = set(x for vs in self.subgraph.values() for x in vs)
            roots = [k for k in sub_keys if k not in sub_vals]
            if not roots:
                roots = sub_keys

            for i, left in enumerate(left_list):
                idx = left.i if left.i is not None else i

                all_nodes = list(set(sub_keys) | set(x for vs in self.subgraph.values() for x in vs))
                inner_iters = [n.i for n in all_nodes]
                max_inner = max((it for it in inner_iters if isinstance(it, int)), default=-1)
                stride = max_inner + 1 if max_inner >= 0 else 1
                copied = {}
                for orig in all_nodes:
                    orig_i = orig.i
                    if isinstance(orig_i, int):
                        new_i = idx * stride + orig_i
                    else:
                        new_i = idx
                    copied[orig] = orig.replicate(new_i)

                for orig_k, orig_vs in self.subgraph.items():
                    k_copy = copied[orig_k]
                    mapping[k_copy] = [copied[v] for v in orig_vs]

                mapping[left] = [copied[r] for r in roots]
            return mapping

        raise TypeError("Dynamic resolver is not bound to a Module or subgraph")


def dynamic():
    return Dynamic()
            