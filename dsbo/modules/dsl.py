from __future__ import annotations

import weakref
from copy import deepcopy
from typing import Iterable, Mapping, overload


class ModuleDslMixin:
    i: int | None
    _replica_counts: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def replicate(self, i: int) -> ModuleDslMixin:
        """Shallow copy of object with updated .i for repeated elements. 
        Distingushiable from the original by .i index, but share any mutable sub-objects.
        Requires stateless module definitions.
        TODO: implement stateful modules with (optional) mutable variables."""
        cls = type(self)
        copy = object.__new__(cls)
        copy.__dict__.update(self.__dict__)
        copy.i = i
        return copy

    def __or__(self, other) -> DependencyDict:
        match other:
            case ModuleDslMixin():
                return DependencyDict({self: [], other: []})
            case dict() as d:
                merged = {k: list(dict.fromkeys(vs)) for k, vs in d.items()}
                merged.setdefault(self, [])
                return DependencyDict(merged)
            case _:
                raise TypeError(f"Unsupported operand type(s) for |: '{type(self).__name__}' and '{type(other).__name__}'")

    def __rshift__(self, other) -> DependencyDict:
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

    @overload
    def __mul__(self, other: int) -> DependencyDict: ...

    @overload
    def __mul__(self, other: Dynamic) -> Dynamic: ...

    def __mul__(self, other: int | Dynamic) -> DependencyDict | Dynamic:
        match other:
            case int():
                start = ModuleDslMixin._replica_counts.get(self, 0)
                reps = [self.replicate(start + i) for i in range(other)]
                ModuleDslMixin._replica_counts[self] = start + other
                dd = DependencyDict({r: [] for r in reps})
                dd.groups = [DependencyGroup(
                    template=self,
                    count=other,
                    node_to_template={r: self for r in reps},
                )]
                return dd
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
        self.groups: list[DependencyGroup] = []

    @property
    def template_to_nodes(self) -> dict[object, list[ModuleDslMixin]]:
        out: dict[object, list[ModuleDslMixin]] = {}
        for g in self.groups:
            for tmpl, replicas in g.template_to_nodes.items():
                out.setdefault(tmpl, []).extend(replicas)
        return out

    @property
    def surrogate_modules(self) -> list[ModuleDslMixin]:
        return [g.template for g in self.groups
                if isinstance(g.template, ModuleDslMixin)
                and getattr(g.template, 'surrogate', None) is not None]

    @staticmethod
    def _combine_groups(a: list[DependencyGroup], b: list[DependencyGroup]) -> list[DependencyGroup]:
        return list(a) + list(b)

    def __or__(self, other):
        match other:
            case dict() as d:
                out = DependencyDict({k: list(v) for k, v in self.items()})
                for k, vs in d.items():
                    out[k] = list(dict.fromkeys(out.get(k, []) + list(vs)))
                out.groups = DependencyDict._combine_groups(
                    self.groups,
                    d.groups if isinstance(d, DependencyDict) else [],
                )
                return out
            case ModuleDslMixin():
                return self | DependencyDict({other: []})
            case _:
                raise TypeError(f"Unsupported operand type(s) for |: 'DependencyDict' and '{type(other).__name__}'")

    def __mul__(self, other):
        if isinstance(other, Dynamic):
            other.subgraph = self
            return other

        if isinstance(other, int):
            all_nodes = list(set(self.keys()) | set(x for vs in self.values() for x in vs))
            max_inner = max((n.i for n in all_nodes if isinstance(getattr(n, 'i', None), int)), default=-1)
            stride = max_inner + 1 if max_inner >= 0 else 1
            out = {}
            node_to_template: dict[ModuleDslMixin, ModuleDslMixin] = {}
            for outer_i in range(other):
                copied = {
                    orig: orig.replicate(outer_i * stride + orig.i if isinstance(getattr(orig, 'i', None), int) else outer_i)
                    for orig in all_nodes
                }
                node_to_template.update({copy: orig for orig, copy in copied.items()})
                for orig_k, orig_vs in self.items():
                    out[copied[orig_k]] = [copied[v] for v in orig_vs]
            dd = DependencyDict(out)
            dd.groups = list(self.groups) + [DependencyGroup(
                template=self,
                count=other,
                node_to_template=node_to_template,
            )]
            return dd

        raise TypeError(f"Unsupported operand type(s) for *: 'DependencyDict' and '{type(other).__name__}'")

    def __rshift__(self, other: ModuleDslMixin | DependencyDict | Dynamic) -> DependencyDict:
        from collections.abc import Iterable
        from collections.abc import Mapping

        def merge(a, b):
            return list(dict.fromkeys(a + list(b)))

        result = {k: list(v) for k, v in self.items()}
        rhs_groups = other.groups if isinstance(other, DependencyDict) else []

        def _make_depdict(d: dict) -> DependencyDict:
            dd = DependencyDict(d)
            dd.groups = DependencyDict._combine_groups(self.groups, rhs_groups)
            return dd

        def _leaves(mapping) -> set:
            all_nodes = set(mapping.keys()) | set(x for vs in mapping.values() for x in vs)
            has_out = {k for k, v in mapping.items() if v}
            return (all_nodes - has_out) or all_nodes

        def _roots(mapping) -> set:
            vals = set(x for vs in mapping.values() for x in vs)
            roots = set(mapping.keys()) - vals
            return roots or set(mapping.keys())

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

        # --- Auto 1:1 pairing by replica index ---
        # When both the leaves AND the RHS roots are replicated with matching
        # count and the two sets are disjoint, pair them 1:1 by .i index.
        # This makes  A * n >> B * n  work without requiring dynamic().
        def _replicated_sorted(nodes):
            return sorted(
                [n for n in nodes if isinstance(getattr(n, 'i', None), int)],
                key=lambda x: getattr(x, 'i', 0),
            )

        rhs_roots_set = _roots(rhs_map) if None not in rhs_map else set()
        rep_leaves    = _replicated_sorted(self_leaves)
        rep_rhs_roots = _replicated_sorted(rhs_roots_set)

        if (rep_leaves and rep_rhs_roots
                and len(rep_leaves) == len(rep_rhs_roots)
                and not rhs_roots_set.intersection(self_leaves)):
            for leaf, rhs_root in zip(rep_leaves, rep_rhs_roots):
                result[leaf] = merge(result.get(leaf, []), [rhs_root])
            for k, vs in rhs_map.items():
                result.setdefault(k, [])
                if vs:
                    result[k] = merge(result[k], vs)
            return _make_depdict(result)

        # --- Explicit target list (None key) ---
        if None in rhs_map:
            targets = rhs_map.pop(None)
            for leaf in self_leaves:
                result[leaf] = merge(result.get(leaf, []), targets)

        if rhs_map and any(k in self_leaves for k in rhs_map.keys()):
            for leaf in self_leaves:
                if leaf in rhs_map:
                    result[leaf] = merge(result.get(leaf, []), rhs_map[leaf])
            for k, vs in rhs_map.items():
                if k not in self_leaves:
                    result[k] = merge(result.get(k, []), vs)
            return _make_depdict(result)


        if rhs_map:
            other_vals = set(x for vs in rhs_map.values() for x in vs)
            other_roots = set(rhs_map.keys()) - other_vals
            if not other_roots:
                other_roots = set(rhs_map.keys())

            for leaf in self_leaves:
                result[leaf] = merge(result.get(leaf, []), other_roots)

            for k, vs in rhs_map.items():
                result[k] = merge(result.get(k, []), vs)

        return _make_depdict(result)


class Dynamic:
    def __init__(self):
        self.module: ModuleDslMixin | None = None
        self.subgraph: DependencyDict | None = None

    def __or__(self, other):
        if not isinstance(other, Dynamic):
            return NotImplemented

        def _as_subgraph(d):
            if d.module is not None and d.subgraph is None:
                return DependencyDict({d.module: []})
            if d.subgraph is not None and d.module is None:
                return d.subgraph
            raise TypeError("Cannot combine unbound Dynamic instances with |")

        combined = Dynamic()
        combined.subgraph = _as_subgraph(self) | _as_subgraph(other)
        return combined

    def __rrshift__(self, left_leaves):
        left_list = sorted(left_leaves, key=lambda x: getattr(x, 'i', 0) or 0)

        # If bound to a single Module (Module * dynamic()), produce
        # left -> [replica] mapping.
        if self.module is not None and self.subgraph is None:
            mapping = {}
            for i, left in enumerate(left_list):
                idx = getattr(left, 'i', None) if getattr(left, 'i', None) is not None else i
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
                idx = getattr(left, 'i', None) if getattr(left, 'i', None) is not None else i

                all_nodes = list(set(sub_keys) | set(x for vs in self.subgraph.values() for x in vs))
                inner_iters = [getattr(n, 'i', None) for n in all_nodes]
                max_inner = max((it for it in inner_iters if isinstance(it, int)), default=-1)
                stride = max_inner + 1 if max_inner >= 0 else 1
                copied = {}
                for orig in all_nodes:
                    orig_i = getattr(orig, 'i', None)
                    if isinstance(orig_i, int):
                        new_i = idx * stride + orig_i
                    else:
                        new_i = idx
                    copied[orig] = orig.replicate(new_i)

                # Patch any copied submodule's _parent to point to the
                # corresponding copied parent (if present).
                for orig in all_nodes:
                    copy_obj = copied[orig]
                    parent = getattr(orig, '_parent', None)
                    if parent is not None and parent in copied:
                        object.__setattr__(copy_obj, '_parent', copied[parent])

                for orig_k, orig_vs in self.subgraph.items():
                    k_copy = copied[orig_k]
                    mapping[k_copy] = [copied[v] for v in orig_vs]

                mapping[left] = [copied[r] for r in roots]
            return mapping

        raise TypeError("Dynamic resolver is not bound to a Module or subgraph")


def dynamic():
    return Dynamic()


class DependencyGroup:
    def __init__(
        self,
        template: ModuleDslMixin | DependencyDict,
        count: int,
        node_to_template: dict[ModuleDslMixin, ModuleDslMixin],
    ) -> None:
        self.count = count
        self.template = template
        self.nodes = frozenset(node_to_template.keys())
        self.node_to_template = node_to_template
        self.template_to_nodes = self._invert_node_to_template(node_to_template)

    def _invert_node_to_template(self, node_to_template: dict[ModuleDslMixin, ModuleDslMixin]) -> dict[ModuleDslMixin, list[ModuleDslMixin]]:
        template_to_nodes: dict[ModuleDslMixin, list[ModuleDslMixin]] = {}
        for node, tmpl in node_to_template.items():
            template_to_nodes.setdefault(tmpl, []).append(node)
        return template_to_nodes