import pytest
from unittest.mock import patch, MagicMock

from dsbo.modules.dsl import DependencyDict, Module

def test_rshift_modules():
    a = Module()
    b = Module()

    dep_expr = a >> b

    assert isinstance(dep_expr, dict)
    assert dep_expr == {a: [b]}

def test_rshift_invalid():
    a = Module()

    with pytest.raises(TypeError):
        _ = a >> "not a module"

    with pytest.raises(TypeError):
        _ = a >> 123

def test_mult_modules():
    a = Module()

    dep_expr = a * 3

    assert isinstance(dep_expr, dict)

    replicas = list(dep_expr.keys())
    assert len(replicas) == 3
    for i, module in enumerate(replicas):
        assert isinstance(module, Module)
        assert module.iter == i
        assert dep_expr[module] == []

def test_mult_modules_invalid():
    a = Module()

    with pytest.raises(TypeError):
        _ = a * "not an int"

    with pytest.raises(TypeError):
        _ = a * 3.5


def test_or_modules():
    a = Module()
    b = Module()

    dep_exp = a | b

    assert isinstance(dep_exp, dict)
    assert dep_exp == {a: [], b: []}


def test_or_rshift():
    a = Module()
    b = Module()
    c = Module()
    d = Module()

    dep_exp = (a >> b) | (c >> d)

    assert isinstance(dep_exp, dict)
    assert dep_exp == {a: [b], c: [d]}


def test_or_rshift_merge():
    a = Module(); b = Module()
    c = Module(); d = Module()
    e = Module()

    dep_exp = ((a >> b) | (c >> d)) >> e

    assert isinstance(dep_exp, dict)
    assert dep_exp == {a: [b], c: [d], b: [e], d: [e]}


def test_or_mul_rshift_merge():
    a = Module(); b = Module()
    c = Module(); d = Module()
    e = Module()

    a2 = a * 2
    b2 = b * 2
    c2 = c * 2
    d2 = d * 2

    dep_exp = ((a2 >> b2) | (c2 >> d2)) >> e

    b_replicas = [x for vs in b2.values() for x in vs]
    d_replicas = [x for vs in d2.values() for x in vs]

    for node in b_replicas + d_replicas:
        assert node in dep_exp
        assert dep_exp[node] == [e]
    # Ensure no original (non-replica) module was left pointing to `e` —
    # that would indicate an extra intermediate node was inserted by the DSL.
    for orig in (a, b, c, d):
        assert not (orig in dep_exp and dep_exp[orig] == [e])
            