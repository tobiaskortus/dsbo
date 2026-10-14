import pytest
import torch

from dsbo.parameters import Categorical, Constant, Continuous, Group, Integer, Vector
from dsbo.parameters.types import _collect, _collect_vectors


def test_continuous_from_unit():
    out = Continuous(-2.0, 3.0).from_unit(torch.tensor([[0.0], [0.5], [1.0]]))
    assert torch.allclose(out, torch.tensor([[-2.0], [0.5], [3.0]]))


def test_integer_from_unit_bins():
    p = Integer(1, 6)
    midpoints = ((torch.arange(6) + 0.5) / 6).unsqueeze(1)
    assert p.from_unit(midpoints).flatten().tolist() == [1, 2, 3, 4, 5, 6]
    assert p.from_unit(torch.tensor([[0.0], [1.0]])).flatten().tolist() == [1, 6]


@pytest.mark.parametrize(("value", "expected"), [(2.6, 3), (2.4, 2), (2.5, 2), (3.5, 4)])
def test_integer_decode_rounds(value, expected):
    p = Integer(0, 10)
    decoded = p.decode(torch.tensor([value]))
    assert decoded == expected and isinstance(decoded, int)


def test_categorical_encode_unknown():
    p = Categorical(["a", "b", "c"])
    assert p.encode("b") == [0.0, 1.0, 0.0]
    with pytest.raises(ValueError):
        p.encode("d")


def test_categorical_from_unit():
    p = Categorical(["a", "b", "c"])
    out = p.from_unit(torch.tensor([[0.0], [1 / 3], [0.4], [2 / 3], [0.8], [1.0]]))
    assert torch.equal(out.sum(dim=1), torch.ones(6))
    assert out.argmax(dim=1).tolist() == [0, 1, 1, 2, 2, 2]
    assert p.decode(torch.tensor([0.1, 0.7, 0.2])) == "b"


def test_collect_flattens():
    params = {
        "x": Continuous(0.0, 1.0),
        "g": Group(inner=Continuous(0.0, 1.0), deep=Group(deeper=Integer(0, 1))),
        "v": Vector(0.0, 2.0, size=2),
        "k": Constant(7),
    }
    flat = _collect(params)
    assert list(flat) == ["x", "g.inner", "g.deep.deeper", "v_1", "v_2", "k"]
    assert flat["v_1"] == Continuous(0.0, 2.0)
    assert flat["k"] is params["k"]


def test_collect_key_fn():
    assert list(_collect({"g": Group(inner=Continuous(0.0, 1.0))}, key_fn="_".join)) == ["g_inner"]


def test_collect_vectors():
    params = {"g": Group(w=Vector(0.0, 1.0, size=2)), "x": Continuous(0.0, 1.0)}
    spec, flat_keys = _collect_vectors(params)["g.w"]
    assert spec is params["g"].params["w"]
    assert flat_keys == ["g.w_1", "g.w_2"]
