import re

import pytest
import torch
import torch.autograd.forward_ad as fw

from dsbo.parameters import (Candidate, Categorical, Constant, Continuous,
                             Encoding, Group, Integer, ParameterSpace, UniformSampler, Vector)


def full_space():
    return ParameterSpace(
        x=Continuous(-2.0, 3.0),
        n=Integer(1, 6),
        m=Categorical(["a", "b", "c"]),
        v=Vector(0.0, 1.0, size=2),
        g=Group(inner=Continuous(10.0, 20.0)),
        k=Constant(42),
    )


def sample_values():
    return {"x": 1.5, "n": 4, "m": "b", "v_1": 0.1, "v_2": 0.9, "g.inner": 12.0}


def test_iter_flat_order():
    assert [name for name, _ in full_space()] == ["x", "n", "m", "v_1", "v_2", "g.inner", "k"]


def test_repeat_expands_groups():
    space = ParameterSpace(**ParameterSpace.repeat("layer", 2, w=Continuous(0.0, 1.0)))
    assert [name for name, _ in space] == ["layer_0.w", "layer_1.w"]


@pytest.mark.parametrize(("encoding", "include_constants", "expected"), [
    (Encoding.ORDINAL, False, 6), ("ordinal", True, 7),
    (Encoding.ONE_HOT, False, 8), ("one_hot", True, 9),
])
def test_dims_width(encoding, include_constants, expected):
    assert full_space().dims(encoding, include_constants=include_constants) == expected


def test_dims_unknown_encoding():
    with pytest.raises(ValueError, match="Unknown encoding"):
        full_space().dims("unknown")


def test_encode_order_skips_constants():
    t = full_space().encode(sample_values())
    assert torch.allclose(t, torch.tensor([1.5, 4.0, 0.0, 1.0, 0.0, 0.1, 0.9, 12.0]))


def test_encode_expands_vectors():
    t = full_space().encode({**sample_values(), "v": [0.25, 0.75]})
    assert t[5:7].tolist() == [0.25, 0.75]


def test_encode_unknown_key():
    with pytest.raises(KeyError):
        full_space().encode({"x": 1.0})


def test_unit_to_design_tensor():
    space = full_space()
    t = space.unit_to_design_tensor(torch.zeros(2, space.dims()))
    assert t.shape == (2, space.dims(Encoding.ONE_HOT))
    v = Candidate(space, t[0]).detach().values
    assert v["x"] == pytest.approx(-2.0) and v["n"] == 1 and v["m"] == "a"
    assert v["g.inner"] == pytest.approx(10.0)


def test_unit_to_design_tensor_1d_and_constant_only():
    space = full_space()
    assert space.unit_to_design_tensor(torch.zeros(space.dims())).shape == (1, 8)
    assert ParameterSpace(k=Constant(1)).unit_to_design_tensor(torch.zeros(3, 0)).shape == (3, 0)


def test_sample_without_sampler():
    space = full_space()
    with pytest.raises(RuntimeError, match="No sampler"):
        space.sample()
    with pytest.raises(RuntimeError, match="No sampler"):
        space.sample_batch(batch_size=4)


def test_sample_valid_design():
    space = full_space()
    space.register_sampler(UniformSampler(seed=0))
    v = space.sample().detach().values
    assert -2.0 <= float(v["x"]) <= 3.0
    assert isinstance(v["n"], int) and 1 <= v["n"] <= 6
    assert v["m"] in ("a", "b", "c") and v["k"] == 42
    assert torch.all((v["v"] >= 0.0) & (v["v"] <= 1.0))
    assert 10.0 <= v["g.inner"] <= 20.0


def test_sample_constant_only():
    c = ParameterSpace(k=Constant(7)).sample(UniformSampler(seed=0))
    assert c.tensor.numel() == 0 and c.values == {"k": 7}


def test_explicit_sampler_overrides_registered():
    space = ParameterSpace(x=Continuous(0.0, 1.0))
    space.register_sampler(UniformSampler(seed=0))
    c = space.sample(sampler=UniformSampler(seed=1))
    assert torch.equal(c.tensor, UniformSampler(seed=1).ask(1, 1)[0])


def test_sample_grad_flags():
    space = ParameterSpace(x=Continuous(0.0, 1.0))
    assert space.sample(UniformSampler(seed=0), requires_grad=True).tensor.requires_grad

    batch = space.sample_batch(UniformSampler(seed=0), batch_size=3)
    assert batch.space is space and batch.tensor.shape == (3, 1)
    assert all(not c.tensor.requires_grad for c in batch)

    grads = space.sample_batch(UniformSampler(seed=0), requires_grad=True)
    assert all(c.tensor.requires_grad for c in grads)


def test_values_roundtrip():
    c = Candidate.from_values(full_space(), sample_values(), requires_grad=True)
    assert c.tensor.requires_grad
    v = c.detach().values
    assert v["x"] == pytest.approx(1.5) and v["n"] == 4 and v["m"] == "b"
    assert torch.allclose(v["v"], torch.tensor([0.1, 0.9]))
    assert v["g.inner"] == pytest.approx(12.0) and v["k"] == 42


def test_item_slices_cover_tensor():
    space = full_space()
    c = Candidate.from_values(space, sample_values())
    widths = [c[name].numel() if isinstance(c[name], torch.Tensor) else 1
              for name, p in space if not isinstance(p, Constant)]
    assert sum(widths) == c.tensor.numel() == 8


def test_getitem():
    c = Candidate.from_values(full_space(), sample_values())
    assert c["x"] == pytest.approx(1.5)
    assert torch.allclose(c["m"], torch.tensor([0.0, 1.0, 0.0]))
    assert c["k"] == 42
    assert "g.inner" in c and "g" not in c


def test_parameter_blocks_are_named_tensor_views_with_gradients():
    space = full_space()
    candidate = Candidate.from_values(space, sample_values(), requires_grad=True)
    blocks = list(candidate.parameters())

    assert [name for name, _, _ in blocks] == ["x", "n", "m", "v_1", "v_2", "g.inner"]
    assert [tuple(tensor.shape) for _, _, tensor in blocks] == [(1,), (1,), (3,), (1,), (1,), (1,)]
    assert all(parameter is dict(space)[name] for name, parameter, _ in blocks)
    assert candidate._index is space._index
    torch.testing.assert_close(torch.cat([tensor for _, _, tensor in blocks]), candidate.tensor)
    for _, _, tensor in blocks:
        assert tensor.untyped_storage().data_ptr() == candidate.tensor.untyped_storage().data_ptr()

    sum((i + 1) * tensor.sum() for i, (_, _, tensor) in enumerate(blocks)).backward()
    torch.testing.assert_close(candidate.grad, torch.tensor([1., 2., 3., 3., 3., 4., 5., 6.]))


def test_parameter_blocks_can_be_transformed_and_concatenated_with_gradients():
    space = full_space()
    candidate = Candidate(space, space.encode(sample_values()).double().requires_grad_())
    blocks = []
    for _, parameter, tensor in candidate.parameters():
        blocks.append(tensor + 1 if isinstance(parameter, Continuous) else tensor)
    updated = Candidate(space, torch.cat(blocks))

    assert updated.tensor.dtype == candidate.tensor.dtype
    assert updated.tensor.device == candidate.tensor.device
    assert updated.values["x"].item() == pytest.approx(2.5)
    assert updated.values["n"] == 4 and updated.values["m"] == "b"
    torch.testing.assert_close(updated.values["v"], candidate.values["v"] + 1)
    assert candidate.values["x"].item() == pytest.approx(1.5)

    updated.tensor.sum().backward()
    torch.testing.assert_close(candidate.grad, torch.ones_like(candidate.tensor))


def test_constant_only_space_has_no_encoded_parameter_blocks():
    candidate = Candidate.from_values(ParameterSpace(k=Constant(42)), {})
    assert list(candidate.parameters()) == []
    assert candidate.values == {"k": 42}


def test_binding_uses_names_for_reused_definitions_and_custom_nested_keys():
    shared = Continuous(0.0, 10.0)
    space = ParameterSpace(
        key_fn="/".join,
        g=Group(x=shared, y=shared, m=Categorical(["only"])),
        v=Vector(0.0, 1.0, size=2),
    )
    candidate = Candidate.from_values(space, {"g/x": 1., "g/y": 2., "g/m": "only", "v": [0.1, 0.2]})
    assert [name for name, _, _ in candidate.parameters()] == ["g/x", "g/y", "g/m", "v_1", "v_2"]
    updated = Candidate(space, torch.cat([
        tensor + 2 if name == "g/y" else tensor
        for name, _, tensor in candidate.parameters()
    ]))
    assert updated.values["g/x"] == pytest.approx(1.0)
    assert updated.values["g/y"] == pytest.approx(4.0)
    assert updated.values["g/m"] == "only"


def test_parameter_iteration_and_concatenation_preserve_forward_ad():
    candidate = Candidate.from_values(full_space(), sample_values())
    tangent = torch.arange(8, dtype=candidate.tensor.dtype)
    with fw.dual_level():
        dual = candidate.set_tangent(tangent)
        assembled = torch.cat([tensor for _, _, tensor in dual.parameters()])
        primal, assembled_tangent = fw.unpack_dual(assembled)
        torch.testing.assert_close(primal, candidate.tensor)
        torch.testing.assert_close(assembled_tangent, tangent)


def test_eq_compares_decoded_values():
    space = full_space()
    a = Candidate.from_values(space, sample_values())
    assert a == Candidate.from_values(space, sample_values())
    assert a != Candidate.from_values(space, {**sample_values(), "m": "a"})
    assert (a == "not a candidate") is False

    n = ParameterSpace(n=Integer(0, 3))
    assert Candidate(n, torch.tensor([4.0])) == Candidate(n, torch.tensor([4.2]))


def test_eq_incompatible_spaces():
    a = Candidate.from_values(ParameterSpace(x=Continuous(0.0, 1.0)), {"x": 1.0})
    assert a != Candidate.from_values(ParameterSpace(y=Continuous(0.0, 1.0)), {"y": 1.0})
    assert a != Candidate.from_values(ParameterSpace(x=Constant(1.0)), {})
    v2 = Candidate.from_values(ParameterSpace(v=Vector(0.0, 1.0, size=2)), {"v": [0.0, 0.0]})
    v3 = Candidate.from_values(ParameterSpace(v=Vector(0.0, 1.0, size=3)), {"v": [0.0, 0.0, 0.0]})
    assert v2 != v3


def test_candidate_grad_and_detach():
    space = ParameterSpace(x=Continuous(0.0, 1.0))
    c = Candidate.from_values(space, {"x": 0.5}, requires_grad=True)
    assert c.grad is None and c.requires_grad_(True) is c
    (2.0 * c.values["x"]).backward()
    assert torch.allclose(c.grad, torch.tensor([2.0]))
    assert not c.detach().tensor.requires_grad


def test_set_tangent():
    c = Candidate.from_values(ParameterSpace(x=Continuous(0.0, 1.0)), {"x": 0.5})
    tangent = torch.tensor([3.0])
    with fw.dual_level():
        primal, dual_tangent = fw.unpack_dual(c.set_tangent(tangent).tensor)
    assert torch.equal(primal, c.tensor) and torch.equal(dual_tangent, tangent)
    with pytest.raises(AssertionError):
        c.set_tangent(torch.tensor([1.0, 2.0]))


def test_batch_grad():
    batch = full_space().sample_batch(UniformSampler(seed=0), batch_size=3, requires_grad=True)
    with pytest.raises(ValueError, match="Not all candidates have gradients"):
        batch.grad
    batch.tensor.sum().backward()
    assert batch.grad.shape == batch.tensor.shape
    assert all(not d.tensor.requires_grad for d in batch.detach())

    fresh = full_space().sample_batch(UniformSampler(seed=1), batch_size=2)
    assert fresh.requires_grad_(True) is fresh
    assert all(c.tensor.requires_grad for c in fresh)


def test_repr():
    body = repr(Candidate.from_values(full_space(), sample_values()))
    assert body.startswith("Candidate(") and body.endswith(")")
    parts = dict(re.findall(r"([\w.]+)=((?:\[[^\]]*\])|[^,]+)", body[len("Candidate("):-1]))
    assert list(parts) == ["x", "n", "m", "v", "g.inner", "k"]
    assert float(parts["x"]) == pytest.approx(1.5)
    assert parts["n"] == "4" and parts["m"] == "b" and parts["k"] == "42"
    assert float(parts["g.inner"]) == pytest.approx(12.0)
    assert [float(x) for x in parts["v"].strip("[]").split(", ")] == pytest.approx([0.1, 0.9])


def test_batch_set_tangent_rows():
    batch = full_space().sample_batch(UniformSampler(seed=0), batch_size=2)
    tangent = torch.arange(len(batch), dtype=torch.float32).unsqueeze(1).repeat(1, batch.tensor.shape[1])
    with fw.dual_level():
        duals = batch.set_tangent(tangent)
        for i, (_, d) in enumerate(fw.unpack_dual(c.tensor) for c in duals):
            assert torch.equal(d, tangent[i])


def test_batch_set_tangent_wrong_size():
    batch = full_space().sample_batch(UniformSampler(seed=0), batch_size=2)
    tangent = torch.zeros(1, batch.tensor.shape[1])
    with pytest.raises(AssertionError):
        batch.set_tangent(tangent)
