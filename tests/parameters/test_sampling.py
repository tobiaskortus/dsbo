import pytest
import torch

from dsbo.parameters.sampling import UniformSampler, SobolSampler, LatinHypercubeSampler

SEEDED = [UniformSampler, SobolSampler, LatinHypercubeSampler]


@pytest.mark.parametrize("sampler_cls", SEEDED)
def test_seeded_samplers_reproduce_their_sequence(sampler_cls):
    assert torch.equal(sampler_cls(seed=42).ask(8, 2), sampler_cls(seed=42).ask(8, 2))


@pytest.mark.parametrize("seed", [0, None])
@pytest.mark.parametrize("sampler_cls", SEEDED)
def test_successive_asks_do_not_repeat_the_previous_batch(sampler_cls, seed):
    sampler = sampler_cls(seed=seed)
    assert not torch.equal(sampler.ask(8, 2), sampler.ask(8, 2))


@pytest.mark.parametrize("sampler_cls", SEEDED)
@pytest.mark.parametrize(("n", "dims"), [(8, 1), (16, 16)])
def test_ask_returns_the_requested_shape(sampler_cls, n, dims):
    assert sampler_cls(seed=0).ask(n, dims).shape == (n, dims)


def test_latin_hypercube_places_exactly_one_point_per_stratum_and_dimension():
    n, dims = 12, 3
    strata = LatinHypercubeSampler(seed=0).ask(n, dims).mul(n).floor().long()
    for d in range(dims):
        assert sorted(strata[:, d].tolist()) == list(range(n))
