from __future__ import annotations

import torch
import torch.nn.functional as F

from dsbo.parameters import Candidate, CandidateBatch, Categorical, Integer, ParameterSpace
from dsbo.surrogates import SurrogateStrategy


class CovarianceStrategy(SurrogateStrategy):
    def __init__(
        self,
        *,
        sigma: float = 1.5,
        generate_scaling: float = 1.0,
        trust_region_scale: float = 0.8,
        num_candidates: int = 20,
        include_center: bool = True,
        min_scale: float = 2.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        
        self._seed = seed
        self.sigma = sigma
        self.min_scale = min_scale
        self.num_candidates = num_candidates
        self.include_center = include_center
        self.generate_scaling = generate_scaling
        self.trust_region_scale = trust_region_scale
        self._generator: torch.Generator | None = None

        self._center: torch.Tensor | None = None
        self._base_covariance: torch.Tensor | None = None
        self._covariance: torch.Tensor | None = None

    def generate_candidates(
        self,
        candidate: Candidate,
        space: ParameterSpace,
        finetune: bool = False,
    ) -> CandidateBatch:
        if self._seed is not None and self._generator is None:
            self._generator = torch.Generator(device=candidate.tensor.device).manual_seed(self._seed)

        center = candidate.detach()
        self._center = center.tensor
        if self._covariance is None:
            self._base_covariance = self._initial_covariance(self._center)
            self._covariance = self._base_covariance.clone()

        sampled = self.num_candidates - 1 if self.include_center else self.num_candidates
        candidates = [center] if self.include_center else []
        if sampled > 0:
            candidates.extend(self._sample_candidates(center, sampled))
        return CandidateBatch(candidates)

    def update(self, candidate: Candidate) -> None:
        if self._center is None:
            return

        diff = candidate.tensor.detach() - self._center
        norm = torch.linalg.norm(diff)
        direction = diff / (norm + 1e-4)
        scale = self.min_scale * max(1.0, 4.0 * float(norm))
        self._covariance = self._base_covariance.square()
        self._covariance += (scale - 1.0) * torch.outer(direction, direction)

    def is_in_trust_region(self, candidate: Candidate) -> bool:
        if self._center is None or self._covariance is None:
            return False

        diff = candidate.tensor.detach() - self._center
        quadratic = torch.dot(diff, torch.linalg.solve(self._covariance, diff))
        return bool(quadratic < self.trust_region_scale)

    def _initial_covariance(self, tensor: torch.Tensor) -> torch.Tensor:
        sigma = torch.full_like(tensor, self.sigma)
        return torch.diag(sigma.square())

    def _sample_candidates(self, candidate: Candidate, count: int) -> CandidateBatch:
        current = candidate.tensor

        covariance = self._covariance * self.generate_scaling
        noise = torch.randn(
            count, current.numel(), generator=self._generator,
            dtype=current.dtype, device=current.device,
        )
        draws = current + noise @ torch.linalg.cholesky(covariance).T

        parameters = list(candidate.parameters())
        proposals = draws.split([tensor.numel() for _, _, tensor in parameters], dim=-1)
        samples = {}
        for (name, parameter, tensor), proposal in zip(parameters, proposals):
            if isinstance(parameter, Categorical):
                choices = torch.multinomial(
                    tensor, count, replacement=True,
                    generator=self._generator,
                )
                samples[name] = F.one_hot(choices, num_classes=tensor.numel()).to(tensor)
            else:
                values = proposal.clamp(parameter.lower, parameter.upper)
                samples[name] = values.round() if isinstance(parameter, Integer) else values

        tensors = torch.cat(list(samples.values()), dim=-1)
        return CandidateBatch([Candidate(candidate.space, tensor) for tensor in tensors.unbind(0)])
