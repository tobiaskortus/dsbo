import torch
from abc import ABC, abstractmethod

from dsbo.parameters import Candidate
from dsbo.backends.aido.sampler import LocalSampler, LocalNormalSampler, _ensure_pd

class SurrogateStrategy(ABC):
    def __init__(self):
        self._sampler = None
        self._param_state = None

    @abstractmethod
    def update(self, design: Candidate):
        pass

    @abstractmethod
    def trust_region(self, design: Candidate) -> bool:
        pass

    @property
    def sampler(self) -> LocalSampler:
        return self._sampler
    

class CovarianceStrategy(SurrogateStrategy):
    def __init__(self, cov: torch.Tensor, scale=1.0):
        super().__init__()
        self._cov = cov
        self._scale = scale

    def _update_covariance(self, design: Candidate, min_scale: float = 2.0):
        diff = (design.tensor - self._param_state).detach()
        norm = torch.norm(diff) + 1e-8
        direction = diff / norm

        s = min_scale * max(1.0, 4.0 * norm.item())
        adjustment = (s - 1.0) * torch.outer(direction, direction)
        cov = self._cov + adjustment
        return cov
        
    def update(self, design: Candidate):
        # Skip covariance update on first call
        if self._param_state is not None:
            self._cov = self._update_covariance(design)

        self._param_state = design.tensor.detach().clone()
        self._sampler = LocalNormalSampler(x_t=design, K=self._cov)

    def trust_region(self, design: Candidate) -> bool:
        if self._param_state is None:
            return False

        diff = (design.tensor - self._param_state).detach()
        return torch.dot(diff, torch.mv(torch.linalg.inv(self._cov), diff)).item() < self._scale