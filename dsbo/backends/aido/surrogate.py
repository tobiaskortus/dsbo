import torch
from torch import nn
from typing import Callable
from abc import ABC, abstractmethod

from dsbo.parameters import Candidate, Sampler


class LocalSampler(Sampler):
    def __init__(self, x_t: Candidate):
        super().__init__()
        self.x_t = x_t

    def ask(self) -> torch.Tensor:
        raise NotImplementedError("LocalSampler is a placeholder and should not be used directly.")


class LocalNormalSampler(LocalSampler):
    def __init__(self, x_t: Candidate, K: torch.Tensor):
        super().__init__(x_t)

        assert K.shape == (x_t.tensor.shape[0], x_t.tensor.shape[0]), \
              "Covariance matrix K must match the dimensionality of the candidate."

        self.K = K
        self.normal = torch.distributions.MultivariateNormal(x_t.tensor, covariance_matrix=K)
        
    def ask(self) -> torch.Tensor:
        return self.normal.sample()


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
        self._scale = scale
        self._cov = cov

    def _update_covariance(self, design: Candidate, min_scale: float = 2.0):
        diff = (design.tensor - self._param_state).detach()
        norm = torch.norm(diff) + 1e-8
        direction = diff / norm

        s = min_scale * max(1.0, 4.0 * norm.item())
        adjustment = (s - 1.0) * torch.outer(direction, direction)
        return torch.diag(self._cov)**2 + adjustment
        
    def update(self, design: Candidate):
        # Skip covariance update on first call
        if self._param_state is not None:
            self._cov = self._update_covariance(design)


        #Update the reference point for trust region and the local sampler
        self._param_state = design.tensor.detach().clone()
        self._sampler = LocalNormalSampler(x_t=design, K=self._cov)

    def trust_region(self, design: Candidate) -> bool:
        if self._param_state is None:
            return False

        diff = (design.tensor - self._param_state).detach()
        return torch.dot(diff, torch.mv(torch.linalg.inv(self._cov), diff)).item() < self._scale


class LocalSurrogate(nn.Module):
    def __init__(self, strategy: SurrogateStrategy):
        super().__init__()
        self._strategy = strategy

    def trust_region(self, candidate: Candidate) -> bool:
        return self._strategy.trust_region(candidate)
    
    def fit(self, design: Candidate, workflow_callback: Callable[[dict], dict[str, float]]):
        pass