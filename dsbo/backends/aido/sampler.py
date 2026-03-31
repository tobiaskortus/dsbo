import torch
from dsbo.parameters import Candidate, Sampler

def _ensure_pd(cov: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    cov = (cov + cov.T) / 2
    if cov.dim() == 1:
        cov = torch.diag(cov)
    cov = cov + eps * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
    return cov


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

        K_pd = _ensure_pd(K)
        self.K = K_pd
        self.normal = torch.distributions.MultivariateNormal(x_t.tensor, covariance_matrix=K_pd)
        
    def ask(self) -> torch.Tensor:
        return self.normal.sample()