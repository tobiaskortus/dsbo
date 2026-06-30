import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from diffusers.schedulers import DDPMScheduler, DDIMScheduler

from typing import Callable

from dsbo.parameters import Candidate, Sampler
from dsbo.backends.aido.strategy import SurrogateStrategy
from dsbo.surrogates.diffusion import MlpDiffusion1D, Trainer

class LocalSurrogate(nn.Module):
    def __init__(self, strategy: SurrogateStrategy):
        super().__init__()
        self._strategy = strategy

    def trust_region(self, candidate: Candidate) -> bool:
        return self._strategy.trust_region(candidate)
    
    def fit(self, design: Candidate, workflow_callback: Callable[[dict], dict[str, float]]):
        pass


def dict_to_tensor(d: dict[str, float]) -> torch.Tensor:
    return torch.tensor(list(d.values()), dtype=torch.float32)


class Normalizer:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, T: torch.Tensor) -> None:
        self.mean = T.mean(dim=0, keepdim=True)
        self.std = T.std(dim=0, unbiased=False, keepdim=True)
        self.std = torch.clamp(self.std, min=self.eps)
        device = T.device
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        with torch.no_grad():
            mean = T.mean(dim=0, keepdim=True)
            std = T.std(dim=0, unbiased=False, keepdim=True)
            std = torch.clamp(std, min=self.eps)

        # store detached tensors (no grad history)
        self.mean = mean.detach()
        self.std = std.detach()
        device = T.device
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def normalize(self, T: torch.Tensor) -> torch.Tensor:
        return (T - self.mean) / self.std

    def unnormalize(self, T: torch.Tensor) -> torch.Tensor:
        return T * self.std + self.mean


class LocalDiffusionSurrogate(LocalSurrogate):
    def __init__(self, strategy: SurrogateStrategy, hidden_size: int = 256, num_layers: int = 3):
        super().__init__(strategy)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = None
        self.noise_scheduler = None
        self.X_normalizer = Normalizer()
        self.C_normalizer = Normalizer()

    def forward(self, x: torch.Tensor):
        # normalize parameters, sample, then unnormalize outputs
        params = self.C_normalizer.normalize(x)
        ddim_scheduler = DDIMScheduler.from_config(self.noise_scheduler.config, clip_sample=False)
        ddim_scheduler.set_timesteps(1000)
        sample = self.model.sample(params, ddim_scheduler)
        sample = self.X_normalizer.unnormalize(sample)
        return sample.squeeze()

    def fit(self, design: Candidate, workflow_callback: Callable[[list[dict]], list[dict[str, float]]]):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        designs = design.space.sample_batch(self._strategy.sampler, 16, requires_grad=False)
        designs.append(design) #Include current design into training data.

        #TODO: Define a strategy for handling failed simulations.
        # Ensure training inputs are detached from any external computation graph
        C = torch.stack([d.tensor.detach().clone() for d in designs]).to(device)
        X = [dict_to_tensor(x) for x in workflow_callback(designs, catch=(FileNotFoundError, AssertionError))]
        X = torch.nested.as_nested_tensor(X).to_padded_tensor(100).to(device)
        C = C.float(); X = X.unsqueeze(1).float()

        # fit and apply normalizers
        self.X_normalizer.fit(X)
        X = self.X_normalizer.normalize(X)
        self.C_normalizer.fit(C)
        C = self.C_normalizer.normalize(C)

        objective_dim = X.shape[-1]

        dataset    = TensorDataset(X, C)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True, num_workers=0)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="v_prediction",
            clip_sample=False,
        )

        self.model = MlpDiffusion1D(objective_dim=objective_dim, parameter_dim=C.shape[1], n_layers=self.num_layers)
        optimizer = AdamW(self.model.parameters(), lr=1e-3)
        trainer = Trainer(self.model, optimizer=optimizer, noise_scheduler=self.noise_scheduler, device=device)

        trainer.fit(n_epochs=500, dataloader=dataloader)