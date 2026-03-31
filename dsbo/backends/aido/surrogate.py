import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from diffusers.schedulers import DDPMScheduler, DDIMScheduler

from typing import Callable
from abc import ABC, abstractmethod

from dsbo.parameters import Candidate, Sampler
from dsbo.backends.aido.strategy import SurrogateStrategy
from dsbo.surrogates.diffusion import LocalSurrogateDiffuser, Trainer

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

class LocalDiffusionSurrogate(LocalSurrogate):
    def __init__(self, strategy: SurrogateStrategy, hidden_size: int = 256, num_layers: int = 3):
        super().__init__(strategy)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = None
        self.noise_scheduler = None

    def forward(self, x: torch.Tensor):
        ddim_scheduler = DDIMScheduler.from_config(self.noise_scheduler.config, clip_sample=False)
        ddim_scheduler.set_timesteps(1000)
        return self.model.sample(x, ddim_scheduler).squeeze()

    def fit(self, design: Candidate, workflow_callback: Callable[[list[dict]], list[dict[str, float]]]):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        designs = design.space.sample_batch(self._strategy.sampler, 64, requires_grad=False)

        C = torch.stack([d.tensor for d in designs]).to(device)
        X = torch.stack([dict_to_tensor(x) for x in workflow_callback(designs)]).to(device)
        C = C.float(); X = X.unsqueeze(1).float()

        dataset    = TensorDataset(X, C)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True, num_workers=0)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="v_prediction",
            clip_sample=False,
        )

        self.model = LocalSurrogateDiffuser(objective_dim=1, parameter_dim=C.shape[1], n_layers=self.num_layers)
        optimizer = AdamW(self.model.parameters(), lr=1e-4)
        trainer = Trainer(self.model, optimizer=optimizer, noise_scheduler=self.noise_scheduler, device=device)

        trainer.fit(n_epochs=500, dataloader=dataloader)