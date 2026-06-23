from __future__ import annotations

import numpy as np
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from diffusers.schedulers import DDPMScheduler, DDIMScheduler

from dsbo.modules import Module
from dsbo.parameters import Candidate
from dsbo.config import get_active_file_backend
from dsbo.utils.io.workdir import AbstractFilePath
from dsbo.surrogates import SurrogateBase, SurrogateStrategy
from dsbo.surrogates.diffusion import LocalSurrogateDiffuser, Trainer


class LazyShape(tuple):
    _fpath: AbstractFilePath
    _resolved: tuple[int, ...] | None

    def __new__(cls, fpath: AbstractFilePath):
        instance = super().__new__(cls)
        instance._fpath = fpath
        instance._resolved = None
        return instance

    def _resolve(self):
        if self._resolved is not None:
            return
        Storage = get_active_file_backend()
        shards = Storage._discover_shards_from_abstract(self._fpath)
        if not shards:
            raise RuntimeError(
                f"Cannot resolve shape: no shard files found for "
                f"'{self._fpath.filename}'. The module must be run first."
            )
        arr = np.load(shards[0], mmap_mode="r")
        self._resolved = arr.shape[1:]

    def __getitem__(self, idx):  # type: ignore[override]
        self._resolve()
        return self._resolved[idx]

    def __len__(self):
        self._resolve()
        return len(self._resolved)

    def __iter__(self):
        self._resolve()
        return iter(self._resolved)

    def __repr__(self):
        if self._resolved is None:
            return "LazyShape(unresolved)"
        return repr(self._resolved)


class ModuleSurrogateContract:
    def __init__(
        self,
        output_dim: tuple[int, ...],
        output_samples: int,
        conditional_dim: tuple[int, ...] | None = None,
        output_conditional_fn: Callable | None = None,
        data_file: AbstractFilePath | None = None,
        conditional_file: AbstractFilePath | None = None,
    ):
        self.output_dim = output_dim
        self.output_samples = output_samples
        self.conditional_dim = conditional_dim
        self.output_conditional_fn = output_conditional_fn
        self.data_file = data_file
        self.conditional_file = conditional_file

    def to_data_loader(self, batch_size: int = 256) -> DataLoader:
        Storage = get_active_file_backend()
        return Storage.to_data_loader(
            self.data_file, self.conditional_file, batch_size=batch_size
        )

    def write_output(self, design: dict, output_tensor: torch.Tensor):
        Storage = get_active_file_backend()
        primal = output_tensor.detach().cpu().numpy()
        if self.data_file is not None:
            Storage.write_primal(self.data_file(), primal)
        if self.conditional_file is not None:
            cond = self.output_conditional_fn(design) \
                if self.output_conditional_fn is not None \
                else torch.tensor(list(design.values()), dtype=torch.float32)
            Storage.write_primal(self.conditional_file(), cond.detach().cpu().numpy())

    def __repr__(self) -> str:
        return (
            f"ModuleSurrogateContract(output_dim={self.output_dim}, "
            f"samples={self.output_samples})"
        )


class DiffusionModelWrapper(nn.Module):
    def __init__(
        self,
        diffusion_model: LocalSurrogateDiffuser,
        X_mean: torch.Tensor,
        X_std: torch.Tensor,
        C_mean: torch.Tensor,
        C_std: torch.Tensor,
    ):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.register_buffer("X_mean", X_mean)
        self.register_buffer("X_std", X_std)
        self.register_buffer("C_mean", C_mean)
        self.register_buffer("C_std", C_std)


def requires_contract(fn):
    def wrapper(self, design: dict, **kwargs):
        if self.contract is None:
            raise RuntimeError("No contract registered.")
        return fn(self, design, **kwargs)
    return wrapper


def requires_diffusion_model(fn):
    def wrapper(self, design: dict, **kwargs):
        if self._diffusion_model is None:
            raise RuntimeError("Surrogate not trained.")
        return fn(self, design, **kwargs)
    return wrapper

class ModuleSurrogate(SurrogateBase):
    def __init__(self, surrogate: Module | None, strategy: SurrogateStrategy):
        super().__init__(surrogate=surrogate, strategy=strategy)
        self.contract: ModuleSurrogateContract | None = None

    
    def register_contract(self, contract: ModuleSurrogateContract):
        if contract.output_samples <= 0:
            raise ValueError("output_samples > 0 is required.")
        self.contract = contract

    @requires_contract
    @requires_diffusion_model
    def __call__(self, design: dict, **kwargs) -> dict[str, torch.Tensor]:
        pass

    @requires_contract
    def train(self, candidate: Candidate, workflow_callback: Callable):
        pass


class DiffusersModuleSurrogate(ModuleSurrogate):
    def __init__(self, strategy: SurrogateStrategy):
        super().__init__(surrogate=None, strategy=strategy)
        self._diffusion_model: DiffusionModelWrapper | None = None
        self._noise_scheduler: DDPMScheduler | None = None
        self._sample_scheduler: DDIMScheduler | None = None
        self.training_metrics: dict[str, Any] | None = None


    def __call__(self, design: dict, **kwargs) -> dict[str, torch.Tensor]:
        cond = self.contract.output_conditional_fn(design) \
            if self.contract.output_conditional_fn is not None \
            else torch.tensor(list(design.values()), dtype=torch.float32)
        if cond.ndim == 1:
            cond = cond.unsqueeze(0)

        cond = (cond - self._diffusion_model.C_mean) / self._diffusion_model.C_std

        self._diffusion_model.eval()
        self._sample_scheduler.set_timesteps(self._sample_scheduler.config.num_train_timesteps)
        with torch.no_grad():
            B = cond.shape[0]
            #Add an additional leading dimension for vector-shaped outputs to stay compatible with the diffusers API
            output_dim = (1, *self.contract.output_dim) if len(self.contract.output_dim) == 1 else self.contract.output_dim
            x = torch.randn(B, *output_dim, device=cond.device)
            for t in self._sample_scheduler.timesteps:
                v = self._diffusion_model.diffusion_model(
                    x, t.expand(B).to(cond.device), cond
                ).sample
                x = self._sample_scheduler.step(v, t, x).prev_sample

        x = x.squeeze(1)
        if len(self.contract.output_dim) > 1:
            x = x.view(B, *self.contract.output_dim)

        x = x * self._diffusion_model.X_std + self._diffusion_model.X_mean
        self.contract.write_output(design, x)
        return {}


    def train(self, candidate: Candidate, workflow_callback: Callable):

        candidates = self.strategy.generate_candidates(candidate, candidate.space)
        workflow_callback(candidates)

        loader = self.contract.to_data_loader(batch_size=256)

        output_dim = self.contract.output_dim
        cond_dim = self.contract.conditional_dim

        loader.dataset.compute_stats()

        model = LocalSurrogateDiffuser(
            objective_dim=output_dim,
            parameter_dim=cond_dim,
            emb_dim=128,
            hidden_dim=128,
            n_layers=4,
        )

        self._diffusion_model = DiffusionModelWrapper(
            model,
            loader.dataset.mean_x, loader.dataset.std_x,
            loader.dataset.mean_c, loader.dataset.std_c,
        )
        self._noise_scheduler = DDPMScheduler(num_train_timesteps=250, prediction_type="v_prediction")
        self._sample_scheduler = DDIMScheduler(num_train_timesteps=250, prediction_type="v_prediction")

        optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100, eta_min=1e-6)

        loss_history = Trainer(
            model, self._noise_scheduler, optim, lr_scheduler,
            device=next(model.parameters()).device,
            model_tag=type(self).__name__.lower(),
        ).fit(n_epochs=100, dataloader=loader, show_progress=False)

        self.training_metrics = {
            "n_samples": len(loader.dataset),
            "n_candidates": len(candidates),
            "output_dim": output_dim,
            "cond_dim": cond_dim,
            "loss_history": loss_history,
            "final_loss": loss_history[-1] if loss_history else None,
        }

        return self._diffusion_model