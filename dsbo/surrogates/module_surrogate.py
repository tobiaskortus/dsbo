from __future__ import annotations

import numpy as np
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from diffusers.schedulers import DDPMScheduler, DDIMScheduler

from dsbo.modules import Module
from dsbo.utils.io import Workdir
from dsbo.parameters import Candidate
from dsbo.utils.io.autograd import NumpyFile
from dsbo.utils.io.workdir import AbstractFilePath
from dsbo.surrogates import SurrogateBase, SurrogateStrategy
from dsbo.surrogates.diffusion import LocalSurrogateDiffuser, LocalSurrogateDiffuser1DCNN, Trainer


import os
from pathlib import Path


def normalize(X: torch.Tensor) -> torch.Tensor:
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True) + 1e-8
    return (X - mean) / std, mean, std


def load_data(file_path: AbstractFilePath | None) -> torch.Tensor | None:
    #TODO: This is just for testing purposes
    if file_path is None:
        return None

    from dsbo.utils.io.workdir import PathAnnotation
    if isinstance(file_path, PathAnnotation):
        file_path = file_path.path

    wd = Workdir._current_workdir.get()
    if wd is None:
        return None

    search_root = Path(wd.root)
    name, ext = os.path.splitext(file_path.filename)
    rel = file_path.pathname

    patterns = [f"**/{rel}/{file_path.filename}"]
    if file_path._unique:
        patterns.append(f"**/{rel}/{name}_*{ext}")
    patterns.append(f"**/batch_*/{rel}/{file_path.filename}")
    if file_path._unique:
        patterns.append(f"**/batch_*/{rel}/{name}_*{ext}")

    matches = sorted(set(
        str(p) for pat in patterns
        for p in search_root.glob(pat)
        if p.is_file() and "_gradient" not in p.name and "_probe" not in p.parts
    ))

    chunks = []
    for m in matches:
        try:
            chunks.append(torch.from_numpy(np.load(m)["primal"]).float())
        except Exception:
            continue
    if not chunks:
        return None
    out = torch.cat(chunks, dim=0)
    return out if out.ndim > 1 else out.unsqueeze(-1)



class ModuleSurrogateContract:
    def __init__(
        self,
        output_dim: tuple[int, ...],
        output_samples: int,
        output_conditional_fn: Callable | None = None,
        data_file: AbstractFilePath | None = None,
        conditional_file: AbstractFilePath | None = None,
    ):
        self.output_dim = output_dim
        self.output_samples = output_samples
        self.output_conditional_fn = output_conditional_fn
        self.data_file = data_file
        self.conditional_file = conditional_file

    def load_training_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        X = load_data(self.data_file)
        C = load_data(self.conditional_file)
        if X is None or C is None:
            raise RuntimeError(
                "Could not load training data — make sure the module has "
                "been run for this candidate before calling train()."
            )
        if X.ndim == 1:
            X = X.unsqueeze(-1)
        if C.ndim == 1:
            C = C.unsqueeze(-1)
        return X, C

    def write_output(self, design: dict, output_tensor: torch.Tensor):
        primal = output_tensor.detach().cpu().numpy()
        if self.data_file is not None:
            NumpyFile.write_primal(self.data_file(), primal)
        if self.conditional_file is not None:
            cond = self.output_conditional_fn(design) \
                if self.output_conditional_fn is not None \
                else torch.tensor(list(design.values()), dtype=torch.float32)
            NumpyFile.write_primal(self.conditional_file(), cond.detach().cpu().numpy())

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
        if contract.output_dim is None or contract.output_samples <= 0:
            raise ValueError("output_dim and output_samples > 0 are required.")
        self.contract = contract

    @requires_contract
    @requires_diffusion_model
    def __call__(self, design: dict, **kwargs) -> dict[str, torch.Tensor]:
        pass

    @requires_contract
    def train(self, candidate: Candidate, workflow_callback: Callable):
        pass


class ModuleSurrogate(ModuleSurrogate):
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
            x = torch.randn(B, 1, self.contract.output_dim[0], device=cond.device)
            for t in self._sample_scheduler.timesteps:
                v = self._diffusion_model.diffusion_model(
                    x, t.expand(B).to(cond.device), parameters=cond
                ).sample
                x = self._sample_scheduler.step(v, t, x).prev_sample

        x = x.squeeze(1)
        if len(self.contract.output_dim) > 1:
            x = x.view(B, *self.contract.output_dim)
        x = x * self._diffusion_model.X_std + self._diffusion_model.X_mean
        print(f"[SURROGATE __call__] output shape={x.shape}, mean={x.mean():.3f}")

        self.contract.write_output(design, x)
        return {}


    def train(self, candidate: Candidate, workflow_callback: Callable):

        candidates = self.strategy.generate_candidates(candidate, candidate.space)
        workflow_callback(candidates)

        # TODO: improve this!!!!
        X, C = self.contract.load_training_data()

        X_norm, X_mean, X_std = normalize(X)
        C_norm, C_mean, C_std = normalize(C)

        model = LocalSurrogateDiffuser(
            objective_dim=X.shape[-1],
            parameter_dim=C.shape[-1],
            emb_dim=128,
            hidden_dim=128,
            n_layers=4,
        )

        self._diffusion_model = DiffusionModelWrapper(model, X_mean, X_std, C_mean, C_std)
        self._noise_scheduler = DDPMScheduler(num_train_timesteps=250, prediction_type="v_prediction")
        self._sample_scheduler = DDIMScheduler(num_train_timesteps=250, prediction_type="v_prediction")

        optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100, eta_min=1e-6)
        loader = DataLoader(TensorDataset(X_norm, C_norm), batch_size=min(len(X), 256), shuffle=True)

        loss_history = Trainer(model, self._noise_scheduler, optim, lr_scheduler, device=X.device).fit(
            n_epochs=100, dataloader=loader, show_progress=False
        )

        self.training_metrics = {
            "n_samples": len(X),
            "n_candidates": len(candidates),
            "output_dim": X.shape[-1],
            "cond_dim": C.shape[-1],
            "loss_history": loss_history,
            "final_loss": loss_history[-1] if loss_history else None,
        }

        return self._diffusion_model