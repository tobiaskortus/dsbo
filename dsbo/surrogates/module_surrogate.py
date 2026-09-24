from __future__ import annotations
import logging
from dataclasses import dataclass

from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers.schedulers import DDPMScheduler, DDIMScheduler

from dsbo.utils.io import Input, LazyShape
from dsbo.autograd import train_op
from dsbo.config import get_active_file_backend
from dsbo.modules import dynamic_submodule
from dsbo.utils.io.storage.autograd import AutogradFileInterface
from dsbo.utils.io.datasets import ModuleSurrogateDataset
from dsbo.surrogates import SurrogateBase, SurrogateStrategy
from dsbo.surrogates.diffusion import MlpDiffusion1D, Trainer
from dsbo.utils.io.workdir import AbstractFilePath, PathAnnotation


class ModuleSurrogateContract:
    def __init__(
        self,
        output_dim: tuple[int, ...],
        output_samples: int,
        conditional_dim: tuple[int, ...] | None = None,
        output_conditional_fn: Callable | None = None,
        data_file: AbstractFilePath | PathAnnotation | None = None,
        conditional_file: AbstractFilePath | PathAnnotation | None = None,
    ):
        self.output_dim = output_dim
        self.output_samples = output_samples
        self.conditional_dim = conditional_dim
        self.output_conditional_fn = output_conditional_fn
        self.data_file = data_file
        self.conditional_file = conditional_file

    @staticmethod
    def _abstract_path(file: AbstractFilePath | PathAnnotation) -> AbstractFilePath:
        return file.path if isinstance(file, PathAnnotation) else file

    def to_data_loader(self, batch_size: int = 256) -> DataLoader:
        Storage: type[AutogradFileInterface] = get_active_file_backend()
        if self.data_file is None or self.conditional_file is None:
            raise RuntimeError(
                "A module surrogate contract requires data and conditional files "
                "to create a data loader."
            )
        data_file = self._abstract_path(self.data_file)
        conditional_file = self._abstract_path(self.conditional_file)

        if isinstance(self.output_dim, LazyShape):
            self.output_dim.resolve(data_file())
        if isinstance(self.conditional_dim, LazyShape):
            self.conditional_dim.resolve(conditional_file())

        dataset = ModuleSurrogateDataset(
            data_file(),
            conditional_file(),
            backend=Storage,
        )
        return DataLoader(dataset, batch_size=batch_size)

    def write_output(self, design: dict, output_tensor: torch.Tensor):
        Storage: type[AutogradFileInterface] = get_active_file_backend()
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


class StatefulModuleSurrogateWrapper(nn.Module):
    def __init__(
        self,
        diffusion_model: MlpDiffusion1D,
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


class ModuleSurrogate(SurrogateBase):
    def __init__(self, strategy: SurrogateStrategy):
        super().__init__(strategy=strategy)
        self.contract: ModuleSurrogateContract | None = None

    def register_contract(self, contract: ModuleSurrogateContract):
        if contract.output_samples <= 0:
            raise ValueError("output_samples > 0 is required.")
        self.contract = contract

        #Automatically register the required input files for the module‚
        for attr in ("data_file", "conditional_file"):
            path = getattr(contract, attr)
            if path is not None:
                setattr(self, attr, Input(ModuleSurrogateContract._abstract_path(path)))

class DiffusersModuleSurrogate(ModuleSurrogate):
    def __init__(self, strategy: SurrogateStrategy):
        super().__init__(strategy=strategy)
        self._diffusion_model: StatefulModuleSurrogateWrapper | None = None
        self._noise_scheduler: DDPMScheduler | None = None
        self._sample_scheduler: DDIMScheduler | None = None
        self.training_metrics: dict[str, Any] | None = None

        self._num_train_timesteps = 1000
        self._num_infer_timesteps = 100

    def _create_schedulers(self) -> None:
        self._noise_scheduler = DDPMScheduler(
            clip_sample=False,
            num_train_timesteps=self._num_train_timesteps,
            prediction_type="v_prediction",
            beta_schedule="squaredcos_cap_v2",
        )
        self._sample_scheduler = DDIMScheduler(
            clip_sample=False,
            num_train_timesteps=self._num_train_timesteps,
            prediction_type="v_prediction",
            beta_schedule="squaredcos_cap_v2",
        )


    def __call__(self, design: dict, **kwargs) -> dict[str, torch.Tensor]:
        self.get_runtime_model()
        if self._diffusion_model is None:
            raise RuntimeError("Surrogate not trained.")
        cond = self.contract.output_conditional_fn(design) \
            if self.contract.output_conditional_fn is not None \
            else torch.tensor(list(design.values()), dtype=torch.float32)
        if cond.ndim == 1:
            cond = cond.unsqueeze(0)

        cond = (cond - self._diffusion_model.C_mean) / self._diffusion_model.C_std

        self._diffusion_model.eval()
        self._sample_scheduler.set_timesteps(self._num_infer_timesteps)
        with torch.no_grad():
            B = cond.shape[0]
            #Add an additional leading dimension for vector-shaped outputs to stay compatible with the diffusers API
            output_dim = (1, *self.contract.output_dim) if len(self.contract.output_dim) == 1 else self.contract.output_dim
            x = torch.randn(B, *output_dim, device=cond.device)
            for t in self._sample_scheduler.timesteps:
                v = self._diffusion_model.diffusion_model(
                    x, t.expand(B).to(cond.device), cond
                ).sample

                x = self._sample_scheduler.step(v, t, x, eta=0.8).prev_sample

        x = x.squeeze(1)
        if len(self.contract.output_dim) > 1:
            x = x.view(B, *self.contract.output_dim)

        x = x * self._diffusion_model.X_std + self._diffusion_model.X_mean
        self.contract.write_output(design, x)
        return {}


    def _train_from_loader(self, loader: DataLoader, num_candidates: int):
        output_dim = self.contract.output_dim
        cond_dim = self.contract.conditional_dim

        loader.dataset.compute_stats()

        model = MlpDiffusion1D(
            objective_dim=output_dim,
            parameter_dim=cond_dim,
            emb_dim=256,
            hidden_dim=256,
            n_layers=12,
            dropout=0.1,
        )

        diffusion_model = StatefulModuleSurrogateWrapper(
            model,
            loader.dataset.mean_x, loader.dataset.std_x,
            loader.dataset.mean_c, loader.dataset.std_c,
        )

        self._create_schedulers()

        optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=200, eta_min=1e-6)

        loss_history = Trainer(
            model, self._noise_scheduler, optim, lr_scheduler,
            device=next(model.parameters()).device,
            model_tag=type(self).__name__.lower(),
        ).fit(n_epochs=200, dataloader=loader, show_progress=False)

        self.training_metrics = {
            "n_samples": len(loader.dataset),
            "n_candidates": num_candidates,
            "output_dim": tuple(output_dim),
            "cond_dim": tuple(cond_dim),
            "loss_history": loss_history,
            "final_loss": loss_history[-1] if loss_history else None,
        }

        return diffusion_model


    @train_op
    @dynamic_submodule
    def train(self, design, candidates):
        model_file_path = self.model_file_path() if callable(self.model_file_path) else self.model_file_path
        if model_file_path is None:
            raise RuntimeError(
                f"{type(self).__name__} training requires a model_file_path; "
                "the module surrogate training plan must set model_file_path."
            )
        loader = self.contract.to_data_loader(batch_size=256)
        
        try:
            diffusion_model = self._train_from_loader(loader, len(candidates))
            torch.save(self._model_state(diffusion_model), model_file_path)
        finally:
            loader.dataset.close()

        return {}

    def _model_state(self, diffusion_model) -> dict:
        return {
            "output_dim": tuple(self.contract.output_dim),
            "conditional_dim": tuple(self.contract.conditional_dim),
            "training_metrics": self.training_metrics,
            "model": diffusion_model.state_dict(),
        }

    def state_dict(self) -> dict:
        return self._model_state(self._diffusion_model)

    def from_state_dict(self, state: dict) -> None:
        if isinstance(self.contract.output_dim, LazyShape):
            self.contract.output_dim.set(state["output_dim"])
        if isinstance(self.contract.conditional_dim, LazyShape):
            self.contract.conditional_dim.set(state["conditional_dim"])
        self.training_metrics = state["training_metrics"]

        model = MlpDiffusion1D(
            objective_dim=self.contract.output_dim,
            parameter_dim=self.contract.conditional_dim,
            emb_dim=256,
            hidden_dim=256,
            n_layers=12,
            dropout=0.1,
        )
        wrapper = StatefulModuleSurrogateWrapper(
            model,
            X_mean=torch.zeros(self.contract.output_dim[-1:]),
            X_std=torch.ones(self.contract.output_dim[-1:]),
            C_mean=torch.zeros(self.contract.conditional_dim[-1:]),
            C_std=torch.ones(self.contract.conditional_dim[-1:]),
        )
        wrapper.load_state_dict(state["model"])
        self._diffusion_model = wrapper
        self.surrogate = wrapper

        self._create_schedulers()
