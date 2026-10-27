import math
import inspect
import numpy as np
from tqdm.auto import tqdm
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from diffusers.utils import BaseOutput
from diffusers.schedulers import SchedulerMixin
from torch.optim.lr_scheduler import LRScheduler
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        f = torch.exp(-math.log(10_000) * torch.arange(dim // 2) / (dim // 2 - 1))
        self.register_buffer("f", f)
        self.proj = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = t[:, None].float() * self.f[None]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.proj(emb)


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.net = nn.Linear(cond_dim, channels * 2)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gamma_beta = self.net(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma, beta


class MLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.film = FiLM(cond_dim, hidden_dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        gamma, beta = self.film(cond)
        h = h * (1 + gamma) + beta
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return x + h


@dataclass
class MlpDiffusion1DOutput(BaseOutput):
    sample: torch.Tensor


class MlpDiffusion1D(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        objective_dim: int | tuple[int, ...] = 64,
        emb_dim: int = 128,
        hidden_dim: int = 256,
        parameter_dim: int | tuple[int, ...] = 16,
        n_layers: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not isinstance(objective_dim, int):
            if len(objective_dim) != 1:
                raise ValueError(
                    f"objective_dim must be a 1-D vector, got shape {tuple(objective_dim)}. "
                    "Multi-dimensional outputs are not supported."
                )
            objective_dim = objective_dim[0]
        if not isinstance(parameter_dim, int):
            if len(parameter_dim) != 1:
                raise ValueError(
                    f"parameter_dim must be a 1-D vector, got shape {tuple(parameter_dim)}. "
                    "Multi-dimensional conditioning is not supported."
                )
            parameter_dim = parameter_dim[0]
        self.cond_dim = parameter_dim

        self.time_emb = PositionalEncoding(emb_dim)
        film_cond_dim = parameter_dim

        self.cond_proj = nn.Sequential(
            nn.Linear(parameter_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.in_fc = nn.Linear(objective_dim, emb_dim)
        self.blocks = nn.ModuleList([MLPBlock(emb_dim, hidden_dim, film_cond_dim, dropout) \
                                        for _ in range(n_layers)])
        self.out_fc = nn.Linear(emb_dim, objective_dim)

    def forward(
        self,
        x: torch.Tensor,        # (B, 1, seq_len)
        t: torch.Tensor,        # (B,)
        c: torch.Tensor | None = None,
        return_dict: bool = True,
    )  -> MlpDiffusion1DOutput | torch.Tensor:
        
        if x.dim() == 3:
            x = x.squeeze(1)
        
        x = self.in_fc(x)
        x = x + self.time_emb(t)
        
        if c is not None:
            x = x + self.cond_proj(c)

        for blk in self.blocks:
            x = blk(x, c)

        x = self.out_fc(x)
        x = x.unsqueeze(1)

        if return_dict:
            return MlpDiffusion1DOutput(sample=x)
        return x

    def sample(self, c: torch.Tensor, scheduler: SchedulerMixin) -> torch.Tensor:
        B = c.shape[0]
        x = torch.randn(B, 1, self.config.objective_dim, device=c.device)

        for t_scalar in scheduler.timesteps:
            t_batch = t_scalar.expand(B).to(c.device)
            v_pred = self(x, t_batch, c).sample
            step_out = scheduler.step(v_pred, t_scalar, x)
            x = step_out.prev_sample

        return x
    

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        noise_scheduler: SchedulerMixin,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler = None,
        ema_model=None,
        device=None,
        grad_clip: float = 1.0,
        model_tag: str | None = None,
    ):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.ema_model = ema_model
        self.device = device or next(model.parameters()).device
        self.grad_clip = grad_clip
        self._model_tag = model_tag

        if lr_scheduler is not None:
            step_sig = inspect.signature(lr_scheduler.step)
            params = list(step_sig.parameters.values())
            required_positional = [
                p for p in params
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                and p.default is inspect._empty
            ]
            self._lr_scheduler_needs_metric = len(required_positional) > 0
        else:
            self._lr_scheduler_needs_metric = False

    def _batch_loss(self, x0, cond) -> torch.Tensor:
        x0, cond = x0.to(self.device), cond.to(self.device)
        B = x0.shape[0]
        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device).long()
        noise = torch.randn_like(x0)
        x_t = self.noise_scheduler.add_noise(x0, noise, t)
        target = self.noise_scheduler.get_velocity(x0, noise, t)

        pred = self.model(x_t, t, cond).sample
        # Model outputs (B, 1, n) — squeeze to match target (B, n)
        return F.mse_loss(pred.squeeze(1), target)

    def step(self, x0, cond) -> float:
        loss = self._batch_loss(x0, cond)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.ema_model is not None:
            self.ema_model.step(self.model.parameters())

        return loss.item()

    def fit(
        self,
        n_epochs: int,
        dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        patience: int | None = None,
        show_progress: bool = True,
    ) -> list[float]:
        self.model.train()
        epoch_losses = []
        val_losses = []
        best_val = float("inf")
        best_model_state = None
        best_ema_state = None
        epochs_no_improve = 0
        outer = range(n_epochs)
        if show_progress:
            outer = tqdm(outer, desc="Epochs")

        from dsbo.storage import get_current_storage
        storage = get_current_storage()

        for epoch in outer:
            self.model.train()
            epoch_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
            for x0, cond in pbar:
                batch_loss = self.step(x0, cond)
                epoch_loss += batch_loss
                pbar.set_postfix({"loss": f"{batch_loss:.5f}"})

            avg = epoch_loss / len(dataloader)
            epoch_losses.append(avg)
            pbar.close()

            if self.lr_scheduler is not None:
                if self._lr_scheduler_needs_metric:
                    self.lr_scheduler.step(avg)
                else:
                    self.lr_scheduler.step()

            if show_progress:
                outer.set_postfix({"loss": f"{avg:.5f}"})

            if storage is not None and self._model_tag is not None:
                storage.intermediates.add_scalar(f"{self._model_tag}/mse_loss", epoch, avg)

            if val_dataloader is not None:
                self.model.eval()
                val_sum = 0.0
                with torch.no_grad():
                    for x0, cond in val_dataloader:
                        val_sum += self._batch_loss(x0, cond).item()
                val_avg = val_sum / len(val_dataloader)
                val_losses.append(val_avg)
                self.model.train()

                if storage is not None and self._model_tag is not None:
                    storage.intermediates.add_scalar(f"{self._model_tag}/val_mse_loss", epoch, val_avg)

                if val_avg < best_val:
                    best_val = val_avg
                    best_model_state = {
                        k: v.detach().clone() for k, v in self.model.state_dict().items()
                    }
                    if self.ema_model is not None:
                        best_ema_state = {
                            k: v.detach().clone() for k, v in self.ema_model.state_dict().items()
                        }
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if patience is not None and epochs_no_improve >= patience:
                        break

        if best_model_state is not None:
            # Keep the best-validating weights, not the final epoch's.
            self.model.load_state_dict(best_model_state)
            if self.ema_model is not None and best_ema_state is not None:
                self.ema_model.load_state_dict(best_ema_state)

        return epoch_losses
