from __future__ import annotations

import torch
from diffusers.schedulers import DDPMScheduler, DDIMScheduler
from typing import Any, Mapping, TYPE_CHECKING
from torch.utils.data import Subset, DataLoader, default_collate

from dsbo.autograd import train_op
from dsbo.utils.io import Input, Workdir
from dsbo.modules import dynamic_submodule
from dsbo.config import get_active_file_backend
from dsbo.utils.io.datasets import design_split
from dsbo.surrogates import SurrogateBase, SurrogateStrategy
from dsbo.surrogates.diffusion import MlpDiffusion1D, Trainer
from dsbo.utils.io.workdir import AbstractFilePath, DefaultFilenames, DefaultPaths


if TYPE_CHECKING:
    from dsbo.parameters import CandidateBatch


def _collate_workflow_surrogate_batch(batch):
    x, y = default_collate(batch)
    return y, x


def surrogate_predictions_to_dict(tensor: torch.Tensor, keys: list[str], n: int) -> dict[str, torch.Tensor]:
    tensor = tensor.reshape(-1, len(keys))
    return {k: tensor[:, i] for i, k in enumerate(keys)}


class WorkflowSurrogate(SurrogateBase):
    requires_design_tensor = True

    def __init__(self, strategy: SurrogateStrategy):
        super().__init__(strategy=strategy)
        self._data_path = Input(Workdir.register_file(
                DefaultPaths.TRAINING_RECORDS,
                DefaultFilenames.RESULTS,
            )
        )

    def __call__(self, design, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement __call__ to predict objectives."
        )

    def backward(self, design, grad_output=None, design_tensor=None) -> None:
        if grad_output is None:
            return
        keys = list(grad_output.keys())
        results = self(design, design_tensor=design_tensor)
        tensors = [results[k] for k in keys]
        grad_tensors = [grad_output[k] for k in keys]
        torch.autograd.backward(tensors, grad_tensors)

    def get_train_val_loaders(
        self,
        training_data_path: str,
        candidates: CandidateBatch,
    ) -> tuple[DataLoader, DataLoader]:
        from dsbo.utils.io.datasets.datasets import WorkflowSurrogateDataset

        dataset = WorkflowSurrogateDataset(
            training_data_path,
            candidates.tensor.detach(),
            backend=get_active_file_backend(),
        )
        self._records_dataset = dataset
        self.keys = dataset.keys
        self.n = dataset.n

        # Split by candidate *before* fitting normalization, then compute the
        # moments over the training rows only. Both loaders below reuse these
        # train-derived constants, so validation data never leaks into the
        # training normalization (and is normalized identically at inference).
        train_indices, val_indices = design_split(
            n_candidates=candidates.tensor.shape[0],
            rows_per_design=self.n,
            val_split=self.val_split,
            seed=self.seed,
        )
        dataset.compute_stats(train_indices)
        self._mean_x = dataset.mean_x
        self._std_x = dataset.std_x
        self._mean_y = dataset.mean_y
        self._std_y = dataset.std_y

        num_workers = getattr(self, "num_workers", 8)
        loader_options = {
            "batch_size": self.batch_size,
            "num_workers": num_workers,
            "collate_fn": _collate_workflow_surrogate_batch,
        }
        if num_workers:
            loader_options.update(
                persistent_workers=True,
                prefetch_factor=getattr(self, "prefetch_factor", 4),
            )
        return (
            DataLoader(Subset(dataset, train_indices), shuffle=True, **loader_options),
            DataLoader(Subset(dataset, val_indices), shuffle=False, **loader_options),
        )

    @property
    def data_path(self) -> AbstractFilePath:   
        return self._data_path


class DefaultWorkflowSurrogate(WorkflowSurrogate):
    def __init__(
        self,
        strategy: SurrogateStrategy,
        emb_dim: int = 256,
        hidden_dim: int = 256,
        n_layers: int = 12,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        max_epochs: int = 200,
        batch_size: int = 256,
        grad_clip: float = 1.0,
        val_split: float = 0.2,
        sample_count: int = 16,
        num_train_timesteps: int = 1000,
        num_infer_timesteps: int = 50,
        eta: float = 0.8,
        seed: int = 0,
        patience: int | None = None,
        num_workers: int = 8,
        prefetch_factor: int = 4,
    ):

        super().__init__(strategy=strategy)
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if prefetch_factor < 1:
            raise ValueError("prefetch_factor must be positive.")

        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.val_split = val_split
        self.sample_count = sample_count
        self.num_train_timesteps = num_train_timesteps
        self.num_infer_timesteps = num_infer_timesteps
        self.eta = eta
        self.seed = seed
        self.patience = patience
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self.keys: list[str] = []
        self.n: int = 1
        self._mean_x: torch.Tensor | None = None
        self._std_x: torch.Tensor | None = None
        self._mean_y: torch.Tensor | None = None
        self._std_y: torch.Tensor | None = None
        self._diffusion_model: MlpDiffusion1D | None = None
        self._noise_scheduler: DDPMScheduler | None = None
        self._sample_scheduler: DDIMScheduler | None = None

    def _create_schedulers(self) -> None:
        self._noise_scheduler = DDPMScheduler(
            clip_sample=False,
            num_train_timesteps=self.num_train_timesteps,
            prediction_type="v_prediction",
            beta_schedule="squaredcos_cap_v2",
        )
        self._sample_scheduler = DDIMScheduler(
            clip_sample=False,
            num_train_timesteps=self.num_train_timesteps,
            prediction_type="v_prediction",
            beta_schedule="squaredcos_cap_v2",
        )

    @train_op
    @dynamic_submodule
    def train(self, design, candidates):
        data_path = self.data_path()
        model_file_path = self.model_file_path()

        train_loader, val_loader = self.get_train_val_loaders(data_path, candidates)
        try:
            objective_dim = len(self.keys)
            parameter_dim = candidates.tensor.shape[1]

            model = MlpDiffusion1D(
                objective_dim=objective_dim,
                parameter_dim=parameter_dim,
                emb_dim=self.emb_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_layers,
                dropout=self.dropout,
            )
            self._diffusion_model = model
            self._create_schedulers()

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs, eta_min=1e-6
            )

            Trainer(model,
                self._noise_scheduler,
                optimizer,
                lr_scheduler,
                device=next(model.parameters()).device,
                grad_clip=self.grad_clip,
                model_tag=type(self).__name__.lower(),
            ).fit(
                n_epochs=self.max_epochs,
                dataloader=train_loader,
                val_dataloader=val_loader,
                patience=self.patience,
                show_progress=False,
            )

            torch.save(self.state_dict(), model_file_path)
        finally:
            # Always release the records HDF5 handle, even if training errors.
            if getattr(self, "_records_dataset", None) is not None:
                self._records_dataset.close()
                self._records_dataset = None
        return {}


    def state_dict(self) -> dict:
        if self._diffusion_model is None:
            return {}
        
        return {
            "model": self._diffusion_model.state_dict(),
            "objective_dim": self._diffusion_model.config.objective_dim,
            "parameter_dim": self._diffusion_model.cond_dim,
            "mean_x": self._mean_x,
            "std_x": self._std_x,
            "mean_y": self._mean_y,
            "std_y": self._std_y,
            "keys": self.keys,
            "n": self.n,
        }

    def from_state_dict(self, state: dict) -> None:
        model = MlpDiffusion1D(
            objective_dim=state["objective_dim"],
            parameter_dim=state["parameter_dim"],
            emb_dim=self.emb_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )
        model.load_state_dict(state["model"])
        model.eval()
        self.keys = state["keys"]
        self.n = state["n"]
        self._mean_x = state.get("mean_x")
        self._std_x = state.get("std_x")
        self._mean_y = state.get("mean_y")
        self._std_y = state.get("std_y")
        self._diffusion_model = model
        self.surrogate = model
        self._create_schedulers()

    def __call__(self, design: Mapping[str, Any], design_tensor: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        self.get_runtime_model()

        model = self._diffusion_model
        if model is None:
            raise RuntimeError(f"{type(self).__name__} has no trained surrogate model.")

        B = self.sample_count
        c = design_tensor.float()
        if self._mean_x is not None:
            c = (design_tensor - self._mean_x) / self._std_x
        c = c.unsqueeze(0).expand(B, -1)

        model.eval()
        self._sample_scheduler.set_timesteps(self.num_infer_timesteps)
        objective_dim = model.config.objective_dim

        x = torch.randn(B, 1, objective_dim, device=c.device)

        for t in self._sample_scheduler.timesteps:
            v = model(x, t.expand(B).to(c.device), c).sample
            x = self._sample_scheduler.step(v, t, x, eta=self.eta).prev_sample

        x = x.squeeze(1)
        if self._mean_y is not None and self._std_y is not None:
            x = x * self._std_y + self._mean_y

        return surrogate_predictions_to_dict(x, self.keys, B)
