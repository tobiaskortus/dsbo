from __future__ import annotations

import torch
import pytorch_lightning as pl
from typing import Any, Mapping, TYPE_CHECKING
from torch.utils.data import Subset, random_split, DataLoader
from pytorch_lightning.callbacks import EarlyStopping, Callback

from dsbo.utils.io import Input, Workdir
from dsbo.modules import dynamic_submodule
from dsbo.config import get_active_file_backend
from dsbo.surrogates import SurrogateBase, SurrogateStrategy
from dsbo.utils.io.workdir import DefaultFilenames, DefaultPaths


if TYPE_CHECKING:
    from dsbo.parameters import CandidateBatch

def surrogate_predictions_to_dict(tensor: torch.Tensor, keys: list[str], n: int) -> dict[str, torch.Tensor]:
    tensor = tensor.reshape(-1, len(keys))
    return {k: tensor[:, i] for i, k in enumerate(keys)}


class WorkflowSurrogate(SurrogateBase):
    requires_design_tensor = True

    def __init__(self, surrogate: Any, strategy: SurrogateStrategy):
        super().__init__(surrogate, strategy)

    def __call__(self, design, **kwargs):
        pass

    def to_datasets(self, training_data_path: str, candidates: CandidateBatch) -> tuple[Subset, Subset]:
            from dsbo.utils.io.datasets.datasets import WorkflowSurrogateDataset

            dataset = WorkflowSurrogateDataset(
                training_data_path,
                candidates.tensor,
                backend=get_active_file_backend(),
            )
            self._records_dataset = dataset
            self.keys = dataset.keys
            self.n = dataset.n
            val_size = max(1, int(len(dataset) * self.val_split))
            train_size = len(dataset) - val_size

            return random_split(dataset, [train_size, val_size])


class _EpochLoggingCallback(Callback):
    """WorkflowTaskLogger captures stdout into the task log file.
    NOTE: Hopefully only a temporary fix"""

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss")
        val_loss = metrics.get("val_loss")
        epoch = trainer.current_epoch
        parts = [f"Epoch {epoch}"]
        if train_loss is not None:
            parts.append(f"train_loss={float(train_loss):.6f}")
        if val_loss is not None:
            parts.append(f"val_loss={float(val_loss):.6f}")
        print(" | ".join(parts))


class MLPSurrogateModule(pl.LightningModule):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: tuple[int, ...] = (64, 64),
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        layers: list[torch.nn.Module] = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.extend([torch.nn.Linear(prev_dim, h_dim), torch.nn.ReLU()])
            prev_dim = h_dim
        layers.append(torch.nn.Linear(prev_dim, out_dim))
        self.mlp = torch.nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=16, num_workers=0, shuffle=True)  # type: ignore[attr-defined]

    def val_dataloader(self):
        if hasattr(self, "val_dataset") and self.val_dataset is not None:  # type: ignore[attr-defined]
            return DataLoader(self.val_dataset, batch_size=16, num_workers=0)  # type: ignore[attr-defined]
        return None


class DefaultWorkflowSurrogate(WorkflowSurrogate):
    hidden_dims: tuple[int, ...] = (64, 64)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 200
    patience: int = 20
    val_split: float = 0.2
    train_batch_size: int = 16
    sample_count: int = 16

    def __init__(
        self,
        strategy: SurrogateStrategy,
        hidden_dims: tuple[int, ...] | None = None,
        learning_rate: float | None = None,
        weight_decay: float | None = None,
        max_epochs: int | None = None,
        patience: int | None = None,
        val_split: float | None = None,
        sample_count: int | None = None,
    ):
        # Build a dummy surrogate (will be replaced during training)
        dummy = torch.nn.Linear(1, 1)
        super().__init__(surrogate=dummy, strategy=strategy)

        self.data_path = Input(
            Workdir.register_file(DefaultPaths.TRAINING_RECORDS, DefaultFilenames.RESULTS)
        )

        if hidden_dims is not None:
            self.hidden_dims = hidden_dims
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if weight_decay is not None:
            self.weight_decay = weight_decay
        if max_epochs is not None:
            self.max_epochs = max_epochs
        if patience is not None:
            self.patience = patience
        if val_split is not None:
            self.val_split = val_split
        if sample_count is not None:
            self.sample_count = sample_count

        # Populated during training
        self.keys: list[str] = []
        self.n: int = 1
        self._mean_x: torch.Tensor | None = None
        self._std_x: torch.Tensor | None = None
        self._mean_y: torch.Tensor | None = None
        self._std_y: torch.Tensor | None = None

    @dynamic_submodule
    def train(self, design, candidates):
        data_path = self.data_path()
        model_file_path = self.model_file_path()

        train_ds, val_ds = self.to_datasets(data_path, candidates)

        self._records_dataset.compute_stats()
        self._mean_x = self._records_dataset.mean_x
        self._std_x = self._records_dataset.std_x
        self._mean_y = self._records_dataset.mean_y
        self._std_y = self._records_dataset.std_y

        in_dim = train_ds[0][0].shape[0]
        out_dim = train_ds[0][1].shape[0]

        model = MLPSurrogateModule(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=self.hidden_dims,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        model.train_dataset = train_ds 
        model.val_dataset = val_ds

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            mode="min",
            verbose=False,
        )

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=[early_stopping, _EpochLoggingCallback()],
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
        )

        try:
            trainer.fit(model)
            torch.save(self._model_state(model), model_file_path)
        finally:
            # Always release the records HDF5 handle, even if training errors.
            if getattr(self, "_records_dataset", None) is not None:
                self._records_dataset.close()
                self._records_dataset = None
        return {}

    def _model_state(self, model) -> dict:
        return {
            "state_dict": model.state_dict(),
            "in_dim": model.hparams.in_dim,
            "out_dim": model.hparams.out_dim,
            "keys": self.keys,
            "n": self.n,
            "mean_x": self._mean_x,
            "std_x": self._std_x,
            "mean_y": self._mean_y,
            "std_y": self._std_y,
        }

    def state_dict(self) -> dict:
        return self._model_state(self.surrogate)

    def from_state_dict(self, state: dict) -> None:
        model = MLPSurrogateModule(
            in_dim=state["in_dim"],
            out_dim=state["out_dim"],
            hidden_dims=self.hidden_dims,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        model.load_state_dict(state["state_dict"])
        model.eval()
        self.keys = state["keys"]
        self.n = state["n"]
        self._mean_x = state.get("mean_x")
        self._std_x = state.get("std_x")
        self._mean_y = state.get("mean_y")
        self._std_y = state.get("std_y")
        self.surrogate = model


    def __call__(self, design: Mapping[str, Any], design_tensor: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        model = self.get_runtime_model()
        x = design_tensor
        if self._mean_x is not None:
            x = (design_tensor - self._mean_x) / self._std_x
        x = x.unsqueeze(0).repeat(self.n, 1)
        y = model(x)
        if self._mean_y is not None:
            y = y * self._std_y + self._mean_y

        return surrogate_predictions_to_dict(y, self.keys, self.n)

    def backward(self, design, grad_output=None, design_tensor=None) -> None:
        if grad_output is None:
            return
        keys = list(grad_output.keys())
        results = self(design, design_tensor=design_tensor)
        tensors = [results[k] for k in keys]
        grad_tensors = [grad_output[k] for k in keys]
        torch.autograd.backward(tensors, grad_tensors)
