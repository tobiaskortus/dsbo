from __future__ import annotations

import torch
from typing import Any, Mapping, TYPE_CHECKING


from dsbo.surrogates import SurrogateBase, SurrogateStrategy
from lightning.pytorch.utilities import disable_possible_user_warnings



def reduce(result: dict[str, torch.Tensor] | ResultsBatch) -> dict[str, torch.Tensor] | ResultsBatch:
    if isinstance(result, ResultsBatch):
        return ResultsBatch([reduce(r) for r in result])

    return {k: v.mean() for k, v in result.items()}

def expand(result: dict[str, torch.Tensor] | ResultsBatch) -> dict[str, torch.Tensor] | ResultsBatch:
    if isinstance(result, ResultsBatch):
        return ResultsBatch([expand(r) for r in result])

    vector_lengths = [v.shape[0] for v in result.values() if v.ndim == 1 and v.shape[0] > 1]
    
    if vector_lengths:
        assert len(set(vector_lengths)) == 1, f"All vectors must have the same length, got lengths: {vector_lengths}"
        target_len = vector_lengths[0]
    else:
        raise Exception("No vector entries found in the result to determine target length for expansion.")
    
    expanded = {}
    for k, v in result.items():
        if v.ndim == 0 or (v.ndim == 1 and v.shape[0] == 1):
            expanded[k] = v.repeat(target_len)
        else:
            expanded[k] = v

    return expanded


def surrogate_predictions_to_dict(tensor: torch.Tensor, keys: list[str], n: int) -> dict[str, torch.Tensor]:
    if n > 1:
        return {k: tensor[:, i] for i, k in enumerate(keys)}
    
    return {k: tensor[i] for i, k in enumerate(keys)}


class WorkflowSurrogate(SurrogateBase):
    def __init__(self, surrogate: Any, strategy: SurrogateStrategy):
        super().__init__(surrogate, strategy)

    def __call__(self, design, **kwargs):
        pass


from typing import Any, Callable

import torch
from torch.utils.data import TensorDataset, random_split, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from dsbo.modules import Module
from dsbo.parameters import Candidate, CandidateBatch
from dsbo.surrogates import SurrogateBase, SurrogateStrategy
from dsbo.surrogates.workflow_surrogate import (
    WorkflowSurrogate,
    surrogate_predictions_to_dict,
    reduce,
)
from dsbo.workflow import ResultsBatch


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
        return DataLoader(self.train_dataset, batch_size=16, num_workers=1, persistent_workers=True, shuffle=True)  # type: ignore[attr-defined]

    def val_dataloader(self):
        if hasattr(self, "val_dataset") and self.val_dataset is not None:  # type: ignore[attr-defined]
            return DataLoader(self.val_dataset, batch_size=16, num_workers=1, persistent_workers=True)  # type: ignore[attr-defined]
        return None


class MLPWorkflowSurrogate(WorkflowSurrogate):
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


    @staticmethod
    def _filter_failed(
        candidates: CandidateBatch, results: ResultsBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        X = candidates.tensor
        Y = results.tensor

        # TODO: define a consistent "failed" definition
        return X, Y

    def to_datasets(self, candidates: CandidateBatch, results: ResultsBatch) -> tuple[TensorDataset, TensorDataset]:
        X, Y = self._filter_failed(candidates, results)
        total = X.shape[0]

        if total < 2:
            raise ValueError(
                f"Need at least 2 valid samples for training, got {total}."
            )

        val_size = max(1, int(total * self.val_split))
        train_size = total - val_size

        # Split indices
        indices = torch.randperm(total)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_dataset = TensorDataset(X[train_indices], Y[train_indices])
        val_dataset = TensorDataset(X[val_indices], Y[val_indices])

        return train_dataset, val_dataset

    # -- training ------------------------------------------------------------

    def train(
        self,
        candidate: Candidate,
        run_batch: Callable[[CandidateBatch], ResultsBatch],
    ) -> None:
        disable_possible_user_warnings()
  

        candidates = self.strategy.generate_candidates(
            candidate, space=candidate.space, n=self.sample_count
        )
        results = reduce(ResultsBatch(run_batch(candidates)))

        self.keys = list(results.keys)
        in_dim = candidate.space.dims()
        out_dim = len(self.keys)

        train_ds, val_ds = self.to_datasets(candidates, results)

        model = MLPSurrogateModule(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=self.hidden_dims,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        model.train_dataset = train_ds 
        model.val_dataset = val_ds

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            mode="min",
            verbose=False,
        )

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=[early_stop],
            enable_progress_bar=True,
            enable_checkpointing=False,
            logger=False,
        )
        trainer.fit(model)

        self.write_to_registry(model)

        return model


    def __call__(self, design: Candidate, **kwargs) -> dict[str, torch.Tensor]:
        x = design.tensor

        in_dim = x.shape[0] if x.ndim == 1 else x.shape[1]
        out_dim = len(self.keys)

        model = MLPSurrogateModule(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=self.hidden_dims,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        model.load_state_dict(self.load_from_registry())
        model.eval()

        y = model(x)

        return surrogate_predictions_to_dict(y, self.keys, self.n)