import math

import torch
import torch.nn as nn

from dsbo.surrogates.surrogate_model import SurrogateModel

from tests.workflow.conftest import (
    SimpleModule,
    MultiOutputModule,
    _isolated_study_storage,
    _suppress_tqdm,
    _temp_workdir,
)


class TinyModelBase(SurrogateModel):
    def __init__(self):
        super().__init__()
        self.network: nn.Linear | None = None
        self.build_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self.predict_calls: list[torch.Tensor] = []
        self.output_shape: tuple[int, ...] = ()

    def build(self, output_shape, condition_shape) -> None:
        self.build_calls.append((tuple(output_shape), tuple(condition_shape)))
        self.output_shape = tuple(output_shape)
        self.network = nn.Linear(
            math.prod(condition_shape),
            math.prod(output_shape),
        )

    def _loss(self, batch):
        targets, conditions = batch
        prediction = self.predict(conditions)
        return nn.functional.mse_loss(prediction, targets.reshape_as(prediction))

    def configure_optimizers(self):
        return torch.optim.Adam(self.network.parameters(), lr=1e-2)

    def predict(self, conditions: torch.Tensor) -> torch.Tensor:
        self.predict_calls.append(conditions.detach().clone())
        return self.network(conditions.flatten(start_dim=1)).reshape(
            conditions.shape[0], *self.output_shape
        )


class TinySurrogateModel(TinyModelBase):
    def train_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss


class ValidatingTinySurrogateModel(TinySurrogateModel):
    def train_validation_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss


class TinyPretrainFinetuneModel(TinyModelBase):
    def pretrain_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def finetune_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss
