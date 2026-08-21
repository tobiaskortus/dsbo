from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

import lightning as L
import torch

from dsbo.surrogates import LifecycleStage


class SurrogateModel(L.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage: LifecycleStage | None = None

    @abstractmethod
    def build(self, output_shape: tuple[int, ...], condition_shape: tuple[int, ...]) -> None:
        ...

    @abstractmethod
    def predict(self, c: torch.Tensor | None) -> torch.Tensor:
        ...

    def training_step(self, batch, batch_idx):
        if self.stage is None:
            raise RuntimeError("SurrogateModel.stage must be set before training.")
        return getattr(self, f"{self.stage.value}_step")(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        if self.stage is None:
            raise RuntimeError("SurrogateModel.stage must be set before validation.")
        return getattr(self, f"{self.stage.value}_validation_step")(batch, batch_idx)

    def train_step(self, batch, batch_idx):
        raise NotImplementedError()

    def pretrain_step(self, batch, batch_idx):
        raise NotImplementedError()

    def finetune_step(self, batch, batch_idx):
        raise NotImplementedError()

    def train_validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def pretrain_validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def finetune_validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def load_state_transform(self, weights: Mapping[str, Any]) -> None:
        self.load_state_dict(weights)

    def finetune_pre_transform(self) -> None:
        pass

    def finetune_post_transform(self) -> None:
        pass

    def restore(
        self,
        weights: Mapping[str, Any],
        checkpoint_stage: LifecycleStage,
        model_stage: LifecycleStage,
    ) -> None:
        if model_stage is not LifecycleStage.FINETUNE:
            self.load_state_transform(weights)
            return

        if checkpoint_stage is LifecycleStage.FINETUNE:
            self.finetune_pre_transform()
            self.load_state_transform(weights)
            self.finetune_post_transform()
            return

        self.load_state_transform(weights)
        self.finetune_pre_transform()
        self.finetune_post_transform()

    def implements(self, stage: LifecycleStage) -> bool:
            method_name = f"{stage.value}_step"
            return getattr(type(self), method_name) is not getattr(
                SurrogateModel,
                method_name,
            )
    
    def implements_validation(self, stage: LifecycleStage | None = None) -> bool:
        stage = self.stage if stage is None else stage
        if stage is None:
            return False
        method_name = f"{stage.value}_validation_step"
        return getattr(type(self), method_name) is not getattr(
            SurrogateModel,
            method_name,
        )
