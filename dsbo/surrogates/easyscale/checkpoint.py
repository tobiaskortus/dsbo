from __future__ import annotations

import torch
import pickle
import lightning as L
from typing import Any
from pathlib import Path

from dsbo.surrogates.easyscale.strategy import EasyScaleStrategy
from dsbo.surrogates.easyscale.random import VirtualRankContext, _capture_random_state, _restore_random_state


class EasyScaleCheckpointCallback(L.Callback):
    def __init__(
        self,
        strategy: EasyScaleStrategy,
        checkpoint_path: str,
        rank_zero_state=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.checkpoint_path = checkpoint_path
        self.rank_zero_state = rank_zero_state
        self.saved = False

    @property
    def state_key(self) -> str:
        return "dsbo.easyscale.est-contexts"

    def state_dict(self) -> dict[str, Any]:
        contexts = self.strategy.all_contexts or tuple(self.strategy.contexts.values())
        serialized_state = pickle.dumps(
            {
                "contexts": contexts,
                "rank_zero_state": self.rank_zero_state,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        return {"serialized_state": torch.tensor(list(serialized_state), dtype=torch.uint8)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        tensor = state_dict.get("serialized_state")        
        serialized_state = pickle.loads(bytes(tensor.cpu().tolist()))
        restored = {context.rank: context for context in serialized_state["contexts"]}
        
        for rank, context in self.strategy.contexts.items():
            source = restored[rank]
            context.random_state = source.random_state
            context.sampler_epoch = source.sampler_epoch
            context.sampler_position = source.sampler_position
        self.rank_zero_state = serialized_state.get("rank_zero_state")

    def on_train_start(self, trainer, pl_module) -> None:
        if self.rank_zero_state is not None:
            _restore_random_state(
                self.rank_zero_state,
                self.strategy.root_device,
            )


    def save(self, trainer) -> None:
        self.strategy.gather_contexts()
        self.rank_zero_state = (
            _capture_random_state(self.strategy.root_device)
            if self.strategy.is_global_zero
            else None
        )
        Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(self.checkpoint_path)
        self.strategy.barrier()
        self.saved = True

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if trainer.current_epoch + 1 < trainer.max_epochs and not trainer.should_stop:
            return
        self.save(trainer)
