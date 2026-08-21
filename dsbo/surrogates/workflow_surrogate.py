from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from torch.utils.data import Subset, DataLoader, default_collate

from dsbo.utils.io import Input, Workdir
from dsbo.config import get_active_file_backend
from dsbo.utils.io.datasets import design_split
from dsbo.surrogates import SurrogateBase, SurrogateStrategy, LifecycleStage
from dsbo.surrogates.utils import Standardizer
from dsbo.surrogates.surrogate_model import SurrogateModel
from dsbo.surrogates.trainer import TrainingConfig
from dsbo.utils.io.workdir import AbstractFilePath, DefaultFilenames, DefaultPaths

if TYPE_CHECKING:
    from dsbo.parameters import CandidateBatch


def _collate_workflow_surrogate_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = default_collate(batch)
    return y, x


class WorkflowSurrogate(SurrogateBase):
    requires_design_tensor = True

    def __init__(
        self,
        model: SurrogateModel,
        strategy: SurrogateStrategy,
        training: TrainingConfig,
        sample_count: int | None = None,
    ):
        super().__init__(model, strategy, training)
        if sample_count is not None and sample_count <= 0:
            raise ValueError("sample_count must be positive.")

        self.sample_count: int | None = sample_count
        self.keys: list[str] = []
        self.condition_standardizer: Standardizer | None = None
        self.output_standardizer: Standardizer | None = None
        self._data_path = Input(Workdir.register_file(
                DefaultPaths.TRAINING_RECORDS,
                DefaultFilenames.RESULTS,
            )
        )

    def build_training_data(
        self,
        stage: LifecycleStage,
        design: dict,
        candidates: CandidateBatch,
    ) -> tuple[
        DataLoader,
        DataLoader | None,
        tuple[int, ...],
        tuple[int, ...],
    ]:
        from dsbo.utils.io.datasets.datasets import WorkflowSurrogateDataset

        data_path = self.data_path() if callable(self.data_path) else self.data_path
        dataset = WorkflowSurrogateDataset(
            data_path,
            candidates.tensor.detach(),
            backend=get_active_file_backend(),
        )
        self._records_dataset = dataset
        self.keys = dataset.keys
        n = dataset.n

        # Split by candidate *before* fitting normalization, then compute the
        # moments over the training rows only. Both loaders below reuse these
        # train-derived constants, so validation data never leaks into the
        # training normalization (and is normalized identically at inference).
        with_validation = self.model.implements_validation(stage)
        if with_validation and self.training_config.val_split > 0:
            train_indices, val_indices = design_split(
                n_candidates=candidates.tensor.shape[0],
                rows_per_design=n,
                val_split=self.training_config.val_split,
                seed=self.training_config.seed,
            )
        else:
            train_indices = list(range(len(dataset)))
            val_indices = []
        if self.sample_count is None:
            self.sample_count = n
        dataset.compute_stats(train_indices)
        self.condition_standardizer = Standardizer(dataset.mean_x, dataset.std_x)
        self.output_standardizer = Standardizer(dataset.mean_y, dataset.std_y)

        num_workers = self.training_config.num_workers
        loader_options = {
            "batch_size": self.training_config.batch_size,
            "num_workers": num_workers,
            "collate_fn": _collate_workflow_surrogate_batch,
        }
        if num_workers:
            loader_options.update(
                persistent_workers=True,
                prefetch_factor=self.training_config.prefetch_factor,
            )
        train_loader = DataLoader(Subset(dataset, train_indices), shuffle=True, **loader_options)
        val_loader = (
            DataLoader(Subset(dataset, val_indices), shuffle=False, **loader_options)
            if val_indices
            else None
        )
        return (
            train_loader,
            val_loader,
            (len(self.keys),),
            tuple(candidates.tensor.shape[1:]),
        )

    def __call__(self, design: dict, design_tensor: torch.Tensor | None = None, **kwargs) -> dict[str, torch.Tensor]:
        c = self.condition_standardizer.normalize(design_tensor.float())
        c = c.unsqueeze(0).repeat(self.sample_count, *([1] * c.ndim))

        x = self.model.predict(c)

        expected = (self.sample_count, len(self.keys))
        if x.shape != expected:
            raise ValueError(
                f"{type(self.model).__name__}.predict returned shape {tuple(x.shape)}; "
                f"expected {expected}."
            )

        x = self.output_standardizer.denormalize(x)
        return {key: x[:, i] for i, key in enumerate(self.keys)}

    def backward(
        self,
        design: dict,
        grad_output: dict[str, torch.Tensor] | None = None,
        design_tensor: torch.Tensor | None = None,
    ) -> None:
        if grad_output is None:
            return
        keys = list(grad_output.keys())
        results = self(design, design_tensor=design_tensor)
        tensors = [results[k] for k in keys]
        grad_tensors = [grad_output[k] for k in keys]
        torch.autograd.backward(tensors, grad_tensors)

    def _auxiliary_state(self) -> dict:
        if self.condition_standardizer is None or self.output_standardizer is None:
            raise RuntimeError("Workflow surrogate standardizers are not initialized.")
        # Workflow-specific metadata only (keys/standardization); model weights live
        # in the model blob that the base stores under ``state["model"]``.
        return {
            "keys": self.keys,
            "sample_count": self.sample_count,
            "condition_standardizer": self.condition_standardizer.state_dict(),
            "output_standardizer": self.output_standardizer.state_dict(),
        }

    def _load_auxiliary_state(self, state: dict, schema: dict) -> None:
        self.keys = state["keys"]
        self.sample_count = state["sample_count"]
        self.condition_standardizer = Standardizer(**state["condition_standardizer"])
        self.output_standardizer = Standardizer(**state["output_standardizer"])

    @property
    def data_path(self) -> AbstractFilePath:
        return self._data_path
