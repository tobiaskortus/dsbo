from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from dsbo.utils.io import Input, LazyShape
from dsbo.config import get_active_file_backend
from dsbo.utils.io.storage.autograd import AutogradFileInterface
from dsbo.utils.io.datasets import ModuleSurrogateDataset
from dsbo.surrogates import SurrogateBase, SurrogateStrategy, LifecycleStage
from dsbo.surrogates.utils import Standardizer
from dsbo.surrogates.surrogate_model import SurrogateModel
from dsbo.surrogates.trainer import TrainingConfig
from dsbo.utils.io.workdir import AbstractFilePath, PathAnnotation

if TYPE_CHECKING:
    from dsbo.parameters import CandidateBatch


class ModuleSurrogateContract:
    data_file: AbstractFilePath | PathAnnotation | None
    output_dim: tuple[int, ...] | LazyShape
    output_samples: int
    conditional_dim: tuple[int, ...] | LazyShape | None
    conditional_file: AbstractFilePath | PathAnnotation | None
    output_conditional_fn: Callable[[dict], torch.Tensor] | None

    def __init__(
        self,
        output_dim: tuple[int, ...] | LazyShape,
        output_samples: int,
        conditional_dim: tuple[int, ...] | LazyShape | None = None,
        output_conditional_fn: Callable[[dict], torch.Tensor] | None = None,
        data_file: AbstractFilePath | PathAnnotation | None = None,
        conditional_file: AbstractFilePath | PathAnnotation | None = None,
    ):
        self.data_file = data_file
        self.output_dim = output_dim
        self.output_samples = output_samples
        self.conditional_dim = conditional_dim
        self.conditional_file = conditional_file
        self.output_conditional_fn = output_conditional_fn

    @staticmethod
    def _abstract_path(file: AbstractFilePath | PathAnnotation) -> AbstractFilePath:
        return file.path if isinstance(file, PathAnnotation) else file

    def to_data_loader(self, training_config: TrainingConfig) -> DataLoader:
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
        loader_options = {
            "batch_size": training_config.batch_size,
            "num_workers": training_config.num_workers,
        }
        if training_config.num_workers:
            loader_options.update(
                persistent_workers=True,
                prefetch_factor=training_config.prefetch_factor,
            )
        loader = DataLoader(dataset, **loader_options)
        return loader

    def write_output(self, design: dict, output_tensor: torch.Tensor) -> None:
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


class ModuleSurrogate(SurrogateBase):
    def __init__(
        self,
        model: SurrogateModel,
        strategy: SurrogateStrategy,
        training: TrainingConfig,
    ):
        super().__init__(model, strategy, training)
        self.contract: ModuleSurrogateContract | None = None
        self.condition_standardizer: Standardizer | None = None
        self.output_standardizer: Standardizer | None = None

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
        loader = self.contract.to_data_loader(self.training_config)
        dataset = loader.dataset
        dataset.compute_stats()
        self._train_dataset = dataset
        self.condition_standardizer = Standardizer(dataset.mean_c, dataset.std_c)
        self.output_standardizer = Standardizer(dataset.mean_x, dataset.std_x)
        return (
            loader,
            None,
            tuple(self.contract.output_dim),
            tuple(self.contract.conditional_dim),
        )

    def register_contract(self, contract: ModuleSurrogateContract) -> None:
        if self.contract is not None:
            raise RuntimeError(
                f"Surrogate '{self.artifact_name}' is already bound to another module contract."
            )

        self.contract = contract
        # Register the required input files for the module.
        for attr in ("data_file", "conditional_file"):
            path = getattr(contract, attr)
            if path is not None:
                setattr(self, attr, Input(ModuleSurrogateContract._abstract_path(path)))

    def __call__(self, design: dict, **kwargs) -> dict[str, torch.Tensor]:
        self.restore_model()
        cond = self.contract.output_conditional_fn(design) \
            if self.contract.output_conditional_fn is not None \
            else torch.tensor(list(design.values()), dtype=torch.float32)
        
        if cond.ndim == 1: 
            cond = cond.unsqueeze(0)

        if cond.shape[0] != self.contract.output_samples:
            raise ValueError(
                f"Module surrogate '{self.artifact_name}' received {cond.shape[0]} "
                f"condition rows; contract.output_samples is {self.contract.output_samples}."
            )

        cond = self.condition_standardizer.normalize(cond.float())

        with torch.no_grad():
            x = self.model.predict(cond)

        expected = (cond.shape[0], *tuple(self.contract.output_dim))
        if x.shape != expected:
            raise ValueError(
                f"{type(self.model).__name__}.predict returned shape {tuple(x.shape)}; "
                f"expected {expected}."
            )

        x = self.output_standardizer.denormalize(x)
        self.contract.write_output(design, x)

    def _auxiliary_state(self) -> dict:
        if self.condition_standardizer is None or self.output_standardizer is None:
            raise RuntimeError("Module surrogate standardizers are not initialized.")
        
        return {
            "condition_standardizer": self.condition_standardizer.state_dict(),
            "output_standardizer": self.output_standardizer.state_dict(),
        }

    def _load_auxiliary_state(self, state: dict, schema: dict) -> None:
        if isinstance(self.contract.output_dim, LazyShape):
            self.contract.output_dim.set(schema["output_shape"])
        if isinstance(self.contract.conditional_dim, LazyShape):
            self.contract.conditional_dim.set(schema["condition_shape"])

        self.condition_standardizer = Standardizer(**state["condition_standardizer"])
        self.output_standardizer = Standardizer(**state["output_standardizer"])
