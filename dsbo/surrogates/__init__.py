from __future__ import annotations

import hashlib
from abc import abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

import dsbo
from dsbo.autograd import train_op
from dsbo.modules import Module, dynamic_submodule
from dsbo.parameters import Candidate, ParameterSpace
from dsbo.storage import get_current_storage
from dsbo.storage.artifact import ArtifactRef
from dsbo.surrogates.graph_transformations import GraphMode
from dsbo.surrogates.trainer import Trainer, TrainingConfig, save_checkpoint
from dsbo.utils.io import Output, Workdir
from dsbo.utils.io.workdir import DefaultPaths

if TYPE_CHECKING:
    from dsbo.surrogates.surrogate_model import SurrogateModel
    from dsbo.utils.cache import CacheConfig


class LifecycleStage(str, Enum):
    TRAIN = "train"
    PRETRAIN = "pretrain"
    FINETUNE = "finetune"



def _load_checkpoint(artifact_name: str, ref: ArtifactRef) -> dict:
    storage = get_current_storage()
    try:
        model_file = storage.artifacts.load_file(ref, "checkpoint.pt")
        checkpoint = torch.load(model_file, weights_only=True, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load checkpoint for surrogate '{artifact_name}' "
            f"from artifact {ref!r}."
        ) from exc
    return checkpoint


def _resolve_latest_model_ref(artifact_name: str) -> ArtifactRef:
    storage = get_current_storage()
    try:
        return storage.artifacts.load_artifact(artifact_name).ref
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"No model artifact exists for surrogate '{artifact_name}'."
        ) from exc


def _resolve_finetune_ref(artifact_name: str, initial_stage: LifecycleStage) -> ArtifactRef:
    stages = (
        (LifecycleStage.PRETRAIN,)
        if initial_stage is LifecycleStage.PRETRAIN
        else (LifecycleStage.FINETUNE, LifecycleStage.TRAIN)
    )
    storage = get_current_storage()
    for stage in stages:
        version = storage.artifacts.latest_version(artifact_name, stage=stage.value)
        if version is not None:
            return ArtifactRef(artifact_name, "model", version, stage.value)
    raise RuntimeError(
        f"Cannot finetune surrogate '{artifact_name}' without a prior "
        f"{initial_stage.value} artifact."
    )


def _has_artifact_for(artifact_name: str, *stages: LifecycleStage) -> bool:
    storage = get_current_storage()
    return any(
        storage.artifacts.latest_version(artifact_name, stage=stage.value) is not None
        for stage in stages
    )

def _use_cached_models() -> bool:
    from dsbo.utils.cache import CachePolicy

    cache_config: CacheConfig = dsbo.config.get("cache_config")
    if cache_config is None:
        return True
    if cache_config.cache_policy == CachePolicy.RETRAIN_SURROGATES:
        return False
    return cache_config.should_cache(GraphMode.DEFAULT)


class SurrogateStrategy:
    @abstractmethod
    def is_in_trust_region(self, candidate: Candidate) -> bool:
        ...

    @abstractmethod
    def generate_candidates(self,
        candidate: Candidate,
        space: ParameterSpace,
        finetune: bool = False,
    ) -> list[Candidate]:
        ...


class SurrogateBase(Module):
    def __init__(
        self,
        model: SurrogateModel,
        strategy: SurrogateStrategy,
        training_config: TrainingConfig,
    ):
        super().__init__()
        self.model = model
        self.strategy = strategy
        self.training_config = training_config

        stages = frozenset(stage for stage in LifecycleStage if model.implements(stage))
        initial_stages = stages & {LifecycleStage.TRAIN, LifecycleStage.PRETRAIN}
        if len(initial_stages) != 1:
            raise ValueError(
                f"{type(model).__name__} must implement exactly one initial "
                "train_step or pretrain_step."
            )
        self._initial_stage = next(iter(initial_stages))
        if (self._initial_stage is LifecycleStage.PRETRAIN
            and LifecycleStage.FINETUNE not in stages):
            raise ValueError(
                f"{type(model).__name__} implements pretrain_step and must also "
                "implement finetune_step."
            )

        self._artifact_name: str | None = None
        self.model_file_path: Output | None = None
        self._schema: dict[str, tuple[int, ...]] | None = None
        self._task_stages: dict[Module, LifecycleStage] = {}
        self._lifecycle_tasks: dict[LifecycleStage, Module] = {}
        for stage in stages:
            task = getattr(self, stage.value)
            if isinstance(task, Module):
                self._lifecycle_tasks[stage] = task
                self._task_stages[task] = stage

    def bind_artifact_name(self, target_name: str, target_identity: str) -> None:
        if self._artifact_name is not None:
            raise RuntimeError(
                f"Surrogate '{self._artifact_name}' is already bound and cannot be "
                "patched again."
            )

        model_type = f"{type(self.model).__module__}.{type(self.model).__qualname__}"
        suffix = hashlib.sha256(f"{model_type}|{target_identity}".encode()).hexdigest()[:6]
        self._artifact_name = f"{type(self.model).__name__}_{target_name}_{suffix}"
        self.model_file_path = Output(
            Workdir.register_file(
                DefaultPaths.SURROGATE,
                f"{self._artifact_name}.pt",
            )
        )

    @abstractmethod
    def build_training_data(
        self,
        stage: LifecycleStage,
        design,
        candidates,
    ) -> tuple[DataLoader, DataLoader | None, tuple[int, ...], tuple[int, ...]]:
        ...


    def _train(self, stage: LifecycleStage, design, candidates) -> dict:
        train_loader, val_loader, output_shape, condition_shape = (
            self.build_training_data(stage, design, candidates)
        )
        self._schema = {
            "output_shape": output_shape,
            "condition_shape": condition_shape,
        }
        self.model.stage = stage
        self.model.build(output_shape, condition_shape)

        if stage is LifecycleStage.FINETUNE:
            ref = _resolve_finetune_ref(self.artifact_name, self._initial_stage)
            checkpoint = _load_checkpoint(self.artifact_name, ref)
            self.model.restore(
                checkpoint["model"],
                checkpoint_stage=LifecycleStage(ref.stage),
                model_stage=LifecycleStage.FINETUNE,
            )

        trainer = Trainer(self.training_config, resources=self.resources)
        trainer.train(
            self.model,
            train_loader,
            val_loader=val_loader,
            model_tag=self.artifact_name,
        )
        if self.model_file_path is None:
            raise RuntimeError("Surrogate must be patched before training.")
        save_checkpoint(self.state_dict(), self.model_file_path())
        return {}

    @train_op
    @dynamic_submodule
    def train(self, design, **kwargs):
        return self._train(LifecycleStage.TRAIN, design, kwargs.get("candidates"))

    @train_op
    @dynamic_submodule
    def pretrain(self, design, **kwargs):
        return self._train(LifecycleStage.PRETRAIN, design, kwargs.get("candidates"))

    @train_op
    @dynamic_submodule
    def finetune(self, design, **kwargs):
        return self._train(LifecycleStage.FINETUNE, design, kwargs.get("candidates"))

    def generate_training_candidates(
        self,
        candidate: Candidate,
        training_task: Module,
    ):
        finetune = self._lifecycle_tasks.get(LifecycleStage.FINETUNE) is training_task
        return self.strategy.generate_candidates(
            candidate,
            candidate.space,
            finetune=finetune,
        )

    def _auxiliary_state(self) -> dict:
        return {}

    def _load_auxiliary_state(self, state: dict, schema: dict) -> None:
        pass

    def state_dict(self) -> dict:
        if self._schema is None:
            raise RuntimeError(
                f"Surrogate '{self.artifact_name}' has not been built and has no checkpoint schema."
            )
        return {
            "format_version": 1,
            "schema": dict(self._schema),
            "model": self.model.state_dict(),
            "auxiliary": self._auxiliary_state(),
        }

    def restore_model(self) -> SurrogateModel:
        ref = _resolve_latest_model_ref(self.artifact_name)
        checkpoint = _load_checkpoint(self.artifact_name, ref)
        checkpoint_stage = LifecycleStage(ref.stage)
        schema = checkpoint["schema"]
        self.model.stage = checkpoint_stage
        self.model.build(schema["output_shape"], schema["condition_shape"])
        self.model.restore(
            checkpoint["model"],
            checkpoint_stage=checkpoint_stage,
            model_stage=checkpoint_stage,
        )
        self._load_auxiliary_state(checkpoint["auxiliary"], schema)
        self._schema = dict(schema)
        return self.model

    def publish_trained_model(
        self,
        model_file: str,
        stage: str = LifecycleStage.TRAIN.value,
    ) -> ArtifactRef:
        storage = get_current_storage()
        artifact = storage.artifacts.create_artifact(self.artifact_name, "model")
        artifact.add_file(model_file, name="checkpoint.pt")
        return storage.artifacts.save_artifact(artifact, stage=stage).ref

    def patch(self, surrogate: SurrogateBase):
        raise RuntimeError(
            f"A surrogate ({type(self).__name__}) cannot be patched with another "
            "surrogate; surrogates cannot contain surrogates."
        )

    def is_in_trust_region(self, candidate: Candidate) -> bool:
        if not self.has_inference_model:
            return False
        return self.strategy.is_in_trust_region(candidate)

    def training_stage(self, task: Module) -> str:
        stage = self._task_stages.get(task)
        if stage is None:
            raise ValueError(
                f"{task.name} is not a lifecycle task of {type(self).__name__}."
            )
        return stage.value

    def select_training_task(self, candidate: Candidate) -> Module | None:
        if _use_cached_models() and self.is_in_trust_region(candidate):
            return None
        return self._lifecycle_tasks[self._get_next_stage()]

    def _get_next_stage(self) -> LifecycleStage:
        if not self.has_initial_model:
            return self._initial_stage
        if LifecycleStage.FINETUNE in self._lifecycle_tasks:
            return LifecycleStage.FINETUNE
        return self._initial_stage

    @property
    def is_available(self) -> bool:
        from dsbo.utils.cache import CachePolicy

        cache_config: CacheConfig | None = dsbo.config.get("cache_config")
        exceptions = (CachePolicy.RETRAIN_SURROGATES, CachePolicy.ALWAYS_EXECUTE)
        if cache_config is None or cache_config.cache_policy not in exceptions:
            return self.has_inference_model
        return False

    @property
    def has_initial_model(self) -> bool:
        return _has_artifact_for(self.artifact_name, self._initial_stage)

    @property
    def is_finetuned(self) -> bool:
        return _has_artifact_for(self.artifact_name, LifecycleStage.FINETUNE)

    @property
    def has_inference_model(self) -> bool:
        if self._initial_stage is LifecycleStage.PRETRAIN:
            return self.is_finetuned
        return self.has_initial_model

    @property
    def artifact_name(self) -> str:
        if self._artifact_name is None:
            raise RuntimeError(
                "Surrogate has no artifact name; patch it onto a module or workflow "
                "before use."
            )
        return self._artifact_name
