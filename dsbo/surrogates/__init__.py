from __future__ import annotations

from enum import Enum
from pathlib import Path
from abc import abstractmethod
from typing import BinaryIO, TYPE_CHECKING

import torch

import dsbo
from dsbo.utils.io import Workdir, Output
from dsbo.storage import get_current_storage
from dsbo.storage.artifact import ArtifactRef
from dsbo.utils.io.workdir import DefaultPaths
from dsbo.modules import Module, dynamic_submodule
from dsbo.parameters import ParameterSpace, Candidate
from dsbo.surrogates.graph_transformations import GraphMode

if TYPE_CHECKING:
    from dsbo.utils.cache import CacheConfig


class LifecycleStage(Enum):
    TRAIN = ("train", "trained")
    PRETRAIN = ("pretrain", "pretrained")
    FINETUNE = ("finetune", "finetuned")

    @property
    def task_name(self) -> str:
        return self.value[0]

    @property
    def artifact_stage(self) -> str:
        return self.value[1]


class SurrogateStrategy:
    def __init__(self):
        pass

    @abstractmethod
    def is_in_trust_region(self, candidate: Candidate) -> bool:
        pass
    
    @abstractmethod
    def generate_candidates(self, candidate: Candidate,  space: ParameterSpace, finetune: bool = False) -> list[Candidate]:
        pass


class SurrogateBase(Module):
    def __init__(self, strategy: SurrogateStrategy):
        super().__init__()
        self.strategy = strategy

        object.__setattr__(self, "_data_path", None)
        self.model_file_path = Output(
            Workdir.register_file(DefaultPaths.SURROGATE, f"{self._artifact_name}.pt")
        )

        self._lifecycle_tasks: dict[LifecycleStage, Module] = {}
        self._task_stages: dict[Module, LifecycleStage] = {}

        for stage in LifecycleStage:
            if self.implements(stage.task_name):
                task = getattr(self, stage.task_name)
                if isinstance(task, Module):
                    self._lifecycle_tasks[stage] = task
                    self._task_stages[task] = stage

        if (not self._lifecycle_tasks.get(LifecycleStage.TRAIN) and 
            not self._lifecycle_tasks.get(LifecycleStage.PRETRAIN)):
            raise NotImplementedError(
                f"{type(self).__name__} must implement train or pretrain "
                "to be trained through the workflow."
            )

    def implements(self, method_name: str) -> bool:
        for cls in type(self).__mro__:
            if method_name in cls.__dict__ and cls is not SurrogateBase:
                return True
        return False

    @dynamic_submodule
    def train(self, design, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} does not implement train.")

    @dynamic_submodule
    def pretrain(self, design, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} does not implement pretrain.")

    @dynamic_submodule
    def finetune(self, design, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} does not implement finetune.")
    
    def generate_training_candidates(self, candidate: Candidate, training_task: Module):
        finetune = self._lifecycle_tasks.get(LifecycleStage.FINETUNE) is training_task
        return self.strategy.generate_candidates(candidate, candidate.space, finetune=finetune)

    @abstractmethod
    def state_dict(self) -> dict:
        ...

    @abstractmethod
    def from_state_dict(self, state: dict) -> None:
        ...
        
    def save_to_file(self, model_file: str | Path) -> None:
        torch.save(self.state_dict(), model_file)

    def load_from_file(self, model_file: str | Path | BinaryIO) -> None:
        self.from_state_dict(
            torch.load(model_file, weights_only=True, map_location="cpu")
        )

    def publish_trained_model(self, model_file: str, stage: str = "trained") -> ArtifactRef:
        storage = get_current_storage()
        artifact = storage.artifacts.create_artifact(self._artifact_name, "model")
        artifact.add_file(model_file, name="model.pt")
        return storage.artifacts.save_artifact(artifact, stage=stage).ref

    def patch(self, surrogate: SurrogateBase):
        raise RuntimeError(
            f"A surrogate ({type(self).__name__}) cannot be patched with another "
            "surrogate; surrogates cannot contain surrogates."
        )

    def is_in_trust_region(self, candidate: Candidate) -> bool:
        if not self.is_pretrained and not self.is_finetuned:
            return False
                
        return self.strategy.is_in_trust_region(candidate)

    def training_stage(self, task: Module) -> str:
        stage = self._task_stages.get(task)
        if stage is None:
            raise ValueError(f"{task.name} is not a lifecycle task of {type(self).__name__}.")
        return stage.artifact_stage
        
    def select_training_task(self, candidate: Candidate) -> Module | None:
        if self._use_cached_models() and self.is_in_trust_region(candidate):
            return None

        return self._lifecycle_tasks[self._get_next_stage()]

    def get_runtime_model(self):
        ref = self._resolve_model_ref()
        storage = get_current_storage()
        checkpoint = storage.artifacts.load_file(ref, "model.pt")
        self.load_from_file(checkpoint)
        return self.surrogate

    def _resolve_model_ref(self) -> ArtifactRef:
        storage = get_current_storage()
        name = self._artifact_name
        for stage in (LifecycleStage.FINETUNE, LifecycleStage.PRETRAIN, LifecycleStage.TRAIN):
            artifact_stage = stage.artifact_stage
            version = storage.artifacts.latest_version(name, stage=artifact_stage)
            if version is not None:
                return ArtifactRef(name, "model", version, artifact_stage)
        raise RuntimeError(f"No trained model artifact exists for '{name}'.")

    
    def _get_next_stage(self) -> LifecycleStage:
        if self.implements_finetune and self.is_pretrained:
            return LifecycleStage.FINETUNE
        if self.implements_pretrain:
            return LifecycleStage.PRETRAIN
        return LifecycleStage.TRAIN

    def _has_artifact_for(self, *stages: LifecycleStage) -> bool:
            if not (storage := get_current_storage()):
                return False
            
            return any(
                storage.artifacts.latest_version(self._artifact_name, stage=stage.artifact_stage) is not None
                for stage in stages
            )
    
    def _use_cached_models(self) -> bool:
        from dsbo.utils.cache import CachePolicy
        
        cache_config: CacheConfig = dsbo.config.get("cache_config")
        if cache_config is None:
            return True

        if cache_config.cache_policy == CachePolicy.RETRAIN_SURROGATES:
            return False

        result = cache_config.should_cache(GraphMode.DEFAULT)
        return result

    @property
    def implements_train(self) -> bool:
        return LifecycleStage.TRAIN in self._lifecycle_tasks
    
    @property
    def implements_pretrain(self) -> bool:
        return LifecycleStage.PRETRAIN in self._lifecycle_tasks
    
    @property
    def implements_finetune(self) -> bool:
        return LifecycleStage.FINETUNE in self._lifecycle_tasks
    
    @property
    def implements_pretrain_and_finetune(self) -> bool:
        return self.implements_pretrain and self.implements_finetune

    @property
    def is_available(self):
        from dsbo.utils.cache import CachePolicy
        cache_config: CacheConfig | None = dsbo.config.get("cache_config")
        exceptions = (CachePolicy.RETRAIN_SURROGATES, CachePolicy.ALWAYS_EXECUTE)

        if cache_config is None or cache_config.cache_policy not in exceptions:
            return self.is_pretrained or self.is_finetuned
        
        return False

    @property
    def is_pretrained(self) -> bool:
        return self._has_artifact_for(LifecycleStage.PRETRAIN, LifecycleStage.TRAIN)

    @property
    def is_finetuned(self) -> bool:
        return self._has_artifact_for(LifecycleStage.FINETUNE)

    @property
    def _artifact_name(self) -> str:
        return type(self).__name__.lower()
