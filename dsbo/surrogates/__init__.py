from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import dsbo
from dsbo.storage import get_current_storage
from dsbo.utils.io.workdir import DefaultPaths
from dsbo.modules import ModelLifecycleStage, Module, TrainableModule
from dsbo.parameters import ParameterSpace, Candidate
from dsbo.surrogates.graph_transformations import GraphMode

if TYPE_CHECKING:
    from dsbo.utils.cache import CacheConfig


class SurrogateStrategy:
    def __init__(self):
        pass

    def update(self, candidate: Candidate) -> None:
        pass

    @abstractmethod
    def is_in_trust_region(self, candidate: Candidate) -> bool:
        pass

    @abstractmethod
    def generate_candidates(self, candidate: Candidate, space: ParameterSpace, finetune: bool = False) -> list[Candidate]:
        pass


class SurrogateBase(TrainableModule):
    def __init__(self, strategy: SurrogateStrategy):
        super().__init__(model_dir=DefaultPaths.SURROGATE)
        self.strategy = strategy
        object.__setattr__(self, "_data_path", None)

    def generate_training_candidates(self, candidate: Candidate, training_task: Module):
        finetune = self._lifecycle_tasks.get(ModelLifecycleStage.FINETUNE) is training_task
        self.strategy.update(candidate)
        return self.strategy.generate_candidates(candidate, candidate.space, finetune=finetune)

    def patch(self, surrogate: SurrogateBase):
        raise RuntimeError(
            f"A surrogate ({type(self).__name__}) cannot be patched with another "
            "surrogate; surrogates cannot contain surrogates."
        )

    def is_in_trust_region(self, candidate: Candidate) -> bool:
        if not self.is_pretrained and not self.is_finetuned:
            return False
        return self.strategy.is_in_trust_region(candidate)

    def select_training_task(self, candidate: Candidate) -> Module | None:
        if self._use_cached_models() and self.is_in_trust_region(candidate):
            return None
        return super().select_training_task()

    def get_runtime_model(self):
        ref = self._resolve_model_ref()
        storage = get_current_storage()
        checkpoint = storage.artifacts.load_file(ref, "model.pt")
        self.load_from_file(checkpoint)
        return self.surrogate

    def _use_cached_models(self) -> bool:
        from dsbo.utils.cache import CachePolicy

        cache_config: CacheConfig = dsbo.config.get("cache_config")
        if cache_config is None:
            return True
        if cache_config.cache_policy == CachePolicy.RETRAIN_SURROGATES:
            return False
        return cache_config.should_cache(GraphMode.DEFAULT)

    @property
    def is_available(self):
        from dsbo.utils.cache import CachePolicy

        cache_config: CacheConfig | None = dsbo.config.get("cache_config")
        exceptions = (CachePolicy.RETRAIN_SURROGATES, CachePolicy.ALWAYS_EXECUTE)
        if cache_config is None or cache_config.cache_policy not in exceptions:
            return super().is_available
        return False
