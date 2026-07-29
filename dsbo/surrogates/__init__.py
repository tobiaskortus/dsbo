import torch
import logging
from abc import abstractmethod
from typing import Any, Mapping, TYPE_CHECKING

import dsbo
from dsbo.modules import Module
from dsbo.storage import get_current_storage
from dsbo.parameters import ParameterSpace, Candidate, CandidateBatch

if TYPE_CHECKING:
    from dsbo.utils.cache import CacheConfig

def implements(self, method_name: str) -> bool:
    for cls in type(self).__mro__:
        if method_name in cls.__dict__ and cls is not SurrogateBase:
            return True
    return False


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
    def __init__(self, surrogate: Any, strategy: SurrogateStrategy):
        super().__init__()
        self.surrogate = surrogate
        self.strategy = strategy

    def load_from_registry(self, stage: str | None = None) -> Mapping[str, torch.Tensor]:
        storage = get_current_storage()
        name = type(self).__name__.lower()
        artifact = storage.artifacts.load_artifact(name, stage=stage)
        state_dict = artifact.get_model("model")
        return state_dict

    def write_to_registry(self, surrogate: Any, stage: str | None = None) -> None:
        storage = get_current_storage()
        name = type(self).__name__.lower()
        artifact = storage.artifacts.create_artifact(name, "model")
        artifact.add_model(surrogate, "model")
        storage.artifacts.save_artifact(artifact, stage=stage)

    @property
    def implements_train(self) -> bool:
        return implements(self, 'train')
    
    @property
    def implements_pretrain(self) -> bool:
        return implements(self, 'pretrain')
    
    @property
    def implements_finetune(self) -> bool:
        return implements(self, 'finetune')
    
    @property
    def implements_pretrain_and_finetune(self) -> bool:
        return self.implements_pretrain and self.implements_finetune

    @property
    def _is_pretrained(self) -> bool:
        storage = get_current_storage()
        if storage is None:
            return False
        name = type(self).__name__.lower()
        # Accept any of the trained stages — "pretrained", "trained", "finetuned"
        # Models saved via the single-phase train() path use stage="trained",
        # which is functionally equivalent to being pretrained. However usually this shouldnt happen
        # since train and pretrain/finetune require individual implementaitons.
        return (storage.artifacts.latest_version(name, stage="pretrained") is not None or
                storage.artifacts.latest_version(name, stage="trained") is not None)

    @property
    def _is_finetuned(self) -> bool:
        # NOTE: _is_finetuned only returns if there exists any finetuned version of the model,
        # the surrogate strategy decides if this is still usable
        storage = get_current_storage()
        if storage is None:
            return False
        name = type(self).__name__.lower()
        return (storage.artifacts.latest_version(name, stage="finetuned") is not None or
                storage.artifacts.latest_version(name, stage="trained") is not None)

    def train(self, candidate: Candidate, workflow_callback: Any):
        pass

    def pretrain(self, candidate: Candidate, workflow_callback: Any):
        pass

    def finetune(self, candidate: Candidate, workflow_callback: Any):
        pass

    def fit(self, candidate: Candidate, workflow_callback: Any):
        use_cache = self._use_cached_models()
        name = type(self).__name__.lower()

        if self.implements_train:
            if use_cache:
                try:
                    state_dict = self.load_from_registry()
                    self.load_state_dict(state_dict)
                    return
                except Exception as exc:
                    pass

            self.surrogate = self.train(candidate, workflow_callback)
            self.write_to_registry(self.surrogate, stage="pretrained")
    

        elif self.implements_pretrain_and_finetune:
            if use_cache:
                if self._is_pretrained:
                    state_dict = self.load_from_registry()
                    self.surrogate.load_state_dict(state_dict)
                    self.surrogate = self.finetune(candidate, workflow_callback)
                    self.write_to_registry(self.surrogate, stage="finetuned")
                else:
                    self.surrogate = self.pretrain(candidate, workflow_callback)
                    self.write_to_registry(self.surrogate, stage="pretrained")
            else:
                self.surrogate = self.pretrain(candidate, workflow_callback)
                self.surrogate = self.finetune(candidate, workflow_callback)
        else:
            raise NotImplementedError("Surrogate must implement either 'train' or both 'pretrain' and 'finetune' methods.")

    def is_in_trust_region(self, candidate: Candidate) -> bool:
        pretrained = self._is_pretrained
        finetuned  = self._is_finetuned
        if not pretrained and not finetuned:
            return False

        result = self.strategy.is_in_trust_region(candidate)
        return result

    def _use_cached_models(self) -> bool:
        from dsbo.utils.cache import CachePolicy
        from dsbo.surrogates.graph_transformations import GraphMode

        cache_config: CacheConfig = dsbo.config.get("cache_config")
        if cache_config is None:
            return False

        if cache_config.cache_policy == CachePolicy.RETRAIN_SURROGATES:
            return False

        result = cache_config.should_cache(GraphMode.DEFAULT)
        return result
    
    @property
    def is_available(self):
        from dsbo.utils.cache import CachePolicy
        cache_config: CacheConfig = dsbo.config.get("cache_config")
        exceptions = (CachePolicy.RETRAIN_SURROGATES, CachePolicy.ALWAYS_EXECUTE)
        if cache_config.cache_policy in exceptions:
            return False
        
        return self._is_pretrained or self._is_finetuned
    