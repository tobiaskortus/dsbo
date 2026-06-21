import torch
from abc import abstractmethod
from typing import Any, Mapping

from dsbo.modules import Module
from dsbo.storage import get_current_storage
from dsbo.parameters import ParameterSpace, Candidate, CandidateBatch

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

        self._is_pretrained = False
        self._is_finetuned = False

    def load_from_registry(self) -> Mapping[str, torch.Tensor]:
        # Load latest surrogate model from registry based on the candidate
        # TODO: implement option for alternative strategies (e.g. first, best, most similar, etc.)
        return get_current_storage().artifacts.load_from_model_registry(type(self).__name__.lower(), step=None)
        
    def write_to_registry(self, surrogate: Any) -> None:
       get_current_storage().artifacts.write_to_model_registry(surrogate, type(self).__name__.lower(), step=0)

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

    def train(self, candidate: Candidate, workflow_callback: Any):
        pass

    def pretrain(self, candidate: Candidate, workflow_callback: Any):
        pass

    def finetune(self, candidate: Candidate, workflow_callback: Any):
        pass

    def fit(self, candidate: Candidate, workflow_callback: Any):
        if self.implements_train:
            #Surrogate model is retrained from scratch in every iteration
            self.surrogate = self.train(candidate, workflow_callback)
            self._is_pretrained = True

        elif self.implements_pretrain_and_finetune:
            # A base surrogate model is pretrained once and then finetuned in every iteration

            if not self._is_pretrained:
                self.surrogate = self.pretrain(candidate, workflow_callback)
                self.write_to_registry(self.surrogate)
                self._is_pretrained = True
                requires_finetune = False
            else:
                self.surrogate.load_state_dict(self.load_from_registry())
                self._is_pretrained = True
                requires_finetune = True

            if requires_finetune:
                self.surrogate = self.finetune(candidate, workflow_callback)
                self.surrogate = self.write_to_registry(self.surrogate)
                self._is_finetuned = True
        else:
            raise NotImplementedError("Surrogate must implement either 'train' or both 'pretrain' and 'finetune' methods.")


    def is_in_trust_region(self, candidate: Candidate) -> bool:
        if not self._is_pretrained and not self._is_finetuned:
            # If the surrogate has not been trained at all, 
            # all candidates are outside the trust region
            return False
        
        return self.strategy.is_in_trust_region(candidate)