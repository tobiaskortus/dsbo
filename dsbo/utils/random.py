from __future__ import annotations

import random
import torch
from typing import TYPE_CHECKING, Sequence, overload

if TYPE_CHECKING:
    from dsbo.modules import Module
    from dsbo.workflow import Workflow

def get_random_engine():
    return ModuleRandomEngine.get_current_random_engine()


def seed_random_engine(seed: int | None = None):
    return ModuleRandomEngine.seed_random_engine(seed)


class ModuleRandomEngine:
    _instance: ModuleRandomEngine = None

    def __init__(self, seed: int | None = None):
        if seed is None: 
            seed = random.SystemRandom().getrandbits(64)

        self._rng = random.Random(seed)
        self._seed = seed

    @overload
    def seed_all(self, workflow: Workflow): ...

    @overload
    def seed_all(self, modules: Sequence[Module]): ...

    def seed_all(self, module_collection: Sequence[Module] | Workflow):
        from dsbo.workflow import Workflow

        # Reset the RNG to the engine's current seed, so calling seed_all on the same
        # collection always produces the same module seeds. If the user set a seed via
        # seed_random_engine(N), that seed is used; otherwise the initial random seed is used.
        self._rng = random.Random(self._seed)

        if isinstance(module_collection, Workflow):
            module_collection = module_collection.task_graph.nodes

        module: Module
        for module in module_collection:
            seed = self._rng.getrandbits(64)
            module.set_seed(seed)


    @classmethod
    def seed_random_engine(cls, seed: int | None = None):
        if cls._instance is None or cls._instance._seed != seed:
            cls._instance = ModuleRandomEngine(seed)

        return cls._instance
    
    @classmethod
    def get_current_random_engine(cls):
        if cls._instance is None:
            return cls.seed_random_engine()
        
        return cls._instance