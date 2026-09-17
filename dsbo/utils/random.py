from __future__ import annotations

import contextvars
import torch
import random
import numpy as np
from typing import TYPE_CHECKING, Sequence, overload

if TYPE_CHECKING:
    from dsbo.modules import Module
    from dsbo.workflow import Workflow



def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    ModuleRandomEngine.seed_random_engine(seed)

def get_random_engine():
    return ModuleRandomEngine.get_current_random_engine()

def seed_random_engine(seed: int | None = None):
    return ModuleRandomEngine.seed_random_engine(seed)


class ModuleRandomEngine:
    _current: contextvars.ContextVar[ModuleRandomEngine | None] = contextvars.ContextVar(
        "module_random_engine", default=None
    )

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
        engine = cls._current.get()
        if engine is None or engine._seed != seed:
            engine = ModuleRandomEngine(seed)
            cls._current.set(engine)

        return engine
    
    @classmethod
    def get_current_random_engine(cls):
        engine = cls._current.get()
        if engine is None:
            return cls.seed_random_engine()
        
        return engine
