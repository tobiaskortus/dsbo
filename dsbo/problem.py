from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import cached_property
from typing import Generator, Mapping, Sequence

from dsbo.modules import Module
from dsbo.parameters import Candidate, ParameterSpace
from dsbo.worker.dask import HeterogeneousDaskWorker, WorkerContext
from dsbo.worker.spec import WorkerSpec
from dsbo.workflow import Workflow


class Problem(ABC):
    def __init__(
        self,
        name: str,
        workdir: str
    ) -> None:
        self.name = name
        self.workdir = workdir
        self._worker_specs: dict[type[Module], WorkerSpec] = {}

    @cached_property
    def default_workflow(self) -> Workflow:
        workflow = self.create_default_workflow()
        self._apply_worker_specs(workflow)
        return workflow

    @cached_property
    def batch_reduced_workflow(self) -> Workflow:
        workflow = self.create_batch_reduced_workflow()
        self._apply_worker_specs(workflow)
        return workflow

    @property
    def has_worker_specs(self) -> bool:
        return bool(self._worker_specs)

    def to(self, worker_specs: Mapping[type[Module], WorkerSpec]) -> Problem:
        self._worker_specs = dict(worker_specs)
        self._reset_workflows()
        return self

    def _reset_workflows(self) -> None:
        # Invalidate cached workflows so they are re-created with the new specs.
        self.__dict__.pop("default_workflow", None)
        self.__dict__.pop("batch_reduced_workflow", None)

    def _apply_worker_specs(self, workflow: Workflow) -> None:
        for module in workflow.task_graph.nodes:
            for module_type, spec in self._worker_specs.items():
                if self._matches_module_type(module, module_type):
                    module.to(spec)
                    break

    @staticmethod
    def _matches_module_type(module: Module, module_type: type[Module]) -> bool:
        if isinstance(module, module_type):
            return True
        # Dynamic submodules are not subclasses of their parent module; resolve
        # their spec through the parent they are bound to.
        return isinstance(getattr(module, "_parent", None), module_type)

    @contextmanager
    def execution_context(self, batch_reduced: bool = False) -> Generator[None]:
        if not self.has_worker_specs:
            yield
            return
        workflow = self.batch_reduced_workflow if batch_reduced else self.default_workflow
        with HeterogeneousDaskWorker.from_workflow(workflow) as worker, \
            WorkerContext.use_worker(worker):
            yield

    @abstractmethod
    def design_space(self) -> ParameterSpace:
        """Return the design space of this problem."""
        
    @abstractmethod
    def create_default_workflow(self) -> Workflow:
        """Compose the complete problem-specific workflow."""

    @abstractmethod
    def create_batch_reduced_workflow(self) -> Workflow:
        """Compose the complete problem-specific workflow, where some of the modules are shared across a batch."""

    @abstractmethod
    def initial_guess(self) -> Candidate:
        """Return a fresh starting candidate for one optimization run."""