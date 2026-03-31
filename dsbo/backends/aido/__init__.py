from __future__ import annotations

import torch
from typing import Optional, Union, Iterable, TYPE_CHECKING

from dsbo.utils.io import Workdir

if TYPE_CHECKING:
    from dsbo.workflow import Workflow
    from dsbo.worker.dask import DaskWorkerBase
    from dsbo.constraints import GeometryConstraintBase


from dsbo.objectives import Objective
from dsbo.backends.aido.strategy import CovarianceStrategy
from dsbo.backends.aido.surrogate import LocalDiffusionSurrogate
from dsbo.parameters import Candidate, ParameterSpace, Continuous


def box_constraint_loss(candidate: Candidate) -> torch.Tensor:
    #TODO implement soft box constraint loss
    return 0


class AidoBackend:
    def __init__(self, study_name: str, workflow: Workflow, design: ParameterSpace, 
                 constraint: Optional[GeometryConstraintBase], objective: Union[Objective, Iterable[Objective]],
                 worker: Optional[DaskWorkerBase] = None,
                 seed: int = 42) -> None:
        
        if isinstance(objective, Objective):
            objective = [objective]

        assert all([isinstance(p, Continuous) for _, p in design]), "Currently only continuous parameters are supported!"
        assert len(objective) == 1, "Currently only single-objective optimization is supported!"

        self.seed = seed
        self.design = design
        self.worker = worker
        self.workflow = workflow
        self.objective = objective
        self.constraint = constraint
        self.study_name = study_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        from dsbo.objectives import check_workflow_validity
        check_workflow_validity(self.workflow, self.objective)


    def optimize(self, n_iterations: int, initial_candidate: Candidate) -> tuple[list, list]:
        assert initial_candidate.tensor.requires_grad, "Initial candidate tensor must be generated with requires_grad=True!"

        candidate = initial_candidate

        cov = 0.05 * torch.diag(torch.ones_like(candidate.tensor))
        surrogate = LocalDiffusionSurrogate(CovarianceStrategy(cov))
        optimizer = torch.optim.AdamW([candidate.tensor], lr=0.1)

        self.workflow.patch(surrogate)

        losses = []
        coordinates = []

        for iteration in range(n_iterations):
            with Workdir.create(f"files/iteration={iteration}"):
                out = self.workflow.run(candidate)
                loss = sum([out[obj.name] * (1 if obj.minimize else -1) for obj in self.objective])
                loss += box_constraint_loss(candidate) * 100.0
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                #Verify with true objective value
                out = self.workflow.run(candidate, use_surrogate=False)
                losses += [[out[obj.name] for obj in self.objective]]
                coordinates += [candidate.tensor.detach().clone().numpy()]


        return losses, coordinates