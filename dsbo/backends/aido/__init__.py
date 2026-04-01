from __future__ import annotations
import os
from shlex import join

import torch
from typing import Optional, Union, Iterable, TYPE_CHECKING

from dsbo.utils.io import Workdir

if TYPE_CHECKING:
    from dsbo.workflow import Workflow
    from dsbo.worker.dask import DaskWorkerBase
    from dsbo.constraints import GeometryConstraintBase
    from dsbo.constraints import ConstraintBase, MultiConstraint

from dsbo.objectives import Objective
from dsbo.backends.aido.strategy import CovarianceStrategy
from dsbo.backends.aido.surrogate import LocalDiffusionSurrogate
from dsbo.parameters import Candidate, ParameterSpace, Continuous


def box_constraint_loss(candidate: Candidate) -> torch.Tensor:
    penalties = []
    for name, p in candidate.space:
        if not isinstance(p, Continuous):
            continue
        offset, _ = candidate._index[name]
        x = candidate.tensor[offset]
        lower = torch.tensor(p.lower, dtype=x.dtype, device=x.device)
        upper = torch.tensor(p.upper, dtype=x.dtype, device=x.device)
        penalties.append(torch.relu(lower - x) ** 2 + torch.relu(x - upper) ** 2)
    return torch.stack(penalties).sum() if penalties else candidate.tensor.new_zeros(1).squeeze()


def constraint_loss(candidate: Candidate, constraint: MultiConstraint | GeometryConstraintBase) -> torch.Tensor:
    constraint_values = constraint(candidate)
    return sum([torch.clip(c, min=0.0) for c in constraint_values])


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
            workdir_root = os.path.join("files", f"aido_run{self.seed}")
            with Workdir.create(os.path.join(workdir_root, f"iteration={iteration}"), keep=True):
                out = self.workflow.run(candidate)

                loss = sum([out[obj.name] * (1 if obj.minimize else -1) for obj in self.objective])
                loss += box_constraint_loss(candidate) * 100.0
                loss += constraint_loss(candidate, self.constraint) * 100 if self.constraint is not None else 0.0

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                #Verify with true objective value
                out = self.workflow.run(candidate, use_surrogate=False)
                losses += [[out[obj.name] for obj in self.objective]]
                coordinates += [candidate.tensor.detach().clone().numpy()]


        return losses, coordinates