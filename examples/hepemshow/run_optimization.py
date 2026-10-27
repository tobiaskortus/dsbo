import numpy as np
import torch
from datetime import datetime

from dsbo.utils.io import Workdir
from dsbo.storage.sqlmodel import StudyStorage
from examples.hepemshow.problem import HepEmShowCalorimeterProblem

problem = HepEmShowCalorimeterProblem("calorimeter_problem", None)

workflow = problem.default_workflow
candidate = problem.initial_guess(requires_grad=True)

optimizer = torch.optim.RMSprop([candidate.tensor], lr=1)
study_path = f"example_hepemshow_{datetime.now():%Y%m%d-%H%M%S-%f}.db"

with StudyStorage.create(study_path) as study:
    for step in range(100):
        with Workdir.create(f"files/example_hepemshow/step={step}", error_if_exists=False):
            optimizer.zero_grad()
            result = workflow.run(candidate)
            constraint = 0.1 * torch.clamp(candidate.tensor.sum() - 150.0, min=0.0)
            loss = result["loss"] + constraint

            study.trials.add_trial(
                step,
                candidate.detach().tensor,
                {
                    "loss": loss.detach(),
                    "eval_mse": result["eval_mse"].detach(),
                    "eval_r2": result["eval_r2"].detach(),
                },
                constraints={"total_thickness": constraint.detach()},
            )

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                candidate.tensor.clamp_(min=problem.LOWER_BOUND, max=problem.UPPER_BOUND)
