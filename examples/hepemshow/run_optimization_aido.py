"""Run the AIDO-style surrogate optimization on the HepEmShow example."""

from __future__ import annotations

from datetime import datetime

import torch

from dsbo.config import config
from dsbo.storage.sqlmodel import StudyStorage
from dsbo.surrogates.workflow_surrogate import DefaultWorkflowSurrogate
from dsbo.utils.io import Workdir
from examples.hepemshow.aido.strategy import CovarianceStrategy
from examples.hepemshow.problem import HepEmShowCalorimeterProblem


def evaluate_candidate(evaluation_workflow, candidate):
    with torch.no_grad(), config.use(cache_config=None):
        return evaluation_workflow.run(candidate.detach(), use_surrogates=False)


def trial_objectives(evaluation, surrogate_loss, constraint):
    return {
        "loss": (evaluation["loss"] + constraint).detach(),
        "eval_mse": evaluation["eval_mse"].detach(),
        "eval_r2": evaluation["eval_r2"].detach(),
        "eval_loss": evaluation["loss"].detach(),
        "surrogate_loss": surrogate_loss.detach(),
        "optimization_loss": (surrogate_loss + constraint).detach(),
    }


problem = HepEmShowCalorimeterProblem("aido_hepemshow", None)

workflow = problem.batch_reduced_workflow
evaluation_workflow = problem.default_workflow

strategy = CovarianceStrategy(
    sigma=2.5,
    generate_scaling=1.0,
    trust_region_scale=0.8,
    num_candidates=20,
    seed=0,
)
surrogate = DefaultWorkflowSurrogate(
    strategy=strategy,
    max_epochs=100,
    batch_size=512,
    learning_rate=1e-3,
    patience=8,
    sample_count=512,
    num_workers=0,
)
workflow.patch(surrogate)


candidate = problem.initial_guess(requires_grad=True)
with torch.no_grad():
    candidate.tensor.clamp_(min=problem.LOWER_BOUND, max=problem.UPPER_BOUND)

optimizer = torch.optim.Adam([candidate.tensor], lr=0.2)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
study_path = f"example_hepemshow_aido_{timestamp}.db"

with  StudyStorage.create(study_path) as study:
    for step in range(8000):
        with Workdir.create(f"files/example_hepemshow_aido/step={step}", error_if_exists=False):
            optimizer.zero_grad()
            result = workflow.run(candidate)

            constraint = 0.1 * torch.clamp(candidate.tensor.sum() - 150.0, min=0.0)
            surrogate_loss = result["loss"].mean()
            loss = surrogate_loss + constraint

            if step % 10 == 0:
                evaluation = evaluate_candidate(evaluation_workflow, candidate)
                objectives = trial_objectives(evaluation, surrogate_loss, constraint)

                study.trials.add_trial(
                    step,
                    candidate.detach().tensor,
                    objectives,
                    constraints={"total_thickness": constraint.detach()},
                )

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                candidate.tensor.clamp_(min=problem.LOWER_BOUND, max=problem.UPPER_BOUND)
