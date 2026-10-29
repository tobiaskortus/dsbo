import numpy as np
import torch
from datetime import datetime

from dsbo.utils.io import Workdir
from dsbo.worker.spec import SLURMWorkerSpec
from dsbo.storage.sqlmodel import StudyStorage
from dsbo.worker.dask import HeterogeneousDaskWorker, WorkerContext

from examples.hepemshow.problem import HepEmShowCalorimeterProblem
from examples.hepemshow.modules.simulation import HepEmShowSimulation
from examples.hepemshow.modules.reconstruction import ParticleEnergyRegression

spec = SLURMWorkerSpec(10, 1, "16GB", "1:00:00", partition="epyc-256").adapt(0, 5)

problem = HepEmShowCalorimeterProblem("calorimeter_problem", None)
problem.to({HepEmShowSimulation: spec, ParticleEnergyRegression: spec})

workflow = problem.default_workflow
candidate = problem.initial_guess(requires_grad=True)

optimizer = torch.optim.RMSprop([candidate.tensor], lr=1)
study_path = f"example_hepemshow_{datetime.now():%Y%m%d-%H%M%S-%f}.db"

with StudyStorage.create(study_path) as study, HeterogeneousDaskWorker.from_workflow(workflow) as worker:
    with WorkerContext.use_worker(worker):
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
