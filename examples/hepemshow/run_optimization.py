import argparse
from datetime import datetime

import torch

from dsbo.storage.sqlmodel import StudyStorage
from dsbo.utils.io import Workdir
from dsbo.worker.dask import HeterogeneousDaskWorker, WorkerContext
from dsbo.worker.spec import SLURMWorkerSpec
from examples.hepemshow.modules.reconstruction import ParticleEnergyRegression
from examples.hepemshow.problem import HepEmShowCalorimeterProblem


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct HepEmShow calorimeter optimization.")
    parser.add_argument(
        "--simulation",
        choices=("python", "container"),
        default="python",
        help="Simulation backend: the Python extension or the native container.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    spec = SLURMWorkerSpec(10, 1, "16GB", "1:00:00", partition="epyc-256").adapt(0, 5)

    problem = HepEmShowCalorimeterProblem("calorimeter_problem", None, backend=args.simulation)
    problem.to({problem.simulation_cls: spec, ParticleEnergyRegression: spec})

    workflow = problem.default_workflow
    candidate = problem.initial_guess(requires_grad=True)

    optimizer = torch.optim.RMSprop([candidate.tensor], lr=1)
    study_path = f"example_hepemshow_{args.simulation}_{datetime.now():%Y%m%d-%H%M%S-%f}.db"

    with StudyStorage.create(study_path) as study, \
         HeterogeneousDaskWorker.from_workflow(workflow) as worker, \
         WorkerContext.use_worker(worker):
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


if __name__ == "__main__":
    main()
