import math
from typing import Any

from dsbo.modules import batch_fuse, with_no_grad
from dsbo.parameters import Candidate, ParameterSpace, Vector
from dsbo.problem import Problem
from dsbo.workflow import Workflow
from examples.hepemshow.modules.reconstruction import ParticleEnergyRegression
from examples.hepemshow.modules.simulation import HepEmShowSimulation
from examples.hepemshow.modules.simulation_container import HepEmShowContainerSimulation


class HepEmShowCalorimeterProblem(Problem):
    NUM_LAYERS = 3
    LOWER_BOUND = 0.0
    UPPER_BOUND = 50.0
    INITIAL_ABSORBER_THICKNESSES = (9.030052185058594, 10.446663856506348, 36.0)
    INITIAL_SCINTILLATOR_THICKNESSES = (37.155208587646484, 29.665525436401367, 27.5)

    CONTAINER_IMAGE = HepEmShowContainerSimulation.CONTAINER_IMAGE

    
    def __init__(self, name: str, workdir: str | None, backend: str = "python") -> None:
        self.simulation_cls, self.simulation_kwargs = self.simulation_config(backend)
        super().__init__(name, workdir)

    def _training_simulation(self, num_simulations: int, num_events: int):
        return self.simulation_cls(
            num_events=math.ceil(num_events / num_simulations),
            absorber_passive=True,
            file_prefix="training",
            **self.simulation_kwargs,
        )

    def _evaluation_simulation(self, num_simulations: int, num_events: int):
        return self.simulation_cls(
            num_events=math.ceil(num_events / num_simulations),
            absorber_passive=True,
            file_prefix="evaluation",
            **self.simulation_kwargs,
        )

    def simulation_config(cls, backend: str) -> tuple[type, dict[str, Any]]:
        if backend == "python":
            return HepEmShowSimulation, {}
        if backend == "container":
            return HepEmShowContainerSimulation, {"image": cls.CONTAINER_IMAGE}
        raise ValueError(f"Unknown simulation backend: {backend!r}")

    def create_batch_reduced_workflow(self):
        training_simulation = self._training_simulation(10, 400)
        evaluation_simulation = self._evaluation_simulation(5, 400)
        reconstruction = ParticleEnergyRegression(training_simulation, evaluation_simulation)
        reconstruction_train = batch_fuse(reconstruction.train)
        return Workflow(with_no_grad(training_simulation * 10 >> reconstruction_train) |
                        evaluation_simulation * 5 >> reconstruction.eval |
                        reconstruction_train >> reconstruction.eval)

    def create_default_workflow(self):
        training_simulation = self._training_simulation(10, 40_000)
        evaluation_simulation = self._evaluation_simulation(5, 10_000)
        reconstruction = ParticleEnergyRegression(training_simulation, evaluation_simulation)
        return Workflow(with_no_grad(training_simulation * 10 >> reconstruction.train) |
                        evaluation_simulation * 5 >> reconstruction.eval |
                        reconstruction.train >> reconstruction.eval)

    def design_space(self):
        return ParameterSpace(
            absorber=Vector(self.LOWER_BOUND, self.UPPER_BOUND, size=self.NUM_LAYERS),
            gap=Vector(self.LOWER_BOUND, self.UPPER_BOUND, size=self.NUM_LAYERS),
        )
        
    def initial_guess(self, requires_grad: bool = False):
        return Candidate.from_values(
            self.design_space(),
            {
                "absorber": list(self.INITIAL_ABSORBER_THICKNESSES),
                "gap": list(self.INITIAL_SCINTILLATOR_THICKNESSES),
            },
            requires_grad=requires_grad,
        )
