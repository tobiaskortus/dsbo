import math

import torch

from dsbo.parameters import Candidate, ParameterSpace, Vector
from dsbo.problem import Problem
from dsbo.workflow import Workflow
from dsbo.modules import batch_fuse, with_no_grad

from examples.hepemshow.modules.simulation import HepEmShowSimulation
from examples.hepemshow.modules.reconstruction import ParticleEnergyRegression

class HepEmShowCalorimeterProblem(Problem):
    NUM_LAYERS = 3
    LOWER_BOUND = 0.0
    UPPER_BOUND = 50.0
    INITIAL_ABSORBER_THICKNESSES = (9.030052185058594, 10.446663856506348, 36.0)
    INITIAL_SCINTILLATOR_THICKNESSES = (37.155208587646484, 29.665525436401367, 27.5)
    TRAINING_EVENTS = 40_000
    EVALUATION_EVENTS = 10_000

    def __init__(self, name, workdir):
        super().__init__(name, workdir)

    @classmethod
    def _training_simulation(cls, num_simulations: int) -> HepEmShowSimulation:
        return HepEmShowSimulation(
            num_events=math.ceil(cls.TRAINING_EVENTS / num_simulations),
            absorber_passive=True,
            file_prefix="training",
        )

    @classmethod
    def _evaluation_simulation(cls, num_simulations: int) -> HepEmShowSimulation:
        return HepEmShowSimulation(
            num_events=math.ceil(cls.EVALUATION_EVENTS / num_simulations),
            absorber_passive=True,
            file_prefix="evaluation",
        )

    def create_batch_reduced_workflow(self):
        training_simulation = self._training_simulation(10)
        evaluation_simulation = self._evaluation_simulation(5)
        reconstruction = ParticleEnergyRegression(training_simulation, evaluation_simulation, epochs=20)
        reconstruction_train = batch_fuse(reconstruction.train)
        return Workflow(with_no_grad(training_simulation * 10 >> reconstruction_train) |
                        evaluation_simulation * 5 >> reconstruction.eval |
                        reconstruction_train >> reconstruction.eval)
        

    def create_default_workflow(self):
        training_simulation = self._training_simulation(10)
        evaluation_simulation = self._evaluation_simulation(5)
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