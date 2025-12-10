from typing import Optional
from typing import Sequence

from dsbo.simulation import SimulationBase
from dsbo.transforms import SimulationTransform
from dsbo.reconstruction import ReconstructionBase

class Workflow:
    def __init__(self, 
                 simulation: SimulationBase, 
                 postprocessing: Optional[SimulationTransform],
                 reconstruction: ReconstructionBase) -> None:
        
        self.simulation = simulation
        self.postprocessing = postprocessing
        self.reconstruction = reconstruction

    def run(self, design: dict, trial_id: Optional[int] = None) -> Sequence[float]:
        self.simulation.simulate(design)

        if self.postprocessing is not None:
            self.postprocessing(self.simulation.simulation_dir)
            
        self.reconstruction.train(self.simulation.simulation_dir, trial_id)
        objective_vals = self.reconstruction.validate(self.simulation.simulation_dir, trial_id)

        assert isinstance(objective_vals, Sequence), "Objective must be a sequence of floats."
        return objective_vals

    def __cleanup__(self) -> None:
        self.simulation.remove_simulation_files()
        self.reconstruction.remove_reconstruction_files()