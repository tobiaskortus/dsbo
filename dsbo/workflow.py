from typing import Optional

from dsbo.simulation import SimulationBase
from dsbo.transforms import SimulationTransform
from dsbo.reconstruction import ReconstructionBase

class Workflow:
    def __init__(self, 
                 simulation: SimulationBase, 
                 postprocessing: Optional[SimulationTransform],
                 reconstruction: ReconstructionBase) -> None:
        
        self.simulation = simulation
        self.posprocessing = postprocessing
        self.reconstruction = reconstruction

    def run(self, design: dict):
        self.simulation.simulate(design)

        if self.posprocessing is not None:
            self.posprocessing(self.simulation.simulation_dir)
            
        self.reconstruction.train(self.simulation.simulation_dir)
        objective = self.reconstruction.validate(self.simulation.simulation_dir)
        #self.__cleanup__()
        return objective

    def __cleanup__(self) -> None:
        self.simulation.remove_simulation_files()
        self.reconstruction.remove_reconstruction_files()