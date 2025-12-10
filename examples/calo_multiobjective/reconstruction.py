
import os
import numpy as np
from typing import Optional
from collections.abc import Sequence

from dsbo.utils import python_script_args
from dsbo.reconstruction import ReconstructionContainerWrapper

class G4CaloReconstruction(ReconstructionContainerWrapper):
    def __init__(self, workdir, container_path: str):
        super().__init__(workdir= workdir, container_path=container_path, cuda_enabled=True)
        self.script_path = "examples/calo_multiobjective/container_scripts/train.py"

    def train(self, simulation_dir: str, trial_id: Optional[int] = None) -> None:
        self.output_path = os.path.join(simulation_dir, "output_df.pkl")
        self.reco_path = os.path.join(self.reconstruction_dir, "reco_df")
        self.model_path = os.path.join(self.reconstruction_dir, "model")
        self.results_dir = self.reconstruction_dir

        os.makedirs(self.model_path, exist_ok=True)

        args = [self.output_path, self.reco_path,  str(False), self.model_path, self.results_dir]
        train_args = python_script_args(self.script_path, args=args)
        super().call_container(train_args, True)
    
    def validate(self, simulation_dir: str, trial_id: Optional[int] = None) -> Sequence[float]:
        return tuple(map(float, np.loadtxt(self.reco_path)))