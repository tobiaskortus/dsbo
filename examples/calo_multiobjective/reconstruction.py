
import os
import numpy as np
from typing import Optional
from collections.abc import Sequence

from dsbo.container import python_script_args
from dsbo.reconstruction import ReconstructionContainerWrapper
from dsbo.utils.io import DefaultPaths, Workdir

class G4CaloReconstruction(ReconstructionContainerWrapper):
    def __init__(self, container_path: str, **kwargs):
        super().__init__(container_path=container_path, cuda_enabled=True, **kwargs)
        self.script_path = "examples/calo_multiobjective/container_scripts/train.py"

        self.input_path  = Workdir.register(DefaultPaths.SIMULATION)
        self.output_path = Workdir.register(DefaultPaths.RECONSTRUCTION)

    def train(self, **kwargs) -> None:
        self.output_file = os.path.join(self.input_path(), "output_df.pkl")
        self.reco_path = os.path.join(self.output_path(), "reco_df")
        self.model_path = os.path.join(self.output_path(), "model")
        self.results_dir = self.output_path()

        os.makedirs(self.model_path, exist_ok=True)

        args = [self.output_file, self.reco_path,  str(False), self.model_path, self.results_dir]
        train_args = python_script_args(self.script_path, args=args)
        super().call_container(train_args, True)
    
    def validate(self, **kwargs) -> Sequence[float]:
        return tuple(map(float, np.loadtxt(self.reco_path)))