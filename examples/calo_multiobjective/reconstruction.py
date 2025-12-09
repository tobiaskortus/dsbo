
import os
import pandas as pd
from typing import Optional
from collections.abc import Sequence

from dsbo.utils import python_script_args
from dsbo.reconstruction import ReconstructionContainerWrapper

class G4CaloReconstruction(ReconstructionContainerWrapper):
    def __init__(self, workdir, container_path: str):
        super().__init__(workdir= workdir, container_path=container_path, cuda_enabled=True)
        self.script_path = "examples/calo_multiobjective/container_scripts/train.py"

    def train(self, simulation_dir: str) -> None:
        output_path = os.path.join(simulation_dir, "output_df.pkl")
        reco_path = os.path.join(self._reconstruction_dir, "reco_df")
        model_path = os.path.join(self._reconstruction_dir, "model")

        os.makedirs(model_path, exist_ok=True)

        args = [output_path, reco_path,  str(False), model_path]
        train_args = python_script_args(self.script_path, args=args)
        super().call_container(train_args, True)
    
    def validate(self, simulation_dir: str) -> Sequence[float]:
        #args = args=[simulation_dir, self._reconstruction_dir, str(True)]
        #val_args = python_script_args(self.script_path, args=args)
        #super().call_container(val_args, True)

        df = pd.read_parquet(os.path.join(self._reconstruction_dir, "reco_df"))
        return [df["Loss"].values.mean()]