
import os
import pandas as pd
from typing import Optional
from collections.abc import Sequence

from dsbo.utils import python_script_args
from dsbo.reconstruction import ReconstructionContainerWrapper

class G4CaloReconstruction(ReconstructionContainerWrapper):
    def __init__(self, workdir, container_path: str, container_extra_flags: Optional[str] = "--nv"):
        super().__init__(workdir= workdir, container_path=container_path, container_extra_flags=container_extra_flags)
        self.script_path = "examples/calo/container_scripts/reconstruction.py"

    def train(self, simulation_dir: str) -> None:
        output_path = os.path.join(simulation_dir, "output_df.parquet")
        reco_path = os.path.join(self._reconstruction_dir, "reco_df")
        model_path = os.path.join(self._reconstruction_dir, "model.pt")
        args = [output_path, reco_path,  str(False), model_path]
        train_args = python_script_args(self.script_path, args=args)
        super().call_container(train_args, True)
    
    def validate(self, simulation_dir: str) -> Sequence[float]:
        #args = args=[simulation_dir, self._reconstruction_dir, str(True)]
        #val_args = python_script_args(self.script_path, args=args)
        #super().call_container(val_args, True)

        df = pd.read_parquet(os.path.join(self._reconstruction_dir, "reco_df"))
        return [df["Loss"].values.mean()]