import os
import json
from typing import Optional, List

from dsbo.simulation import SimulationContainerWrapperBase
from dsbo.utils import python_script_args

class G4CaloSimulation(SimulationContainerWrapperBase):
    def __init__(self, workdir: str, container_path: str, container_extra_flags: Optional[str] = ""):
        super().__init__(workdir, container_path, container_extra_flags)

    def simulate(self, design: dict):
        design_path = os.path.join(self._workdir, "design.json")
        simulation_path = os.path.join(self.simulation_dir, "output_df")

        with open(design_path, "w") as f:
            json.dump(design, f)

        args = python_script_args("examples/calo/container_scripts/simulation.py", [design_path, simulation_path])
        super().call_container(args, True)
        