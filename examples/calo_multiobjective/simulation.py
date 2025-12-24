from sqlalchemy import false
from dsbo.simulation import SimulationContainerWrapperBase
from dsbo.transforms import SimulationTransform, load_df_by_extension
from dsbo.minipandas import MiniFrame, concat
from dsbo.container import python_script_args
from typing import Optional, List
import pandas as pd
import numpy as np
import pickle
import json
import os

from dsbo.utils.io import DefaultPaths, Workdir


class G4CaloSimulation(SimulationContainerWrapperBase):
    def __init__(self, container_path: str, container_extra_flags: Optional[str] = ""):
        super().__init__(container_path, container_extra_flags)

    def simulate(self, design: dict) -> None:
        design_path = os.path.join(self.input_dir(), "design.json")
        simulation_path = os.path.join(self.output_dir(), "output_df.npy")

        extra_args = {
            "num_layers": 5,
            "num_events": 200,
            "granularity:0": 25,
            "granularity:1": 25,
            "granularity:2": 25,
            "granularity:3": 25,
            "granularity:4": 25,
            #"N:pi+": {
            #    "min_value":0,
            #    "max_value":1,
            #},
            #"N:proton":{
            #    "min_value":0,
            #   "max_value":1,
            #},
            "N:e+":{
                "min_value":0,
                "max_value":1,
            },
            "N:gamma":{
                "min_value":0,
                "max_value":1,
            },
            "minEnergy_GeV": 5.0,           
            "maxEnergy_GeV": 100.0,
            "sharedEnergy": False,
            "exclusiveSimulation": True,
        }

        design.update(extra_args)

        with open(design_path, "w") as f:
            json.dump(design, f)

        args = python_script_args("examples/calo_multiobjective/container_scripts/simulation.py", [design_path, simulation_path])
        super().call_container(args, True)


class G4CaloSimToReco(SimulationTransform):
    def __init__(self) -> None:
        super().__init__()
        self.input_keys = ['sensor_energy', 'sensor_x', 'sensor_y', 'sensor_z', 'sensor_dx', 'sensor_dy', 'sensor_dz', 'sensor_layer']
        self.target_keys = ['true_energy']
        self.classes_keys = ['contains:e+', 'contains:gamma'] #'contains:pi+', 'contains:proton'

        self.input_dir = Workdir.register(DefaultPaths.SIMULATION)
        self.output_dir = Workdir.register(DefaultPaths.SIMULATION)

    def __call__(self, design: dict, **kwargs) -> None:
        simulation_path = os.path.join(self.input_dir(), "output_df.npy")
        output_path = os.path.join(self.output_dir(), "output_df.pkl")

        df = MiniFrame.read_pickle(simulation_path)

        reco_dict = {
            "Inputs": {key: df[key] for key in self.input_keys},
            "Targets": {key: df[key] for key in self.target_keys},
            "Classes": {key: df[key] for key in self.classes_keys}
        }

        with open(output_path, "wb") as f:
            pickle.dump(reco_dict, f)   