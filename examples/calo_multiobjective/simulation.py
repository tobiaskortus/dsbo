from enum import unique
from dsbo.modules.simulation import SimulationContainerWrapperBase
from dsbo.transforms import SimulationTransform, load_df_by_extension
from dsbo.minipandas import MiniFrame, concat
from dsbo.container import python_script_args
from typing import Optional
from filelock import FileLock
import pandas as pd
import numpy as np
import pickle
import json
import os

from dsbo.utils.io import DefaultPaths, Workdir

#TODO: move this into dsbo to make it reusable
def to_serializable(obj):
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def write_design_threadsafe(design: dict, design_path: str) -> None:
    lock_path = design_path + ".lock"
    s_design = {k: to_serializable(v) for k, v in design.items()}
    with FileLock(lock_path):
        with open(design_path, "w") as f:
            json.dump(s_design, f)


class G4CaloSimulation(SimulationContainerWrapperBase):
    def __init__(self, container_path: str, container_extra_flags: Optional[str] = ""):
        super().__init__(container_path, container_extra_flags)
        self.design_path = Workdir.register_file(DefaultPaths.SIMULATION, "design.json", unique_filename=False)
        self.simulation_path = Workdir.register_file(DefaultPaths.SIMULATION, "output_df.npy", unique_filename=True)
        self.script_path = "examples/calo_multiobjective/container_scripts/simulation.py"

    def simulate(self, design: dict) -> None:
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
        write_design_threadsafe(design, self.design_path())

        args = python_script_args(self.script_path, [self.design_path(), self.simulation_path()])
        super().call_container(args, True)


class G4CaloSimToReco(SimulationTransform):
    def __init__(self) -> None:
        super().__init__()
        self.input_keys = ['sensor_energy', 'sensor_x', 'sensor_y', 'sensor_z', 'sensor_dx', 'sensor_dy', 'sensor_dz', 'sensor_layer']
        self.target_keys = ['true_energy']
        self.classes_keys = ['contains:e+', 'contains:gamma'] #'contains:pi+', 'contains:proton'

        self.simulation_path = Workdir.register_file(DefaultPaths.SIMULATION, "output_df.npy", unique_filename=False)
        self.output_path = Workdir.register_file(DefaultPaths.SIMULATION, "output_df.pkl", unique_filename=False)

    def transform(self, design: dict, **kwargs) -> None:
        df = MiniFrame.read_pickle(self.simulation_path())

        reco_dict = {
            "Inputs": {key: df[key] for key in self.input_keys},
            "Targets": {key: df[key] for key in self.target_keys},
            "Classes": {key: df[key] for key in self.classes_keys}
        }

        with open(self.output_path(), "wb") as f:
            pickle.dump(reco_dict, f)   