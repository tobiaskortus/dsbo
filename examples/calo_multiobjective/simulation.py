import os
import json
import numpy as np
from filelock import FileLock

from dsbo.config import get_active_file_backend
from dsbo.utils.process import python_script_args
from dsbo.utils.io import DefaultPaths, Input, Output, Workdir
from dsbo.modules.simulation import SimulationContainerWrapperBase
from dsbo.modules import Module
from dsbo.container import ImageRef


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


SIMULATION_IMAGE = "jkiesele/minicalosim:90baa21"

class G4CaloSimulation(SimulationContainerWrapperBase):
    def __init__(self, num_events: int = 200):
        super().__init__(ImageRef(SIMULATION_IMAGE))

        self.design_path = Workdir.register_file(DefaultPaths.SIMULATION, "design.json", unique_filename=False)
        self.simulation_path = Output(Workdir.register_file(DefaultPaths.SIMULATION, "simulation.npy", unique_filename=True))

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

        self.bind_mount((".", "."))
        self.set_workdir(".")
        super().call_container(args, True)


class G4CaloSimToImg(Module):
    def __init__(self, w_target: int = 25, h_target: int = 25, log: bool = True):
        super().__init__()
        self.w_target = w_target
        self.h_target = h_target
        self.log = log

        self.simulation_path = Input(Workdir.register_file(DefaultPaths.SIMULATION, "simulation.npy", unique_filename=True))
        self.image_path = Output(Workdir.register_file(DefaultPaths.SIMULATION, "simulation_img.h5", unique_filename=True))
        self.conditional_path = Output(Workdir.register_file(DefaultPaths.SIMULATION, "conditional.h5", unique_filename=True))
        self.target_path = Output(Workdir.register_file(DefaultPaths.SIMULATION, "target.h5", unique_filename=True))

    def _encode_design(self, design: dict) -> np.ndarray:
        fields = []
        for key, value in design.items():
            if isinstance(value, (int, float)):
                fields.append(float(value))
            elif isinstance(value, dict):
                for k, v in sorted(value.items()):
                    if isinstance(v, (int, float)):
                        fields.append(float(v))
        return np.array(fields, dtype=np.float32)

    def __call__(self, design: dict, **kwargs):
        import torch
        from torch.nn import functional as F

        Storage = get_active_file_backend()

        data = np.load(self.simulation_path(), allow_pickle=True)
        l = torch.tensor(data['sensor_layer']).int()
        E = torch.tensor(data['sensor_energy']).float()
        y = torch.tensor(data["contains:e+"]).long()

        _, c = l.unique(return_counts=True, dim=-1)
        w, h = self.w_target, self.h_target

        def reshape_interpolate(E_l, w_target, h_target):
            w = h = int(E_l.shape[-1] ** 0.5)
            E_l = E_l.reshape(-1, 1, w, h)
            E_l = F.interpolate(E_l, size=(w_target, h_target), mode='bilinear')
            return E_l

        y = F.one_hot(y, 2)
        E = torch.cat([reshape_interpolate(E_l, w, h) for E_l in E.split(c.tolist(), dim=-1)], dim=1)

        if self.log:
            E = torch.log1p(E)

        img_np = E.numpy()
        target_np = y.numpy()

        design_encoded = self._encode_design(design)
        design_tiled = np.tile(design_encoded, (img_np.shape[0], 1))
        conditional_np = np.concatenate([design_tiled, target_np], axis=1)

        Storage.write_primal(self.image_path(), img_np)
        Storage.write_primal(self.conditional_path(), conditional_np)
        Storage.write_primal(self.target_path(), target_np)
