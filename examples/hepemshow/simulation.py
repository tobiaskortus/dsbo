import pyhepemshow
from hepemshow import get_hepemshow_data_file

import torch
import numpy as np
from pathlib import Path
import multiprocessing as mp
import torch.autograd.forward_ad as fw

from dsbo.utils.io.autograd import NumpyFile
from dsbo.modules.simulation import SimulationBase
from dsbo.utils.io import Workdir, DefaultPaths, Output


def hepem_worker_fn(args):
    data_file, abs_thick, gap_thick, abs_dot, gap_dot, particle, num_events = args
    r = pyhepemshow.run_simulation(
        data_file,
        absorber_thickness=abs_thick,
        gap_thickness=gap_thick,
        absorber_thickness_dot=abs_dot,
        gap_thickness_dot=gap_dot,
        particle=particle,
        particle_energy_dot=1.0,
        num_events=num_events,
    )
    return r["edep_per_layer"], r["edep_per_layer_dot"]


class HepEmShowSimulation(SimulationBase):
    def __init__(self, 
                 num_events: int = 100, 
                 num_sims_per_particle: int = 10, 
                 min_energy: float = 1000,
                 max_energy: float = 100_000,
                 **kwargs):
        
        super().__init__(**kwargs)
        self.num_events = num_events
        self.num_sims_per_particle = num_sims_per_particle

        self.min_energy = min_energy
        self.max_energy = max_energy

        self.simulation_file = Output(Workdir.register_file(DefaultPaths.SIMULATION, "simulation.npz", unique_filename=True))

    def simulate(self, design):
        def _merge(results: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
            edep_per_layer = np.concatenate([r[0] for r in results], axis=0)
            edep_per_layer_dot = np.concatenate([r[1] for r in results], axis=0)
            return edep_per_layer, edep_per_layer_dot

        hepem_data = get_hepemshow_data_file()
        
        absorbers = design["absober"].numpy()
        gaps = design["gap"].numpy()

        absorbers_dot = fw.unpack_dual(design["absober"]).tangent.numpy()
        gaps_dot = fw.unpack_dual(design["gap"]).tangent.numpy()

        # Arguments for simulating positrons
        task_positron = (
            hepem_data,
            absorbers.tolist(),
            gaps.tolist(),
            absorbers_dot.tolist(),
            gaps_dot.tolist(),
            "e+",
            self.num_events,
        )

        # Arguments for simulating gammas
        task_gamma = (
            hepem_data,
            absorbers.tolist(),
            gaps.tolist(),
            absorbers_dot.tolist(),
            gaps_dot.tolist(),
            "gamma",
            self.num_events,
        )

        tasks = [task_positron] * self.num_sims_per_particle + [task_gamma] * self.num_sims_per_particle

        with mp.get_context("fork").Pool(min(mp.cpu_count(), self.num_sims_per_particle * 2)) as pool:
            results = pool.map(hepem_worker_fn, tasks)

        indices = {"e+": 0, "gamma": 1}
        target = torch.tensor([indices[task[5]] for task in tasks], dtype=torch.float32)
        target = target.repeat_interleave(self.num_events)
        target = fw.make_dual(target, torch.zeros_like(target))

        edep_per_layer, edep_per_layer_dot = _merge(results)
        edep_per_layer = fw.make_dual(
            torch.from_numpy(edep_per_layer).to(design["absober"].device),
            torch.from_numpy(edep_per_layer_dot).to(design["absober"].device),
        )

        NumpyFile.write(self.simulation_file(), edep_per_layer)