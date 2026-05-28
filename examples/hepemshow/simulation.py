import pyhepemshow
from pyhepemshow import get_hepemshow_data_file

import torch
import numpy as np
from pathlib import Path
import multiprocessing as mp
import torch.autograd.forward_ad as fw

from dsbo.utils.io.autograd import NumpyFile
from dsbo.modules.simulation import SimulationBase
from dsbo.utils.io import Workdir, DefaultPaths, Output


def hepem_worker_fn(args):
    (data_file, abs_thick, gap_thick, abs_dot, 
     gap_dot, particle, num_events,
     particle_energy_min, particle_energy_max) = args
    
    r = pyhepemshow.run_simulation(
        data_file,
        absorber_thickness=abs_thick,
        gap_thickness=gap_thick,
        absorber_thickness_dot=abs_dot,
        gap_thickness_dot=gap_dot,
        particle=particle,
        particle_energy_min=particle_energy_min,
        particle_energy_max=particle_energy_max,
        num_events=num_events
    )
    return r["edep_per_layer"], r["edep_per_layer_dot"], r["particle_energies"]


class HepEmShowSimulation(SimulationBase):
    def __init__(self, 
                 num_events: int = 100, 
                 num_sims_per_particle: int = 10, 
                 particle_energy_min: float = 1000,
                 particle_energy_max: float = 50_000,
                 **kwargs):
        
        super().__init__(**kwargs)
        self.num_events = num_events
        self.num_sims_per_particle = num_sims_per_particle

        self.particle_energy_min = particle_energy_min
        self.particle_energy_max = particle_energy_max

        self.target_file = Output(Workdir.register_file(DefaultPaths.SIMULATION, "target.npz", unique_filename=True))
        self.simulation_file = Output(Workdir.register_file(DefaultPaths.SIMULATION, "simulation.npz", unique_filename=True))

    def simulate(self, design):
        def _merge(results: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            edep_per_layer = np.concatenate([r[0] for r in results], axis=0)
            edep_per_layer_dot = np.concatenate([r[1] for r in results], axis=0)
            particle_energies = np.concatenate([r[2] for r in results], axis=0)
            return edep_per_layer, edep_per_layer_dot, particle_energies

        hepem_data = get_hepemshow_data_file()

        absorbers = design["absorber"].numpy()
        gaps = design["gap"].numpy()

        absorbers_dot = fw.unpack_dual(design["absorber"]).tangent.numpy()
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
            self.particle_energy_min,
            self.particle_energy_max,
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
            self.particle_energy_min,
            self.particle_energy_max,
        )

        tasks = [task_positron] * self.num_sims_per_particle + [task_gamma] * self.num_sims_per_particle

        with mp.get_context("fork").Pool(min(mp.cpu_count(), self.num_sims_per_particle * 2)) as pool:
            results = pool.map(hepem_worker_fn, tasks)

        indices = {"e+": 0, "gamma": 1}
        target = torch.tensor([indices[task[5]] for task in tasks], dtype=torch.float32)
        target = target.repeat_interleave(self.num_events)
        target = fw.make_dual(target, torch.zeros_like(target))

        edep_per_layer, edep_per_layer_dot, particle_energies = _merge(results)
        edep_per_layer = fw.make_dual(
            torch.from_numpy(edep_per_layer).to(design["absorber"].device),
            torch.from_numpy(edep_per_layer_dot).to(design["absorber"].device),
        )

        particle_energies = torch.from_numpy(particle_energies).to(design["absorber"].device)
        combined_targets = torch.stack([target, particle_energies], dim=1)

        NumpyFile.write(self.target_file(), combined_targets)
        NumpyFile.write(self.simulation_file(), edep_per_layer)