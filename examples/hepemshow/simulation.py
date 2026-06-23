import pyhepemshow
from pyhepemshow import get_hepemshow_data_file

import os
import torch
import random
import numpy as np
from pathlib import Path
import multiprocessing as mp
import torch.autograd.forward_ad as fw

from dsbo.config import get_active_file_backend
from dsbo.modules.simulation import SimulationBase
from dsbo.utils.io import Workdir, DefaultPaths, Output
from dsbo.surrogates.module_surrogate import ModuleSurrogateContract, LazyShape


from concurrent.futures import ThreadPoolExecutor


def hepem_worker_fn(kwargs):
    abs_dot = kwargs.pop("absorber_thickness_dot", None)
    gap_dot = kwargs.pop("gap_thickness_dot", None)
    bar_edep = kwargs.pop("bar_edep", None)

    if bar_edep is not None: #Reverse-mode AD call
        r = pyhepemshow.reverse.run_simulation(bar_edep=bar_edep, **kwargs)
        #return a dict that is compatible with the provided design
        return {
            "absorber": np.asarray(r["bar_thickness_absorber"], dtype=float),
            "gap": np.asarray(r["bar_thickness_gap"], dtype=float),
        }
    
    if fw._current_level > -1: #Forward-mode AD call
        r = pyhepemshow.forward_ad.run_simulation(**kwargs, absorber_thickness_dot=abs_dot, gap_thickness_dot=gap_dot)
    else: #Default call without propagating gradient information
        r = pyhepemshow.forward.run_simulation(**kwargs)

    edep_per_layer = r["edep_per_layer"]
    edep_per_layer_dot = r["edep_per_layer_dot"] if fw._current_level > -1 else None
    particle_energies = r["particle_energies"]

    return edep_per_layer, edep_per_layer_dot, particle_energies


class HepEmShowSimulation(SimulationBase):
    def __init__(
        self,
        num_events: int = 100,
        num_sims_per_particle: int = 10,
        particle_energy_min: float = 1000,
        particle_energy_max: float = 50_000,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_events = num_events
        self.num_sims_per_particle = num_sims_per_particle

        self.verbose = verbose
        self.particle_energy_min = particle_energy_min
        self.particle_energy_max = particle_energy_max

        self.target_file = Output(Workdir.register_file(DefaultPaths.SIMULATION, "target.npz", unique_filename=True))
        self.simulation_file = Output(Workdir.register_file(DefaultPaths.SIMULATION, "simulation.npz", unique_filename=True))
        self._contract: ModuleSurrogateContract | None = None

    def _generate_tasks(self, absorbers, gaps, absorbers_dot=None, gaps_dot=None, grad_outputs=None):
        hepem_data = get_hepemshow_data_file()
        is_forward_ad = fw._current_level > -1 and absorbers_dot is not None and gaps_dot is not None
        is_reverse_ad = grad_outputs is not None
        n = self.num_events

        rng = random.Random(self._seed if self._seed is not None else random.randint(0, 2**32 - 1))

        common = {
            "data_file": hepem_data,
            "absorber_thickness": absorbers.tolist(),
            "gap_thickness": gaps.tolist(),
            "num_events": self.num_events,
            "particle_energy_min": self.particle_energy_min,
            "particle_energy_max": self.particle_energy_max,
            "absorber_thickness_dot": absorbers_dot.tolist() if is_forward_ad else None,
            "gap_thickness_dot": gaps_dot.tolist() if is_forward_ad else None,
            "verbosity": 0 if not self.verbose else 1,
        }

        tasks = []
        for i in range(self.num_sims_per_particle):
            bar_edep = (grad_outputs[i * n:(i + 1) * n] * n).reshape(-1).tolist() if is_reverse_ad else None
            task = {**common, "particle": "e+", "random_seed": rng.randint(0, 2**32 - 1), 
                    **({"bar_edep": bar_edep} if is_reverse_ad else {})}
            tasks.append(task)

        offset = self.num_sims_per_particle * n
        for i in range(self.num_sims_per_particle):
            bar_edep = (grad_outputs[offset + i * n:offset + (i + 1) * n] * n).reshape(-1).tolist() if is_reverse_ad else None
            task = {**common, "particle": "gamma", "random_seed": rng.randint(0, 2**32 - 1), 
                    **({"bar_edep": bar_edep} if is_reverse_ad else {})}
            tasks.append(task)

        return tasks

    def _simulate_impl(self, design, grad_output: np.ndarray | None = None):
        absorbers = np.clip(design["absorber"].detach().numpy(), 1e-3, None)
        gaps      = np.clip(design["gap"].detach().numpy(), 1e-3, None)

        if grad_output is not None:
            tasks = self._generate_tasks(absorbers, gaps, grad_outputs=grad_output)

            n_workers = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else mp.cpu_count()
            n_workers = min(n_workers - 1, self.num_sims_per_particle * 2)
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(hepem_worker_fn, tasks))

            device = design["absorber"].device
            grad_abs = torch.from_numpy(sum(r["absorber"] for r in results)).to(device)
            grad_gap = torch.from_numpy(sum(r["gap"] for r in results)).to(device)
            torch.autograd.backward(
                [design["absorber"], design["gap"]],
                [grad_abs, grad_gap],
            )
            return

        def _merge(results):
            edep_per_layer = np.concatenate([r[0] for r in results], axis=0)
            edep_per_layer_dot = (
                np.concatenate([r[1] for r in results], axis=0)
                if results[0][1] is not None else None
            )
            particle_energies = np.concatenate([r[2] for r in results], axis=0)
            return edep_per_layer, edep_per_layer_dot, particle_energies

        if fw._current_level > -1:
            # TODO: check if both absorber and gap are dual.
            # Might not to be the case if one is defined as a constant.
            absorbers_dot = fw.unpack_dual(design["absorber"]).tangent.numpy()
            gaps_dot = fw.unpack_dual(design["gap"]).tangent.numpy()
        else:
            absorbers_dot = None
            gaps_dot = None

        tasks = self._generate_tasks(absorbers, gaps, absorbers_dot, gaps_dot)
        
        n_workers = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else mp.cpu_count()
        n_workers = min(n_workers - 1, self.num_sims_per_particle * 2)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(hepem_worker_fn, tasks))

        indices = {"e+": 0, "gamma": 1}
        target = torch.tensor([indices[task["particle"]] for task in tasks], dtype=torch.float32)
        target = target.repeat_interleave(self.num_events)

        edep_per_layer, edep_per_layer_dot, particle_energies = _merge(results)
        device = design["absorber"].device

        if fw._current_level > -1:
            target = fw.make_dual(target, torch.zeros_like(target))
            edep_tensor = fw.make_dual(
                torch.from_numpy(edep_per_layer).to(device),
                torch.from_numpy(edep_per_layer_dot).to(device),
            )
        else:
            edep_tensor = torch.from_numpy(edep_per_layer).to(device)

        particle_energies = torch.from_numpy(particle_energies).to(device)
        combined_targets = torch.stack([target, particle_energies], dim=1)
        return edep_tensor, combined_targets

    def simulate(self, design):
        edep_per_layer, combined_targets = self._simulate_impl(design)

        absorbers = torch.as_tensor(design["absorber"]).float()
        gaps = torch.as_tensor(design["gap"]).float()
        n = combined_targets.shape[0]
        design_tiled = torch.cat([
            absorbers.unsqueeze(0).expand(n, -1),
            gaps.unsqueeze(0).expand(n, -1),
        ], dim=-1)
        conditioning = torch.cat([combined_targets, design_tiled], dim=-1)

        Storage = get_active_file_backend()
        Storage.write(self.target_file(), conditioning)
        Storage.write(self.simulation_file(), edep_per_layer)

    def backward(self, design, grad_output: torch.Tensor = None):
        Storage = get_active_file_backend()
        bar_edep = Storage.read_backward_gradient(self.simulation_file()).numpy()
        self._simulate_impl(design, grad_output=bar_edep)

    @property
    def surrogate_contract(self) -> ModuleSurrogateContract:
        if self._contract is not None:
            return self._contract

        def output_conditional_fn(design):
            absorbers = np.clip(design["absorber"].detach().numpy(), 1e-3, None)
            gaps = np.clip(design["gap"].detach().numpy(), 1e-3, None)
            indices = {"e+": 0, "gamma": 1}
            tasks = self._generate_tasks(absorbers, gaps)
            target = torch.tensor([indices[t["particle"]] for t in tasks], dtype=torch.float32)
            target = target.repeat_interleave(self.num_events)
            particle_energies = torch.FloatTensor(target.shape).uniform_(
                self.particle_energy_min, self.particle_energy_max
            )

            design_tiled = torch.cat([
                torch.as_tensor(absorbers).float().unsqueeze(0).expand(len(target), -1),
                torch.as_tensor(gaps).float().unsqueeze(0).expand(len(target), -1),
            ], dim=-1)

            return torch.cat([
                torch.stack([target, particle_energies], dim=1),
                design_tiled,                                   
            ], dim=-1)

        self._contract = ModuleSurrogateContract(
            output_dim=LazyShape(self.simulation_file.path),
            conditional_dim=LazyShape(self.target_file.path),
            output_samples=self.num_events * self.num_sims_per_particle * 2,
            output_conditional_fn=output_conditional_fn,
            data_file=self.simulation_file,
            conditional_file=self.target_file,
        )
        return self._contract
