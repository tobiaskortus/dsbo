import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyhepemshow
import torch
import torch.autograd.forward_ad as fw

from dsbo.config import get_active_file_backend
from dsbo.modules.simulation import SimulationBase
from dsbo.utils.io import DefaultPaths, Output, Workdir


class HepEmShowSimulation(SimulationBase):
    simulation_file: Output
    energy_file: Output
    labels_file: Output

    def __init__(
        self,
        num_events: int = 100,
        num_sims_per_particle: int = 1,
        particle_energy_min: float = 1_000,
        particle_energy_max: float = 10_000,
        particles: tuple[str, ...] = ("e-",),
        num_workers: int | None = None,
        verbose: bool = False,
        data_file: str | Path | None = None,
        absorber_passive: bool = True,
        file_prefix: str = "simulation",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.simulation_file = Output(
            Workdir.register_file(
                DefaultPaths.SIMULATION,
                f"{file_prefix}.h5",
                unique_filename=True,
            )
        )
        self.energy_file = Output(
            Workdir.register_file(
                DefaultPaths.SIMULATION,
                f"{file_prefix}_energies.h5",
                unique_filename=True,
            )
        )
        self.labels_file = Output(
            Workdir.register_file(
                DefaultPaths.SIMULATION,
                f"{file_prefix}_labels.h5",
                unique_filename=True,
            )
        )
        self.num_events = num_events
        self.num_sims_per_particle = num_sims_per_particle
        self.particle_energy_min = particle_energy_min
        self.particle_energy_max = particle_energy_max
        self.particles = particles
        self.num_workers = num_workers
        self.verbose = verbose
        self.absorber_passive = absorber_passive
        self.data_file = (
            Path(data_file)
            if data_file is not None
            else Path(__file__).resolve().parents[1] / "data" / "hepem_data_fe_pbwo4.json"
        )
        self.set_seed(random.randrange(2**31))

    @property
    def _events_per_job(self) -> int:
        return self.num_events * self.num_sims_per_particle * len(self.particles)

    def _build_tasks(self, absorber, gap, absorber_dot=None, gap_dot=None, grad_output=None):
        absorber_shared = absorber[0] if absorber.ndim == 2 else absorber
        gap_shared = gap[0] if gap.ndim == 2 else gap
        common_arguments = {
            "data_file": str(self.data_file),
            "absorber_thickness": absorber_shared.detach().cpu().tolist(),
            "gap_thickness": gap_shared.detach().cpu().tolist(),
            "absorber_passive": self.absorber_passive,
            "verbosity": int(self.verbose),
        }
        if absorber_dot is not None:
            common_arguments["absorber_thickness_dot"] = absorber_dot.detach().cpu().tolist()
        if gap_dot is not None:
            common_arguments["gap_thickness_dot"] = gap_dot.detach().cpu().tolist()

        rng = random.Random(self._seed + (self.i or 0) * 1_000_003)
        tasks = []
        for particle in self.particles:
            for _ in range(self.num_sims_per_particle):
                task = {
                    **common_arguments,
                    "particle": particle,
                    "particle_energy_min": self.particle_energy_min,
                    "particle_energy_max": self.particle_energy_max,
                    "num_events": self.num_events,
                    "random_seed": rng.randrange(2**31),
                }
                if grad_output is not None:
                    start = len(tasks) * self.num_events
                    task["bar_edep"] = grad_output[start : start + self.num_events]
                tasks.append(task)
        return tasks

    @staticmethod
    def _split_task(task: dict[str, Any], chunks: int) -> list[dict[str, Any]]:
        event_count = task["num_events"]
        chunks = min(chunks, event_count)
        base, remainder = divmod(event_count, chunks)
        seed_generator = random.Random(task["random_seed"])
        bar_edep = task.get("bar_edep")
        result = []
        start = 0

        for index in range(chunks):
            count = base + (index < remainder)
            stop = start + count
            chunk = dict(task)
            chunk["num_events"] = count
            chunk["random_seed"] = seed_generator.randrange(2**31)
            if bar_edep is not None:
                chunk["bar_edep"] = bar_edep[start:stop]
            result.append(chunk)
            start = stop

        return result

    def _parallel_tasks(self, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.num_workers is None or self.num_workers <= len(tasks):
            return tasks

        remaining_workers = self.num_workers
        parallel_tasks = []
        for index, task in enumerate(tasks):
            remaining_tasks = len(tasks) - index
            chunks = max(1, remaining_workers // remaining_tasks)
            split = self._split_task(task, chunks)
            parallel_tasks.extend(split)
            remaining_workers -= len(split)
        return parallel_tasks

    def _run_tasks(self, module, tasks):
        def run(task):
            if task.get("bar_edep") is not None:
                task = dict(task)
                task["bar_edep"] = np.asarray(task["bar_edep"]).reshape(-1).tolist()
            return module.run_simulation(**task)

        tasks = self._parallel_tasks(tasks)
        num_workers = min(self.num_workers or len(tasks), len(tasks))
        if num_workers == 1:
            return [run(task) for task in tasks]
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            return list(executor.map(run, tasks))

    @staticmethod
    def _primal_tangent(value):
        if isinstance(value, torch.Tensor):
            return fw.unpack_dual(value)
        return value, None

    @staticmethod
    def _backpropagate_native_gradients(design, results):
        inputs, gradients = [], []

        def add_active(tensor, gradient):
            if tensor.requires_grad:
                inputs.append(tensor)
                gradients.append(torch.as_tensor(gradient, device=tensor.device, dtype=tensor.dtype).expand_as(tensor))

        add_active(design["absorber"], np.sum([result["bar_thickness_absorber"] for result in results], axis=0))
        add_active(design["gap"], np.sum([result["bar_thickness_gap"] for result in results], axis=0))

        if inputs:
            torch.autograd.backward(inputs, gradients)

    def _run_simulation(
        self,
        design: dict[str, torch.Tensor],
        grad_output: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]] | None:
        absorber, absorber_dot = self._primal_tangent(design["absorber"])
        gap, gap_dot = self._primal_tangent(design["gap"])
        if self._no_grad_op:
            absorber_dot = gap_dot = None

        use_forward_ad = absorber_dot is not None or gap_dot is not None
        mode = "forward_ad" if use_forward_ad else "forward"
        if grad_output is not None:
            mode = "reverse"

        module = getattr(pyhepemshow, mode)
        tasks = self._build_tasks(absorber, gap, absorber_dot, gap_dot, grad_output)
        results = self._run_tasks(module, tasks)

        if grad_output is not None:
            self._backpropagate_native_gradients(design, results)
            return None

        tensor_kwargs = dict(device=absorber.device, dtype=absorber.dtype)
        primal = torch.as_tensor(np.concatenate([result["edep_per_cell"] for result in results]), **tensor_kwargs)
        if not use_forward_ad:
            return primal, results

        tangent = torch.as_tensor(
            np.concatenate([result["edep_per_cell_dot"] for result in results]),
            **tensor_kwargs,
        )
        return fw.make_dual(primal, tangent), results

    def _particle_type_labels(self) -> torch.Tensor:
        events_per_particle = self.num_events * self.num_sims_per_particle
        labels = torch.repeat_interleave(
            torch.arange(len(self.particles), dtype=torch.float32),
            events_per_particle,
        )
        return labels[: self._events_per_job]

    def simulate(self, design: dict[str, torch.Tensor]) -> None:
        result = self._run_simulation(design)
        if result is None:
            raise RuntimeError("Forward simulation returned no result.")

        edep, results = result
        energies = torch.as_tensor(
            np.concatenate([result["particle_energies"] for result in results]),
            device=edep.device,
            dtype=edep.dtype,
        )
        storage = get_active_file_backend()
        storage.write(self.simulation_file(), edep)
        storage.write(self.energy_file(), energies)
        storage.write(self.labels_file(), self._particle_type_labels())

    def backward(self, design: dict[str, torch.Tensor], grad_output: torch.Tensor | None = None) -> None:
        storage = get_active_file_backend()
        try:
            bar_edep = storage.read_backward_gradient(self.simulation_file()).numpy()
        except FileNotFoundError:
            return
        self._run_simulation(design, grad_output=bar_edep)
    