from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.autograd.forward_ad as fw

from dsbo.config import get_active_file_backend
from dsbo.container import ContainerDefinition, DefinitionFile
from dsbo.modules.simulation import SimulationContainerWrapperBase
from dsbo.utils.io import DefaultPaths, Input, Output, Workdir
from dsbo.utils.io.storage import HDF5File
from examples.hepemshow.modules.utils import (
    backpropagate_design_gradients,
    resolve_num_workers,
    seed_for_replica,
)

DEFAULT_DATA_FILE = "/opt/hepemshow/data/hepem_data_fe_pbwo4.json"


class HepEmShowContainerSimulation(SimulationContainerWrapperBase):
    supports_forward_ad = False

    CONTAINER_DIR = Path(__file__).resolve().parents[1] / "container"
    REPO_ROOT = CONTAINER_DIR.parents[2]
    CONTAINER_IMAGE = (
        DefinitionFile(
            CONTAINER_DIR / "Dockerfile",
            container_name="hepemshow-native:latest",
            build_flags=(
                "--build-context", f"dsbo-include={REPO_ROOT / 'cpp' / 'include'}",
                "--build-context", f"dsbo-licenses={REPO_ROOT / 'LICENSES'}",
                "--build-context", f"hepemshow-data={REPO_ROOT / 'examples' / 'hepemshow' / 'data'}",
            ),
        ),
        DefinitionFile(CONTAINER_DIR / "singularity.def"),
    )

    simulation_file: Output
    energy_file: Output
    labels_file: Output
    design_file: Output

    def __init__(
        self,
        image: ContainerDefinition | None = None,
        num_events: int = 100,
        particle_energy_min: float = 1_000,
        particle_energy_max: float = 10_000,
        particles: tuple[str, ...] = ("e-",),
        num_sims_per_particle: int = 1,
        absorber_passive: bool = True,
        file_prefix: str = "simulation",
        verbose: bool = False,
        container_extra_flags: str = "",
        num_workers: int | None = None,
    ) -> None:
        super().__init__(self.CONTAINER_IMAGE if image is None else image, container_extra_flags)
        self.inp = Input(
            Workdir.register_file(
                DefaultPaths.SIMULATION,
                f"{file_prefix}_in.h5",
                unique_filename=True,
            )
        )
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
        self.design_file = Output(
            Workdir.register_file(
                DefaultPaths.SIMULATION,
                f"{file_prefix}_design.h5",
                unique_filename=True,
            )
        )
        self.num_events = num_events
        self.num_sims_per_particle = num_sims_per_particle
        self.particle_energy_min = particle_energy_min
        self.particle_energy_max = particle_energy_max
        self.particles = tuple(particles)
        self.absorber_passive = absorber_passive
        self.verbose = verbose
        self.num_workers = num_workers

        self.set_seed(random.randrange(2**31))

    def _bind_workspace(self) -> None:
        workspace = os.path.abspath(Workdir.get_current_dir())
        if (workspace, workspace) not in self._config.bind_mounts:
            self.bind_mount((workspace, workspace))
            self._runtime = None
        if self._config.workdir != workspace:
            self.set_workdir(workspace)
            self._runtime = None

    @staticmethod
    def _absolute(path: str | None) -> str:
        if path is None:
            raise RuntimeError("Cannot resolve a file path during a dry run.")
        return os.path.abspath(path)

    def _common_args(self) -> list[str]:
        arguments = [
            "--data-file", DEFAULT_DATA_FILE,
            "--num-events", str(self.num_events),
            "--num-sims-per-particle", str(self.num_sims_per_particle),
            "--num-threads", str(resolve_num_workers(self.num_workers, self.resources)),
            "--seed", str(seed_for_replica(self._seed, self.i)),
            "--energy-min", str(self.particle_energy_min),
            "--energy-max", str(self.particle_energy_max),
            "--verbosity", str(int(self.verbose)),
        ]
        for particle in self.particles:
            arguments.extend(("--particle", particle))
        if self.absorber_passive:
            arguments.append("--absorber-passive")
        return arguments

    @staticmethod
    def _reject_forward_ad(design: dict[str, torch.Tensor]) -> None:
        for key in ("absorber", "gap"):
            value = design.get(key)
            if isinstance(value, torch.Tensor) and fw.unpack_dual(value).tangent is not None:
                raise NotImplementedError(
                    "HepEmShowContainerSimulation supports reverse-mode differentiation only; "
                    "forward-mode tangents are not implemented."
                )

    def simulate(self, design: dict[str, torch.Tensor]) -> None:
        self._reject_forward_ad(design)
        self._bind_workspace()
        input_path = self._absolute(self.inp())
        HDF5File.write(
            input_path,
            {
                "absorber": design["absorber"].detach().cpu(),
                "gap": design["gap"].detach().cpu(),
            },
        )
        self.call_container([
            "forward",
            "--input", input_path,
            "--edep-output", self._absolute(self.simulation_file()),
            "--energy-output", self._absolute(self.energy_file()),
            "--labels-output", self._absolute(self.labels_file()),
            *self._common_args(),
        ])

        design_vector = torch.cat(
            [
                design["absorber"].detach().cpu().reshape(-1),
                design["gap"].detach().cpu().reshape(-1),
            ]
        )
        total_events = self.num_events * self.num_sims_per_particle * len(self.particles)
        get_active_file_backend().write(
            self._absolute(self.design_file()),
            design_vector.unsqueeze(0).expand(total_events, -1),
        )

    def _incoming_gradient(self) -> str | None:
        try:
            path = self.simulation_file.gradient()
        except FileNotFoundError:
            return None
        return path if os.path.exists(path) else None

    def backward(
        self,
        design: dict[str, torch.Tensor],
        grad_output: torch.Tensor | dict[str, Any] | None = None,
    ) -> None:
        incoming_gradient = self._incoming_gradient()
        if incoming_gradient is None:
            return
        self._bind_workspace()
        input_adjoint = self._absolute(self.inp.gradient())
        self.call_container([
            "backward",
            "--input", self._absolute(self.inp()),
            "--edep-output", self._absolute(self.simulation_file()),
            "--edep-adjoint", self._absolute(incoming_gradient),
            "--input-adjoint", input_adjoint,
            *self._common_args(),
        ])
        backpropagate_design_gradients(design, HDF5File.load_gradient(input_adjoint))
