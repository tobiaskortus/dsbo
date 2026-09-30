
import dask.distributed
import torch
import torch.autograd.forward_ad as fw
from contextlib import nullcontext
from abc import ABC, abstractmethod
from typing import List, Optional, Union
from dask.distributed import Client, LocalCluster

from dsbo.modules import Module
from dsbo.autograd import get_tangent_sampler
from dsbo.container import ContainerWrapperBase
from dsbo.utils.io import Workdir, get_file_suffix, get_subdir, get_workdir, with_remote_workdir


class SimulationBase(Module):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __call__(self, design: dict, **kwargs) -> dict:
        tangent_sampler = get_tangent_sampler()
        if tangent_sampler is not None:
            design = tangent_sampler.sample(design)
        self.simulate(design, **kwargs)

    def sample_count(self, design: dict) -> int:
        raise NotImplementedError(f"{type(self).__name__} does not declare a sample count.")

    def unpack_design_tangents(self, design: dict) -> tuple[dict, dict]:
        sampled = get_tangent_sampler() is not None
        sample_count = self.sample_count(design)
        primals, tangents = {}, {}

        for name, value in design.items():
            if not isinstance(value, torch.Tensor):
                primals[name], tangents[name] = value, None
                continue

            primal, tangent = fw.unpack_dual(value)
            if sampled:
                primal = primal[0]
            elif tangent is not None:
                tangent = tangent.expand(sample_count, *tangent.shape)
            primals[name], tangents[name] = primal, tangent

        return primals, tangents

    @abstractmethod
    def simulate(self, design: dict) -> None:
        pass


class SimulationContainerWrapperBase(SimulationBase, ContainerWrapperBase):
    def __init__(self, container_path: str, container_extra_flags: Optional[str] = ""):
        SimulationBase.__init__(self)
        ContainerWrapperBase.__init__(self, container_path, container_extra_flags)

    def call_container(self, arguments: list, is_exec: bool = False) -> None:
        if is_exec: super().exec_container(arguments)
        else: super().run_container(arguments)
