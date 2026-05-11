import os
import torch
import torch.autograd.forward_ad as fw
import numpy as np
from typing import overload
from abc import ABC, abstractmethod


class AutogradFileInterface:
    @classmethod
    def open(cls, fpath: str, device: str) -> torch.Tensor:
        if fw._current_level != -1:
            primal, tangent = cls.open_primal_tangent(fpath)
            if tangent is not None:
                primal, tangent = torch.from_numpy(primal), torch.from_numpy(tangent)
                return fw.make_dual(primal, tangent)
            else:
                return torch.from_numpy(primal)
        else:
            primal =  cls.open_primal(fpath)
            x = torch.from_numpy(primal)

            if torch.is_grad_enabled():
                x.requires_grad_(True)
                x.register_post_accumulate_grad_hook(
                    lambda grad: cls.write_backward_gradient(grad.detach().cpu().numpy(), fpath)
                )

            return x
            


    @classmethod
    @abstractmethod
    def open_primal(cls, fpath) -> np.ndarray:
        pass

    @classmethod
    @abstractmethod
    def open_primal_tangent(cls, fpath: str) -> tuple[np.ndarray, ...]:
        pass

    
    @overload
    @classmethod
    def write(self, 
              fpath: str, 
              primal: np.ndarray | torch.Tensor, 
              tangent: np.ndarray | torch.Tensor | None):
        ...

    @overload
    @classmethod
    def write(self, fpath: str, tensor: torch.Tensor):
        ...

    @classmethod
    def write(cls, fpath: str, input: torch.Tensor | tuple[np.ndarray | torch.Tensor, ...]):
        primal, tangent = cls._get_primal_tangent(input)

        if tangent is None:
            cls.write_primal(fpath, primal)
        else:
            cls.write_primal_tangent(fpath, primal, tangent)

    @classmethod
    @abstractmethod
    def write_primal(cls, primal: np.array):
        pass

    @classmethod
    @abstractmethod
    def write_primal_tangent(cls, primal: np.array, tangent: np.array):
        pass

    @classmethod
    @abstractmethod
    def write_backward_gradient(cls, gradient: np.array, fpath: str):
        base, stem = os.path.splitext(fpath)
        base += "_gradient"
        np.savez_compressed(f"{base}{stem}", gradient=gradient)

        
    @classmethod
    def _get_primal_tangent(cls, input: torch.Tensor | tuple[np.ndarray | torch.Tensor, ...]) -> tuple[np.ndarray, ...]:
        if isinstance(input, tuple):
            return input

        primal = fw.unpack_dual(input).primal.detach().cpu().numpy()

        if fw._current_level != -1:
            tangent = fw.unpack_dual(input).tangent.detach().cpu().numpy()
            return primal, tangent
        else:
            return primal, None


class NumpyFile(AutogradFileInterface):    
    @classmethod
    def open_primal(cls, fpath):
        return np.load(fpath)["primal"]
    
    @classmethod
    def open_primal_tangent(cls, fpath):
        x = np.load(fpath)
        keys = set(x.keys())

        assert "primal" in keys, "Saved array must contain at least a 'primal' key"

        if "tangent" in keys:
            return x["primal"], x["tangent"]
        else:
            return x["primal"], None

    @classmethod
    def write_primal(cls, fpath: str, primal):
        np.savez_compressed(fpath, primal=primal)
    
    @classmethod
    def write_primal_tangent(cls, fpath: str, primal, tangent):
        np.savez_compressed(fpath, primal=primal, tangent=tangent)
    