import torch
import functools
from typing import Any
from typing import Callable
from contextlib import nullcontext
from dataclasses import dataclass
import torch.autograd.forward_ad as fw


@dataclass(frozen=True)
class SerializedTensor:
    primal: torch.Tensor
    tangent: torch.Tensor | None = None
    grad: torch.Tensor | None = None


def _pack(m: dict[str, Any]) -> dict[str, Any]:
    return {k: (SerializedTensor(*fw.unpack_dual(v)) if isinstance(v, torch.Tensor) else v) for k, v in m.items()}


def _unpack(m: dict[str, Any]) -> dict[str, Any]:
    return {k: (fw.make_dual(v.primal, v.tangent) if isinstance(v, SerializedTensor) and v.tangent is not None and fw._current_level != -1 else v.primal if isinstance(v, SerializedTensor) else v) for k, v in m.items()}


def train_op(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        level = fw._current_level
        for _ in range(level + 1):
            fw.exit_dual_level()
        try:
            with torch.enable_grad():
                return func(*args, **kwargs)
        finally:
            for _ in range(level + 1):
                fw.enter_dual_level()
    return wrapper


def autograd_context(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        is_forward_ad   = kwargs.pop("is_forward_ad", False)
        is_grad_enabled = kwargs.pop("is_grad_enabled", False)
        pack_result     = kwargs.pop("pack_result", False)
        unpack_design   = kwargs.pop("unpack_design", False)

        current_level = fw._current_level
        use_existing_forward_ad_level = is_forward_ad and current_level > -1

        assert current_level == -1 or use_existing_forward_ad_level, (
            "Forward AD level should be -1 when entering autograd_context, but got level {}"
            .format(current_level)
        )

        with torch.set_grad_enabled(is_grad_enabled):
            with fw.dual_level() if is_forward_ad and not use_existing_forward_ad_level else nullcontext():
                if unpack_design and "design" in kwargs:
                    kwargs["design"] = _unpack(kwargs["design"])
                result = func(*args, **kwargs)
                return _pack(result) if pack_result and isinstance(result, dict) else result

    return wrapper