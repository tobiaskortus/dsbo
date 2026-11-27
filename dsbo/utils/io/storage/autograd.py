import os
import uuid
import shutil
import numpy as np
from typing import ClassVar
from contextlib import suppress

import torch
from tensordict import TensorDict
import torch.autograd.forward_ad as fw
from torch.utils.hooks import RemovableHandle

from dsbo.utils.io import gradient as gradient_io
from dsbo.utils.io.storage.utils import assert_same_batch_sizes, assert_valid_keys, to_dict


def link_or_copy(source: str, destination: str) -> None:
    """Create ``destination`` as a hardlink to ``source``, copying when links are unsupported."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def get_primal_tangent(input: torch.Tensor | TensorDict | dict[str, torch.Tensor] | tuple[np.ndarray | torch.Tensor, ...]) -> tuple[np.ndarray | dict[str, np.ndarray], np.ndarray | dict[str, np.ndarray] | None]:
    if isinstance(input, tuple):
        return input

    if isinstance(input, (TensorDict, dict)):
        primal, tangent = {}, {}
        for key, value in input.items():
            primal_value, tangent_value = get_primal_tangent(value)
            primal[key] = primal_value
            if tangent_value is not None:
                tangent[key] = tangent_value
        return primal, tangent or None

    primal = fw.unpack_dual(input).primal.detach().cpu().numpy()

    if fw._current_level != -1:
        tangent_val = fw.unpack_dual(input).tangent
        tangent = tangent_val.detach().cpu().numpy() if tangent_val is not None else None
        return primal, tangent
    else:
        return primal, None


class AutogradFileInterface:
    writer: ClassVar[type | None] = None
    handles: dict[str, list[RemovableHandle]] = {}
    _gradient_written: set[str] = set()
    _registered_extensions: set[str] = set()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        AutogradFileInterface._registered_extensions.update(cls.reducible_extensions())

    @classmethod
    def reducible_extensions(cls) -> frozenset[str]:
        writer = cls.writer
        return writer.reducible_extensions() if writer is not None else frozenset()

    @classmethod
    def default_extension(cls) -> str:
        extensions = cls.reducible_extensions()
        return min(extensions, key=lambda extension: (len(extension), extension)) if extensions else ""

    @classmethod
    def registered_extensions(cls) -> frozenset[str]:
        return frozenset(AutogradFileInterface._registered_extensions)

    @classmethod
    def reset_gradient_state(cls):
        """Call at the start of each backward pass to clear per-pass gradient tracking."""
        if not gradient_io.is_active():
            cls._gradient_written.clear()
        handles = cls._reader_handles()
        for reader_handles in handles.values():
            for handle in reader_handles:
                handle.remove()
        handles.clear()

    @classmethod
    def _reader_handles(cls):
        return gradient_io.get_reader_hook_handles() if gradient_io.is_active() else cls.handles

    @classmethod
    def _open_dict(cls, fpath: str, x: TensorDict) -> TensorDict:
        if torch.is_grad_enabled():
            gradient_io.claim_gradient_reader(fpath)

            for key, value in x.items():
                if not value.is_floating_point():
                    continue

                value.requires_grad_(True)
                handle_holder: dict[str, RemovableHandle] = {}

                def _post_accumulate_grad_hook(t: torch.Tensor, key: str = key, holder: dict[str, RemovableHandle] = handle_holder):
                    handle = holder["handle"]
                    handles = cls._reader_handles()
                    if handle not in handles.get(fpath, ()):
                        raise RuntimeError(
                            f"Trying to record backward gradient for '{fpath}', but this reader has no "
                            "autograd hook registered in the current scope. The reader was opened "
                            "in a different scope or its hook has already been removed."
                        )
                    with gradient_io.gradient_output(fpath, cls) as path:
                        cls.write_gradient_contribution(
                            path, {key: t.grad.detach().cpu().numpy()}, [key]
                        )
                    handle.remove()
                    handles[fpath].remove(handle)
                    if not handles[fpath]:
                        del handles[fpath]

                handle = value.register_post_accumulate_grad_hook(_post_accumulate_grad_hook)
                handle_holder["handle"] = handle
                cls._reader_handles().setdefault(fpath, []).append(handle)

        return x

    @classmethod
    def open(cls, fpath: str, device: str, columns: list[str] | None = None) -> torch.Tensor | TensorDict:
        if fw._current_level != -1:
            primal, tangent = cls.writer.open_primal_tangent(fpath, columns=columns)
            if isinstance(primal, TensorDict):
                if tangent is not None:
                    primal = TensorDict(
                        {
                            key: fw.make_dual(value, tangent[key]) if key in tangent.keys() else value
                            for key, value in primal.items()
                        },
                        batch_size=primal.batch_size,
                    )
                return primal
            if tangent is not None:
                primal, tangent = torch.from_numpy(primal), torch.from_numpy(tangent)
                return fw.make_dual(primal, tangent)
            else:
                return torch.from_numpy(primal)

        primal = cls.writer.open_primal(fpath, columns=columns)
        if isinstance(primal, TensorDict):
            return cls._open_dict(fpath, primal)

        x = torch.from_numpy(primal)

        if torch.is_grad_enabled():
            gradient_io.claim_gradient_reader(fpath)
            x.requires_grad_(True)
            handle_holder: dict[str, RemovableHandle] = {}

            def _post_accumulate_grad_hook(t: torch.Tensor):
                handle = handle_holder["handle"]
                handles = cls._reader_handles()
                if handle not in handles.get(fpath, ()):
                    raise RuntimeError(
                        f"Trying to record backward gradient for '{fpath}', but this reader has no "
                        "autograd hook registered in the current scope. The reader was opened "
                        "in a different scope or its hook has already been removed."
                    )
                with gradient_io.gradient_output(fpath, cls) as path:
                    cls.write_gradient_contribution(path, t.grad.detach().cpu().numpy(), None)
                handle.remove()
                handles[fpath].remove(handle)
                if not handles[fpath]:
                    del handles[fpath]

            handle = x.register_post_accumulate_grad_hook(_post_accumulate_grad_hook)
            handle_holder["handle"] = handle
            cls._reader_handles().setdefault(fpath, []).append(handle)

        return x

    @classmethod
    def write(
        cls,
        fpath: str,
        input: torch.Tensor | TensorDict | dict[str, torch.Tensor],
    ):
        if isinstance(input, TensorDict):
            input = to_dict(input)
        elif isinstance(input, dict):
            assert_valid_keys(input)
            assert_same_batch_sizes(input)
        primal, tangent = get_primal_tangent(input)
        if tangent is None:
            cls.writer.write_primal(fpath, primal)
        else:
            cls.writer.write_primal_tangent(fpath, primal, tangent)

    @classmethod
    def read_backward_gradient(cls, fpath: str) -> torch.Tensor | TensorDict:
        grad_path = gradient_io.resolve_gradient(fpath)
        gradient = cls.writer.load_backward_gradient(fpath, grad_path)
        if isinstance(gradient, TensorDict):
            return gradient
        return torch.from_numpy(gradient)

    @classmethod
    def write_gradient_contribution(
        cls,
        path: str,
        gradient: np.ndarray | dict[str, np.ndarray] | TensorDict,
        keys: list[str] | None,
    ) -> None:
        if isinstance(gradient, TensorDict):
            if keys is not None:
                gradient = gradient.select(*keys)
        elif isinstance(gradient, dict) and keys is not None:
            gradient = {key: gradient[key] for key in keys}
        cls.writer.write_gradient(path, gradient)

    @classmethod
    def remove(cls, fpath: str) -> None:
        try:
            os.unlink(fpath)
        except FileNotFoundError:
            pass

    @classmethod
    def aggregate_gradients(cls, sources: list[str], destination: str) -> None:
        if len(sources) == 1:
            contribution = sources[0]
            with suppress(OSError):
                if os.path.samefile(destination, contribution):
                    return
            # Replace the directory entry atomically without truncating a linked contribution.
            alias = f"{destination}.{uuid.uuid4().hex}.alias"
            link_or_copy(contribution, alias)
            os.replace(alias, destination)
            return
        cls.writer.aggregate_gradients(sources, destination)
