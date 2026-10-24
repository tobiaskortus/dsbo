import os
import shutil
import uuid
from contextlib import suppress
from typing import ClassVar

import numpy as np
import torch
import torch.autograd.forward_ad as fw
from torch.utils.hooks import RemovableHandle

from dsbo.utils.io import gradient as gradient_io


def link_or_copy(source: str, destination: str) -> None:
    """Create ``destination`` as a hardlink to ``source``, copying when links are unsupported."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def get_primal_tangent(input: torch.Tensor | tuple[np.ndarray | torch.Tensor, ...]) -> tuple[np.ndarray, np.ndarray | None]:
    if isinstance(input, tuple):
        return input

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
    def open(cls, fpath: str, device: str, columns: list[str] | None = None) -> torch.Tensor:
        if fw._current_level != -1:
            primal, tangent = cls.writer.open_primal_tangent(fpath, columns=columns)
            if tangent is not None:
                primal, tangent = torch.from_numpy(primal), torch.from_numpy(tangent)
                return fw.make_dual(primal, tangent)
            else:
                return torch.from_numpy(primal)
        else:
            primal = cls.writer.open_primal(fpath, columns=columns)
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
                        cls.write_gradient_contribution(path, fpath, t.grad.detach().cpu().numpy(), columns)
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
        input: torch.Tensor | tuple[np.ndarray | torch.Tensor, ...],
        columns: list[str] | None = None,
    ):
        primal, tangent = get_primal_tangent(input)
        if tangent is None:
            cls.writer.write_primal(fpath, primal, columns=columns)
        else:
            cls.writer.write_primal_tangent(fpath, primal, tangent, columns=columns)

    @classmethod
    def read_backward_gradient(cls, fpath: str) -> torch.Tensor:
        grad_path = gradient_io.resolve_gradient(fpath)
        return torch.from_numpy(cls.writer.load_backward_gradient(fpath, grad_path)).float()

    @classmethod
    def write_gradient_contribution(
        cls,
        path: str,
        fpath: str,
        gradient: np.ndarray,
        columns: list[str] | None,
    ) -> None:
        if columns is None:
            columns = cls.writer.columns(fpath) or None
        cls.writer.write_gradient(path, gradient, columns=columns)

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

