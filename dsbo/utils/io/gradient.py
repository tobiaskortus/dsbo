from __future__ import annotations

import contextvars
import os
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle

from dsbo.utils.io.reduce import (
    GradientAggregationResolver,
    ResolvableSource,
    make_reduction_resolver,
    reduction_sources,
    resolve_value,
)
from dsbo.utils.io.workdir import AbstractFilePath, Workdir


_current_gradient_pass = contextvars.ContextVar("current_gradient_pass", default=None)
_current_gradient_writer = contextvars.ContextVar("current_gradient_writer", default=None)


@dataclass
class WriterScopeState:
    writer_id: str
    contribution_counts: dict[str, int] = field(default_factory=dict)
    pending_outputs: dict[str, tuple[str, Any]] = field(default_factory=dict)
    files_with_reader_hooks: set[str] = field(default_factory=set)
    reader_hook_handles: dict[str, list[RemovableHandle]] = field(default_factory=dict)


def gradient_path(file: str) -> str:
    base, extension = os.path.splitext(os.path.abspath(file))
    return f"{base}_gradient{extension}"


def is_active() -> bool:
    return _current_gradient_pass.get() is not None


def require_writer() -> WriterScopeState:
    writer = _current_gradient_writer.get()
    if writer is None:
        raise RuntimeError("Gradient contributions require an active use_gradient_writer(writer_id) scope.")
    return writer


def get_reader_hook_handles() -> dict[str, list[RemovableHandle]]:
    return require_writer().reader_hook_handles


def claim_gradient_reader(fpath: str) -> None:
    """Register one backward hook for a file in the current module/task.

    NOTE: A module may register at most one reader hook per file during its
    backward pass. A second hook inside a module is intentionally disabled (at least for now).
    """
    if not is_active():
        return
    writer = require_writer()
    if fpath in writer.files_with_reader_hooks:
        raise RuntimeError(
            f"File '{fpath}' registered more than one backward reader in the same module. "
            "Open each file once during the module's backward pass and reuse the "
            "returned tensor; selecting different column subsets with separate "
            "open() calls in one module is currently not supported."
        )
    writer.files_with_reader_hooks.add(fpath)


def allocate_contribution_path(file: str) -> str:
    writer = require_writer()
    destination = gradient_path(file)
    path = next_contribution_path(destination, _current_gradient_pass.get(), writer.writer_id, writer.contribution_counts)
    info = Workdir.get_path_info(file)
    if info is not None:
        Workdir._register_path(path, info)
    return path


def complete_gradient_output(file: str, path: str, backend) -> None:
    """Validate and route a completed gradient file, regardless of its producer."""
    if is_active():
        owner = require_writer().pending_outputs.pop(path)
        if owner != (file, backend):
            raise RuntimeError(f"Gradient output '{path}' belongs to a different primal or backend.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Backward task did not write its gradient output: {path}")
    for target, source, columns in backend.writer.gradient_file_contributions(file, path):
        if not is_active():
            value = backend.writer.load_gradient(source, columns=columns) if isinstance(source, str) else source
            destination = gradient_path(target)
            merge = destination in backend._gradient_written
            if merge:
                value = value + backend.writer.load_gradient(destination, columns=columns)
            backend.writer.write_gradient(destination, value, columns=columns, merge=merge)
            backend._gradient_written.add(destination)
            continue

        if not isinstance(source, str):
            contribution = allocate_contribution_path(target)
            backend.write_gradient_contribution(contribution, target, source, columns)
            source = contribution

        overlay = Workdir._current_gradient_overlays.get()
        destination = gradient_path(target)
        sources = reduction_sources(overlay.get(destination))
        if source in sources:
            continue
        overlay[destination] = make_reduction_resolver(
            sources + [source],
            destination,
            GradientAggregationResolver,
            generation=_current_gradient_pass.get(),
            context=f"gradient {os.path.basename(destination)}",
        )


def next_contribution_path(destination: str, pass_id: str | uuid.UUID, writer_id: str, counts: dict[str, int]) -> str:
    base, extension = os.path.splitext(destination)
    ordinal = counts.get(base, 0)
    counts[base] = ordinal + 1
    return f"{base}.contrib-{pass_id}-{writer_id}-{ordinal}{extension}"


def allocate_gradient_output(file: str, backend) -> str:
    if not is_active():
        base, extension = os.path.splitext(gradient_path(file))
        return f"{base}.contrib-{uuid.uuid4().hex}{extension}"
    path = allocate_contribution_path(file)
    # Reused outputs must not accept stale contents or truncate an inode
    # still exposed by the previous pass's adjacent gradient alias.
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    require_writer().pending_outputs[path] = (file, backend)
    return path


def gradient_destination(file: AbstractFilePath | str) -> tuple[str, str]:
    file = Workdir.resolve_file(file) if isinstance(file, AbstractFilePath) else os.fspath(file)
    destination = gradient_path(file)
    info = Workdir.get_path_info(file)
    if info is not None:
        Workdir._register_path(destination, info)
    return file, destination


def resolve_gradient(file: AbstractFilePath | str) -> str:
    file, destination = gradient_destination(file)
    overlay = Workdir._current_gradient_overlays.get()
    if overlay is None:
        return destination
    resolver = overlay.get(destination)
    if resolver is None:
        raise FileNotFoundError(
            f"No gradient contributions recorded for '{file}' in the current backward pass."
        )
    return resolve_value(resolver)


def resolve_gradient_output(file: AbstractFilePath | str, backend=None) -> str:
    from dsbo.config import get_active_file_backend

    backend = backend or get_active_file_backend()
    file, destination = gradient_destination(file)
    if not is_active():
        return destination
    for path, (primal, owner) in require_writer().pending_outputs.items():
        if gradient_path(primal) == destination and owner is backend:
            return path
    return allocate_gradient_output(file, backend)


@contextmanager
def use_gradient_pass(pass_id: str | uuid.UUID | None = None) -> Generator[dict[str, ResolvableSource], None, None]:
    """Start an isolated backward pass without creating a contribution writer."""
    pass_token = _current_gradient_pass.set(pass_id if pass_id is not None else uuid.uuid4())
    overlay = {}
    overlay_token = Workdir._current_gradient_overlays.set(overlay)
    writer_token = _current_gradient_writer.set(None)
    try:
        yield overlay
    finally:
        _current_gradient_writer.reset(writer_token)
        Workdir._current_gradient_overlays.reset(overlay_token)
        _current_gradient_pass.reset(pass_token)


@contextmanager
def use_gradient_writer(writer_id: str) -> Generator[dict[str, ResolvableSource], None, None]:
    if not is_active():
        raise RuntimeError("use_gradient_writer(writer_id) requires an active backward pass.")
    if not isinstance(writer_id, str) or not writer_id:
        raise ValueError("writer_id must be a non-empty string.")

    current = Workdir._current_gradient_overlays.get()
    overlay = dict(current)
    state = WriterScopeState(writer_id)
    overlay_token = Workdir._current_gradient_overlays.set(overlay)
    writer_token = _current_gradient_writer.set(state)
    try:
        try:
            yield overlay
            for path, (file, backend) in tuple(state.pending_outputs.items()):
                complete_gradient_output(file, path, backend)
        finally:
            with ExitStack() as cleanup:
                cleanup.callback(state.reader_hook_handles.clear)
                for reader_handles in state.reader_hook_handles.values():
                    for handle in reader_handles:
                        cleanup.callback(handle.remove)
        current.clear()
        current.update(overlay)
    finally:
        _current_gradient_writer.reset(writer_token)
        Workdir._current_gradient_overlays.reset(overlay_token)


@contextmanager
def gradient_output(file: str, backend) -> Generator[str, None, None]:
    path = allocate_gradient_output(file, backend)
    try:
        yield path
        complete_gradient_output(file, path, backend)
    finally:
        if not is_active():
            backend.remove(path)



