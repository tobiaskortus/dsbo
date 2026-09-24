from __future__ import annotations

import os
import warnings
import networkx as nx
from contextlib import nullcontext

from dsbo.modules import Module
from dsbo.utils.io import Workdir
from dsbo.utils.io.workdir import AbstractFilePath
from dsbo.utils.io.reduce import LazyReductionResolver


def inner_module(module: Module) -> Module:
    return module.inner if getattr(module, "_is_batch_fused", False) else module


def is_batch_fused(module: Module) -> bool:
    return getattr(module, "_is_batch_fused", False)


def match_outputs_for(module: Module, input_path: AbstractFilePath) -> list[AbstractFilePath]:
    return [path for path in module.registered_outputs if path == input_path]


def resolve_input_overlay_paths(graph, task, input_path, index, candidate_subdirs):
    batch_outputs = []
    for predecessor in nx.ancestors(graph, task):
        if not is_batch_fused(predecessor):
            continue
        producer = inner_module(predecessor)
        batch_outputs.extend(
            Workdir.resolve_file(output_path, subdir="batch_reduce", suffix=producer.i, use_overlays=False)
            for output_path in match_outputs_for(producer, input_path)
        )

    candidate_producers = []
    for predecessor in graph.predecessors(task):
        if is_batch_fused(predecessor):
            continue
        producer = inner_module(predecessor)
        candidate_producers.extend(
            (producer, output_path)
            for output_path in match_outputs_for(producer, input_path)
        )

    if batch_outputs and candidate_producers:
        raise ValueError(f"Ambiguous batch and candidate producers for {input_path.filename}.")
    if len(batch_outputs) > 1:
        raise ValueError(f"Ambiguous batch producers for {input_path.filename}.")
    if batch_outputs:
        return batch_outputs, [producer.i for _ in batch_outputs]

    producers_sorted = sorted(
        candidate_producers,
        key=lambda item: item[0].i if item[0].i is not None else -1,
    )

    if not producers_sorted:
        return [], []

    indices = [p[0].i for p in producers_sorted]

    if index is None:
        return [
            Workdir.resolve_file(output_path, subdir=subdir, suffix=producer.i, use_overlays=False)
            for producer, output_path in producers_sorted
            for subdir in candidate_subdirs
        ], indices

    return [
        Workdir.resolve_file(output_path, subdir=candidate_subdirs[index], suffix=producer.i, use_overlays=False)
        for producer, output_path in producers_sorted
    ], indices


def _reduced_destination(input_path, producer_indices, subdir, suffix):
    base_name, ext = os.path.splitext(input_path.filename)
    indices = sorted(set(i for i in producer_indices if i is not None))
    label = f"_{indices[0]}-{indices[-1]}" if indices else "_merged"
    with Workdir.create_subdir(subdir) if subdir else nullcontext():
        dir_path = os.path.join(Workdir.get_current_dir(), input_path.pathname)
        suffix_str = f"_{suffix}" if suffix is not None else ""
        return os.path.join(dir_path, f"{base_name}{label}{suffix_str}{ext}")


def file_overlays(graph, task, index: int | None, candidate_subdirs):
    module = inner_module(task)
    overlays: dict[AbstractFilePath, LazyReductionResolver] = {}

    for input_path in module.registered_inputs:
        manual_overlay = Workdir.get_file_overlay(input_path)
        if manual_overlay is not None:
            overlays[input_path] = manual_overlay
            continue

        sources, producer_indices = resolve_input_overlay_paths(graph, task, input_path, index, candidate_subdirs)
        if not sources:
            continue

        sources = list(dict.fromkeys(sources))

        subdir = "batch_reduce" if index is None else candidate_subdirs[index]

        destination = _reduced_destination(input_path, producer_indices, subdir, module.i)

        # Direct overlay: single source points directly to the producer's output
        if len(sources) == 1:
            if sources[0] != destination:
                overlays[input_path] = sources[0]
            continue

        # Lazy reduction: multiple sources must be reduced before consumption
        overlays[input_path] = LazyReductionResolver(
            sources,
            destination,
            context=f"{module.name} input {input_path.filename}",
        )
        
    return overlays


def create_candidate_symlinks(module: Module, candidate_subdirs) -> None:
    inner = inner_module(module)
    targets = [subdir for subdir in candidate_subdirs if subdir is not None]
    for path in inner.registered_outputs:
        source = Workdir.resolve_file(path, subdir="batch_reduce", suffix=inner.i, use_overlays=False)
        Workdir.link(
            source,
            *(Workdir.resolve_file(path, subdir=subdir, suffix=inner.i, use_overlays=False) for subdir in targets),
        )


class FusedModule(Module):
    """
    NOTE: This module is currently not used, it was implemented to combine subsequent 
    modules on the same resources to avoid scheduling subsequent tasks that can be combined.

    TODO: Is this still something that is desired or can this be removed?
    """
    def __init__(self, *modules: Module):
        super().__init__()
        self.modules = modules
        for module in modules:
            assert self.modules[0].resources == module.resources, \
                "All modules in a FusedModule must have the same resources."

    def __call__(self, design: dict, **kwargs) -> dict:
        results = {}
        for module in self.modules:
            results.update(module(design=design))
        return results

    def run(self, design: dict, **kwargs) -> dict:
        results = {}
        for module in self.modules:
            results.update(module.run(design=design, **kwargs))
        return results

    @property
    def resources(self):
        return self.modules[0].resources

    @resources.setter
    def resources(self, value):
        pass

    @property
    def name(self):
        return "[" + ",".join(module.name for module in self.modules) + "]"

    @property
    def registered_inputs(self):
        return list(dict.fromkeys(
            path for module in self.modules for path in module.registered_inputs
        ))

    @property
    def registered_outputs(self):
        return list(dict.fromkeys(
            path for module in self.modules for path in module.registered_outputs
        ))


class BatchFusedModule(Module):
    """Container object for batch fused modules. It does not implement 
    any functionality and is only used by the framework internally."""
    _is_batch_fused = True

    def __init__(self, inner: Module):
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "_no_grad_op", inner._no_grad_op)

    def __getattr__(self, name):
        inner = self.__dict__.get("inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    @property
    def i(self):
        return self.inner.i

    @i.setter
    def i(self, value):
        self.inner.i = value

    @property
    def surrogate(self):
        return getattr(self.inner, "surrogate", None)

    @surrogate.setter
    def surrogate(self, value):
        self.inner.surrogate = value

    @property
    def surrogate_contract(self):
        return self.inner.surrogate_contract

    @property
    def resources(self):
        return self.inner.resources

    @resources.setter
    def resources(self, value):
        self.inner.resources = value

    @property
    def name(self):
        return self.inner.name

    @property
    def registered_files(self):
        return self.inner.registered_files

    @property
    def registered_inputs(self):
        return self.inner.registered_inputs

    @property
    def registered_outputs(self):
        return self.inner.registered_outputs

    def replicate(self, i: int):
        return type(self)(self.inner.replicate(i))

    def set_alias(self, alias: str):
        self.inner.set_alias(alias)
        return self

    def patch(self, surrogate):
        self.inner.patch(surrogate)
        return self

    def set_seed(self, seed: int):
        self.inner.set_seed(seed)
        return self

    def op_propagates_grad_(self, propagate_grads: bool = True):
        self.inner.op_propagates_grad_(propagate_grads)
        return self

    def resolve_file(self, file, *, subdir: str | None = None) -> str:
        return Workdir.resolve_file(file, subdir=subdir, suffix=self.inner.i, use_overlays=False)

    def resolve_derived_file(self, file, namespace: str, *, subdir: str | None = None) -> str:
        return Workdir.resolve_derived_file(file, namespace, subdir=subdir, suffix=self.inner.i)

    def __call__(self, design: dict, **kwargs) -> dict:
        result = self.inner(design=design, **kwargs)
        if result:
            warnings.warn(
                f"BatchFusedModule '{self.name}' returned a non-empty dict ({result}); "
                f"results are discarded."
            )
        return {}

    def run(self, design: dict, **kwargs) -> dict:
        result = self.inner.run(design=design, **kwargs)
        if result:
            warnings.warn(
                f"BatchFusedModule '{self.name}' returned a non-empty dict ({result}); "
                f"results are discarded."
            )
        return {}


def batch_fuse(module: Module) -> BatchFusedModule:
    if not isinstance(module, Module):
        raise TypeError("batch_fuse() expects a Module instance.")
    if isinstance(module, BatchFusedModule):
        raise ValueError("A module cannot be wrapped by batch_fuse() twice.")
    if not module.registered_inputs:
        raise ValueError(f"Batch-fused module {module.name} must register at least one Input.")
    module.op_propagates_grad_(False)
    return BatchFusedModule(module)
