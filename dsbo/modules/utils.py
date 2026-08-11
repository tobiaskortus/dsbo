from __future__ import annotations

import torch

from dsbo.modules import Module
from dsbo.config import get_active_file_backend
from dsbo.utils.io import Output
from dsbo.utils.io.workdir import AbstractFilePath


class ResultRecorder(Module):
    def __init__(self, inner: Module, record_path: AbstractFilePath):
        super().__init__()
        self.inner = inner
        self.i = inner.i
        self._record_output = Output(record_path)

    @property
    def surrogate(self):
        raise NotImplementedError(
            "ResultRecorder is an internal module; it never carries a surrogate. "
            "Workflow and module surrogates cannot be combined."
        )

    @surrogate.setter
    def surrogate(self, value):
        raise NotImplementedError(
            "ResultRecorder is an internal module; it never carries a surrogate. "
            "Workflow and module surrogates cannot be combined."
        )

    @property
    def resources(self):
        return self.inner.resources

    @resources.setter
    def resources(self, value):
        if hasattr(self, "inner"):
            self.inner.resources = value

    @property
    def name(self):
        return f"{self.inner.name} (recorder)"

    @property
    def registered_inputs(self):
        return self.inner.registered_inputs

    @property
    def registered_outputs(self):
        return self.inner.registered_outputs + [self._record_output.path]

    def __call__(self, design: dict, **kwargs) -> dict:
        result = self.inner(design=design, **kwargs)
        fpath = self._record_output()
        get_active_file_backend().write_named_datasets(
            fpath,
            {
                key: value.detach().cpu().numpy()
                for key, value in result.items()
                if isinstance(value, torch.Tensor)
            },
        )
        return result

    def set_seed(self, seed: int):
        self.inner.set_seed(seed)
        return self

    def replicate(self, i: int):
        raise NotImplementedError(
            "ResultRecorder is an internal module and cannot be replicated."
        )

    def set_alias(self, alias: str):
        raise NotImplementedError(
            "ResultRecorder is an internal module and cannot be aliased."
        )

    def op_propagates_grad_(self, propagate_grads: bool = True):
        raise NotImplementedError(
            "ResultRecorder is an internal module; gradient propagation is not configurable."
        )
