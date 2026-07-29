from __future__ import annotations

import json
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dsbo.storage.sqlmodel import ArtifactStorage


class Artifact:
    def __init__(self, name: str, type: str, metadata: dict | None = None):
        self.name = name
        self.type = type
        self._metadata = metadata or {}
        self._models: dict[str, nn.Module] = {}
        self._version: int | None = None
        self._stage: str | None = None
        self._step: int | None = None

    @property
    def version(self) -> int | None:
        return self._version

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def add_model(self, model: nn.Module, name: str = "model") -> "Artifact":
        self._models[name] = model
        return self

    def get_model(self, name: str = "model") -> nn.Module:
        if name not in self._models:
            raise KeyError(
                f"No model named '{name}' in artifact '{self.name}'."
                f" Available: {list(self._models)}"
            )
        return self._models[name]

    def save(self, storage: "ArtifactStorage", stage: str | None = None, step: int | None = None) -> "Artifact":
        return storage.save_artifact(self, stage=stage, step=step)

    def _save_to_dir(self, artifact_dir: Path) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for model_name, model in self._models.items():
            path = artifact_dir / f"{model_name}.pt"
            torch.save(model.state_dict(), path)

        meta = {
            "name": self.name,
            "type": self.type,
            "version": self._version,
            "stage": self._stage,
            "step": self._step,
            "metadata": self._metadata,
            "created_at": time.time(),
        }
        with open(artifact_dir / "artifact.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def _load_from_dir(cls, artifact_dir: Path, name: str, type: str, version: int,
                       stage: str | None, step: int | None,
                       metadata: dict) -> "Artifact":
        artifact = cls(name, type, metadata)
        artifact._version = version
        artifact._stage = stage
        artifact._step = step

        for pt_path in artifact_dir.glob("*.pt"):
            model_name = pt_path.stem
            state = torch.load(pt_path, weights_only=True, map_location="cpu")
            artifact._models[model_name] = state

        return artifact
