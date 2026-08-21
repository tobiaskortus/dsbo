from __future__ import annotations

import json
import shutil
import time
import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dsbo.storage.sqlmodel import ArtifactStorage


def _is_plain_basename(name: str) -> bool:
    if name in (".", ".."):
        return False
    normalized = name.replace("\\", "/")
    return PurePosixPath(normalized).name == normalized


@dataclass(frozen=True)
class ArtifactRef:
    name: str
    type: str
    version: int
    stage: str | None = None


class Artifact:
    def __init__(self, name: str, type: str, metadata: dict | None = None):
        self.name = name
        self.type = type
        self._metadata = metadata or {}
        self._models: dict[str, nn.Module] = {}
        self._files: dict[str, Path] = {}
        self._version: int | None = None
        self._stage: str | None = None
        self._step: int | None = None

    @property
    def version(self) -> int | None:
        return self._version

    @property
    def stage(self) -> str | None:
        return self._stage

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    @property
    def ref(self) -> ArtifactRef:
        if self._version is None:
            raise RuntimeError("Artifact must be saved before creating a reference.")
        return ArtifactRef(self.name, self.type, self._version, self._stage)

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

    def add_file(self, path: str | Path, name: str | None = None) -> "Artifact":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No file at '{path}'.")

        file_name = name or path.name
        if not _is_plain_basename(file_name):
            raise ValueError(
                f"Invalid file name '{file_name}': names must be a plain basename."
            )

        self._files[file_name] = path
        return self

    def get_file(self, name: str) -> Path:
        if name not in self._files:
            raise KeyError(
                f"No file named '{name}' in artifact '{self.name}'."
                f" Available: {list(self._files)}"
            )
        return self._files[name]

    def save(self, storage: "ArtifactStorage", stage: str | None = None, step: int | None = None) -> "Artifact":
        return storage.save_artifact(self, stage=stage, step=step)

    def _save_to_dir(self, artifact_dir: Path) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for model_name, model in self._models.items():
            torch.save(model.state_dict(), artifact_dir / f"{model_name}.pt")

        files_dir = artifact_dir / "files"
        for file_name, src in self._files.items():
            dest = files_dir / file_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)

    def _write_manifest(self, artifact_dir: Path) -> None:
        """Write a self-contained, human-readable manifest for the artifact dir.

        The DB row remains authoritative for loading, but this duplicate record is
        kept so an artifact version can be identified/debugged by its directory
        alone (without any storage).
        """
        manifest = {
            "name": self.name,
            "type": self.type,
            "version": self._version,
            "stage": self._stage,
            "step": self._step,
            "metadata": self._metadata,
            "models": sorted(self._models.keys()),
            "files": sorted(self._files.keys()),
            "created_at": time.time(),
        }
        with open(artifact_dir / "artifact.json", "w") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def _load_from_dir(cls, artifact_dir: Path, name: str, type: str, version: int,
                       stage: str | None, step: int | None,
                       metadata: dict) -> "Artifact":
        with open(artifact_dir / "artifact.json") as f:
            manifest = json.load(f)

        artifact = cls(name, type, metadata)
        artifact._version = version
        artifact._stage = stage
        artifact._step = step

        for model_name in manifest.get("models", []):
            state = torch.load(
                artifact_dir / f"{model_name}.pt",
                weights_only=True,
                map_location="cpu",
            )
            artifact._models[model_name] = state

        files_dir = artifact_dir / "files"
        for file_name in manifest.get("files", []):
            artifact._files[file_name] = files_dir / file_name

        return artifact
