from pathlib import Path
import io
import json
import zipfile
import torch
import torch.nn as nn   

class LocalModelRegistry:
    def __init__(self, archive_path: str | Path):
        self.archive_path = Path(archive_path)

    def _data_name(self, tag: str, step: int) -> str:
        return f"{tag}_{step}.pt"

    def save(self, artifact: nn.Module, tag: str, step: int) -> None:
        buffer = io.BytesIO()
        torch.save(artifact.state_dict(), buffer)
        data_name = self._data_name(tag, step)

        if self.archive_path.exists():
            with zipfile.ZipFile(self.archive_path, "r") as zf:
                if data_name in zf.namelist():
                    self._replace_entry(data_name, buffer.getvalue())
                    return

        with zipfile.ZipFile(self.archive_path, "a", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr(data_name, buffer.getvalue())

    def _replace_entry(self, name: str, data: bytes) -> None:
        """Replace an existing entry in the zip archive, preserving all other entries."""
        temp_path = self.archive_path.with_suffix(".tmp.zip")
        with zipfile.ZipFile(self.archive_path, "r") as src:
            with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_STORED) as dst:
                for info in src.infolist():
                    if info.filename == name:
                        continue
                    dst.writestr(info, src.read(info.filename))
                dst.writestr(name, data)
        temp_path.replace(self.archive_path)

    def load(self, tag: str, step: int | None = None) -> dict:
        if step is None:
            step = self._latest_step(tag)
            if step is None:
                raise FileNotFoundError(f"No artifact found for tag '{tag}'")
            
        with zipfile.ZipFile(self.archive_path, "r") as zf:
            data = zf.read(self._data_name(tag, step))
        return torch.load(io.BytesIO(data), weights_only=True)

    def _latest_step(self, tag: str) -> int | None:
        with zipfile.ZipFile(self.archive_path, "r") as zf:
            names = zf.namelist()
        steps = []
        prefix = f"{tag}_"
        for name in names:
            if name.startswith(prefix) and name.endswith(".pt"):
                try:
                    step_str = name[len(prefix):-3]  # strip prefix and .pt
                    steps.append(int(step_str))
                except ValueError:
                    continue
        return max(steps) if steps else None