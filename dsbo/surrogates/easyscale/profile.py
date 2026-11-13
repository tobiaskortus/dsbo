from __future__ import annotations

import time
import torch
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Generator


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def timed(values: dict[str, float] | None, name: str, *, device: torch.device | None = None) -> Generator[None]:
    if values is None:
        yield
        return
    if device is not None:
        synchronize(device)
    started = time.perf_counter()
    try:
        yield
    finally:
        if device is not None:
            synchronize(device)
        values[name] = values.get(name, 0.0) + (time.perf_counter() - started)


def profile_details(timings: Mapping[str, float], prefix: str = "profile_mean_") -> str:
    return " ".join(
        f"{key.removeprefix(prefix)}={value * 1000.0:.2f}ms"
        for key, value in timings.items()
        if key.startswith(prefix)
    )
