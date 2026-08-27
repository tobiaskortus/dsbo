from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PendingEST:
    rank: int
    gradient: torch.Tensor
    mask: torch.Tensor
    loss: float
    metrics: dict[str, tuple]
