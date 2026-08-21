from __future__ import annotations

import torch


class Standardizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean) / self.std

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.std + self.mean

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        mean = torch.as_tensor(state_dict["mean"], dtype=torch.float32)
        std = torch.as_tensor(state_dict["std"], dtype=torch.float32)
        self.mean = mean
        self.std = std
