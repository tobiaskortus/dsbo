from __future__ import annotations

import math

import torch
import torch.autograd.forward_ad as fw
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from dsbo.autograd import train_op
from dsbo.config import get_active_file_backend
from dsbo.modules import Input, Module, dynamic_submodule
from dsbo.storage import get_current_storage
from dsbo.utils.io import DefaultPaths, Output, Workdir


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def shower_features(edep_per_cell: torch.Tensor) -> torch.Tensor:
    return edep_per_cell.flatten(start_dim=1)


class RegressionMetrics:
    def __init__(self) -> None:
        self.count = 0
        self.squared_error_sum = 0.0
        self.weighted_squared_error_sum = 0.0
        self.absolute_error_sum = 0.0
        self.target_sum = 0.0
        self.target_squared_sum = 0.0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        predictions = predictions.detach().cpu().reshape(-1).to(torch.float64)
        targets = targets.detach().cpu().reshape(-1).to(torch.float64)
        error = predictions - targets

        self.count += targets.numel()
        self.squared_error_sum += float((error**2).sum())
        self.weighted_squared_error_sum += float(((error**2) / (targets + 1.0)).sum())
        self.absolute_error_sum += float(error.abs().sum())
        self.target_sum += float(targets.sum())
        self.target_squared_sum += float((targets**2).sum())

    def compute(self) -> dict[str, torch.Tensor]:
        weighted_mse = self.weighted_squared_error_sum / self.count
        mse = self.squared_error_sum / self.count
        mae = self.absolute_error_sum / self.count
        target_mean = self.target_sum / self.count
        total_variance = self.target_squared_sum - self.count * target_mean**2
        r2 = 1.0 - self.squared_error_sum / max(total_variance, 1e-12)
        return {
            "loss": torch.tensor(weighted_mse, dtype=torch.float32),
            "weighted_mse": torch.tensor(weighted_mse, dtype=torch.float32),
            "mse": torch.tensor(mse, dtype=torch.float32),
            "mae": torch.tensor(mae, dtype=torch.float32),
            "rmse": torch.tensor(math.sqrt(mse), dtype=torch.float32),
            "r2": torch.tensor(r2, dtype=torch.float32),
        }


def regression_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    metrics = RegressionMetrics()
    metrics.update(predictions, targets)
    return metrics.compute()


def evaluate_set_metrics(model: ParticleEnergyMLP, loader: DataLoader, device: torch.device) -> dict[str, torch.Tensor]:
    model.eval()
    metrics = RegressionMetrics()
    with torch.no_grad():
        for designs, inputs, normalized_targets in loader:
            predictions = model.denormalize(model(designs.to(device), shower_features(inputs.to(device))))
            targets = model.denormalize(normalized_targets.to(device))
            metrics.update(predictions, targets)
    return metrics.compute()


def record_metrics(
    record_tag: str | None, 
    run_id: int, series: str, 
    epoch: int, 
    metrics: dict[str, torch.Tensor]
) -> None:
    storage = get_current_storage(create=False)
    if record_tag is None or storage is None:
        return
    for name, value in metrics.items():
        storage.intermediates.add_scalar(
            f"{record_tag}/{series}_{name}", epoch, float(value), run_id=run_id
        )


class ParticleEnergyMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        design_dim: int,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = design_dim + input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(width, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            width = hidden_dim
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)
        self.register_buffer("design_mean", torch.zeros(design_dim))
        self.register_buffer("design_std", torch.ones(design_dim))
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        self.register_buffer("target_mean", torch.zeros(()))
        self.register_buffer("target_std", torch.ones(()))

    def forward(self, design: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        design = (design - self.design_mean) / self.design_std
        x = (x - self.input_mean) / self.input_std
        return self.network(torch.cat([design, x], dim=1)).squeeze(-1)

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        return normalized * self.target_std + self.target_mean


def train_particle_energy_regressor(
    edep_per_cell: torch.Tensor,
    energies: torch.Tensor,
    designs: torch.Tensor,
    hidden_dim: int = 128,
    num_hidden_layers: int = 2,
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    validation_fraction: float = 0.1,
    seed: int = 0,
    record_tag: str | None = None,
    run_id: int = 0,
) -> tuple[ParticleEnergyMLP, dict[str, float | int | dict]]:
    device = torch.device(resolve_device())
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(edep_per_cell.shape[0], generator=generator)
    num_validation = int(indices.numel() * validation_fraction)
    validation_indices = indices[:num_validation].tolist()
    training_indices = indices[num_validation:].tolist()

    energies = energies.reshape(-1)
    target_mean = energies[training_indices].mean()
    target_std = energies[training_indices].std().clamp_min(1e-6)
    normalized_energies = (energies - target_mean) / target_std

    flat_edep = shower_features(edep_per_cell)
    input_mean = flat_edep[training_indices].mean(dim=0)
    input_std = flat_edep[training_indices].std(dim=0).clamp_min(1e-6)
    design_mean = designs[training_indices].mean(dim=0)
    design_std = designs[training_indices].std(dim=0).clamp_min(1.0)

    torch.manual_seed(seed)
    model = ParticleEnergyMLP(
        flat_edep.shape[1],
        designs.shape[1],
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
    ).to(device)
    model.input_mean.copy_(input_mean)
    model.input_std.copy_(input_std)
    model.design_mean.copy_(design_mean)
    model.design_std.copy_(design_std)
    model.target_mean.copy_(target_mean)
    model.target_std.copy_(target_std)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    dataset = TensorDataset(designs, edep_per_cell, normalized_energies)
    training_loader = DataLoader(
        Subset(dataset, training_indices),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    evaluation_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=batch_size,
    )

    metric_names = ("loss", "weighted_mse", "mse", "mae", "rmse", "r2")
    history = {f"{split}_{name}": [] for split in ("train", "validation") for name in metric_names}
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_loss = float("inf")
    best_epoch = 0
    best_validation_metrics: dict[str, torch.Tensor] | None = None

    for epoch in range(epochs):
        model.train()
        train_metrics = RegressionMetrics()
        for designs_batch, inputs, normalized_targets in training_loader:
            designs_batch = designs_batch.to(device)
            inputs = shower_features(inputs.to(device))
            normalized_targets = normalized_targets.to(device)
            optimizer.zero_grad()
            predictions = model.denormalize(model(designs_batch, inputs))
            targets = model.denormalize(normalized_targets)
            loss = (((predictions - targets) ** 2) / (targets + 1.0)).mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_metrics.update(predictions, targets)

        train_metrics = train_metrics.compute()
        validation_metrics = evaluate_set_metrics(model, evaluation_loader, device)
        record_metrics(record_tag, run_id, "train", epoch, train_metrics)
        record_metrics(record_tag, run_id, "validation", epoch, validation_metrics)

        for split, split_metrics in (
            ("train", train_metrics),
            ("validation", validation_metrics),
        ):
            for name, value in split_metrics.items():
                history[f"{split}_{name}"].append(float(value))

        validation_loss = float(validation_metrics["loss"])
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_validation_metrics = validation_metrics

    model.load_state_dict(best_state)
    train_metrics = evaluate_set_metrics(model, training_loader, device)
    model = model.cpu()
    return model, {
        "best_epoch": best_epoch,
        "train": train_metrics,
        "validation": best_validation_metrics,
        "history": history,
    }


class ParticleEnergyRegression(Module):
    def __init__(
        self,
        training_simulation: Module,
        evaluation_simulation: Module,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        epochs: int = 10,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        validation_fraction: float = 0.1,
        seed: int = 0,
        record_tag: str | None = "particle_energy_regression",
    ) -> None:
        super().__init__()
        self.training_edep = Input(training_simulation.simulation_file.path)
        self.training_energy = Input(training_simulation.energy_file.path)
        self.training_design = Input(training_simulation.design_file.path)
        self.evaluation_edep = Input(evaluation_simulation.simulation_file.path)
        self.evaluation_energy = Input(evaluation_simulation.energy_file.path)
        self.model_file = Output(Workdir.register_file(DefaultPaths.RECONSTRUCTION, "particle_energy_regressor.pt"))
        self.trained_model = Input(self.model_file.path)

        self.model_options = {
            "hidden_dim": hidden_dim,
            "num_hidden_layers": num_hidden_layers,
        }
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_fraction = validation_fraction
        self.seed = seed
        self.record_tag = record_tag
        self._train_call = 0

    @train_op
    @dynamic_submodule
    def train(self, design, **kwargs) -> dict[str, torch.Tensor]:
        storage = get_active_file_backend()
        with torch.no_grad():
            edep = storage.open(self.training_edep(), "cpu").detach()
            energies = storage.open(self.training_energy(), "cpu").reshape(-1).detach()
            designs = storage.open(self.training_design(), "cpu").detach()

        model, metrics = train_particle_energy_regressor(
            edep,
            energies,
            designs,
            seed=self.seed,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            validation_fraction=self.validation_fraction,
            record_tag=self.record_tag,
            run_id=self._train_call,
            **self.model_options,
        )
        self._train_call += 1
        torch.save(model.state_dict(), self.model_file())
        artifact_version = self._publish_artifact(model, metrics)
        artifact_version = float("nan") if artifact_version is None else float(artifact_version)
        return {
            "train_validation_loss": torch.tensor(float(metrics["validation"]["loss"])),
            "train_artifact_version": torch.tensor(artifact_version),
        }

    @dynamic_submodule
    def eval(self, design, **kwargs) -> dict[str, torch.Tensor]:
        loss, _, _, eval_metrics = self._evaluate(design)
        return {"loss": loss, **eval_metrics}

    @eval.backward
    def backward(self, design, grad_output=None) -> None:
        loss, _, _, _ = self._evaluate(design)
        loss.backward(grad_output["loss"])

    def _publish_artifact(self, model: ParticleEnergyMLP, metrics: dict) -> int | None:
        storage = get_current_storage(create=False)

        if self.record_tag is None or storage is None:
            return None

        artifact = storage.artifacts.create_artifact(
            self.record_tag, "model",
            metadata=self._artifact_metadata(model, metrics)
        )
        artifact.add_model(model, name="regressor")
        return storage.artifacts.save_artifact(artifact, stage="trained").version

    def _artifact_metadata(self, model: ParticleEnergyMLP, metrics: dict) -> dict:
            train_metrics = metrics["train"]
            validation_metrics = metrics["validation"]
            metadata = {
                "record_tag": self.record_tag,
                "best_epoch": metrics["best_epoch"],
                "target_mean": float(model.target_mean),
                "target_std": float(model.target_std),
                "train_loss": float(train_metrics["loss"]),
                "train_mse": float(train_metrics["mse"]),
                "train_mae": float(train_metrics["mae"]),
                "train_rmse": float(train_metrics["rmse"]),
                "train_r2": float(train_metrics["r2"]),
                "validation_loss": float(validation_metrics["loss"]),
                "validation_mse": float(validation_metrics["mse"]),
                "validation_mae": float(validation_metrics["mae"]),
                "validation_rmse": float(validation_metrics["rmse"]),
                "validation_r2": float(validation_metrics["r2"]),
                "train_loss_history": [float(value) for value in metrics["history"]["train_loss"]],
                "validation_loss_history": [
                    float(value) for value in metrics["history"]["validation_loss"]
                ],
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "validation_fraction": self.validation_fraction,
                "seed": self.seed,
                "device": resolve_device(),
            }
            metadata.update({f"option_{key}": value for key, value in self.model_options.items()})
            return metadata
    
    def _evaluate(
        self,
        design: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        device = resolve_device()
        storage = get_active_file_backend()
        edep = storage.open(self.evaluation_edep(), "cpu").to(device)
        design_vector = torch.cat(
            [
                design["absorber"].reshape(-1),
                design["gap"].reshape(-1),
            ]
        )
        design_tensor = design_vector.unsqueeze(0).expand(edep.shape[0], -1)
        with torch.no_grad():
            energies = storage.open(self.evaluation_energy(), "cpu").reshape(-1).to(device)
            state_dict = torch.load(self.trained_model(), map_location=device, weights_only=True)
            model = ParticleEnergyMLP(
                edep.shape[1:].numel(),
                design_tensor.shape[1],
                **self.model_options,
            ).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

        predictions = model.denormalize(model(design_tensor.to(device), shower_features(edep)))
        loss = (((predictions - energies) ** 2)  / (energies + 1.0)).mean()
        with torch.no_grad():
            primal_predictions = fw.unpack_dual(predictions).primal
            metrics = regression_metrics(primal_predictions, energies)
            eval_metrics = {
                "eval_loss": metrics["loss"],
                "eval_mse": metrics["mse"],
                "eval_mae": metrics["mae"],
                "eval_rmse": metrics["rmse"],
                "eval_r2": metrics["r2"],
            }
        return (
            loss.to("cpu"),
            predictions.detach().cpu(),
            energies.cpu(),
            eval_metrics,
        )
