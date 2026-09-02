from __future__ import annotations

import time
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import lightning as L
from torch.utils.data import DataLoader

from dsbo.logger import WorkflowTaskLogger
from dsbo.surrogates.easyscale.protocol import EasyScaleSchedulerConfig
from dsbo.storage import get_current_storage

if TYPE_CHECKING:
    from dsbo.surrogates.surrogate_model import SurrogateModel

@dataclass
class TrainingConfig:
    """Carries the run and data-loading configuration for now
    TODO: rethink and optionally implement a cleaner config.
    """
    max_epochs: int
    grad_clip: float | None
    batch_size: int
    val_split: float
    patience: int | None = None
    num_workers: int = 1
    prefetch_factor: int = 4
    seed: int = 0
    virtual_world_size: int = 1
    elastic: bool | None = None
    easyscale_scheduler: EasyScaleSchedulerConfig = field(
        default_factory=EasyScaleSchedulerConfig
    )

    def __post_init__(self) -> None:
        if isinstance(self.virtual_world_size, bool) or not isinstance(self.virtual_world_size, int):
            raise TypeError("virtual_world_size must be an integer.")
        if self.virtual_world_size < 1:
            raise ValueError("virtual_world_size must be positive.")
        if self.elastic is None:
            self.elastic = self.virtual_world_size > 1


class MetricCallback(L.Callback):
    def __init__(self, model_tag: str | None = None):
        super().__init__()
        self.model_tag = model_tag

    def _log(self, trainer: L.Trainer, pl_module: L.LightningModule, epoch: int, name: str) -> None:
        if not trainer.is_global_zero:
            return
        metric = trainer.callback_metrics.get(f"{name}_loss")
        if metric is None:
            return
        
        value = float(metric)
        WorkflowTaskLogger.get_current().log(f"epoch {epoch} {name}_loss: {value:.6f}")
        
        storage = get_current_storage()
        if storage is not None and self.model_tag is not None:
            storage.intermediates.add_scalar(f"{self.model_tag}/{name}_loss", epoch, value)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self._log(trainer, pl_module, trainer.current_epoch, "train")

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if not trainer.sanity_checking:
            self._log(trainer, pl_module, trainer.current_epoch, "val")
    


class PerfCallback(L.Callback):
    """Measures per-batch data loading and training step time."""
    LOG_EVERY_N = 10

    def __init__(self, log_every_n: int = LOG_EVERY_N):
        super().__init__()
        self.log_every_n = log_every_n
        self._batch_start = 0.0
        self._prev_batch_end = 0.0
        self._fit_start: float | None = None
        self._first_batch_logged = False
        self._metrics = {"data_load": [], "train_step": []}

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not trainer.is_global_zero:
            return
        now = time.perf_counter()
        if self._prev_batch_end:
            self._metrics["data_load"].append((now - self._prev_batch_end) * 1000.0)
        self._batch_start = now
        if not self._first_batch_logged and self._fit_start is not None:
            self._first_batch_logged = True
            WorkflowTaskLogger.get_current().log(
                f"perf: time to first training step = {(now - self._fit_start) * 1000.0:.1f}ms"
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return
        step = (time.perf_counter() - self._batch_start) * 1000.0
        self._prev_batch_end = time.perf_counter()
        self._metrics["train_step"].append(step)
        if self.log_every_n > 0 and (batch_idx + 1) % self.log_every_n == 0:
            WorkflowTaskLogger.get_current().log(
                f"perf step {batch_idx + 1}: train_step={step:.2f}ms"
            )
            profile = getattr(trainer.strategy, "last_step_profile", None)
            if profile:
                strategy = trainer.strategy
                profile_fields = (
                    ("est_data_transfer", "data_transfer"),
                    ("est_compute", "est_compute"),
                    ("gradient_metadata_all_reduce", "gradient_metadata_all_reduce"),
                    ("gradient_tensor_all_reduce", "gradient_tensor_all_reduce"),
                    ("gradient_aggregation_total", "gradient_aggregation"),
                    ("metric_all_reduce", "metric_all_reduce"),
                    ("metric_aggregation_total", "metric_aggregation"),
                    ("gradient_assignment", "gradient_assignment"),
                    ("backward_total", "backward"),
                    ("step_total", "step_total"),
                )
                details = " ".join(
                    f"{label}={profile.get(name, 0.0) * 1000.0:.2f}ms"
                    for name, label in profile_fields
                )
                WorkflowTaskLogger.get_current().log(
                    f"easyscale profile step={batch_idx + 1} "
                    f"rank={getattr(strategy, 'physical_rank', '?')} "
                    f"world={getattr(strategy, 'physical_world_size', '?')} "
                    f"ests={getattr(strategy, 'assigned_ests', ())} "
                    f"gradient_mb={profile.get('gradient_elements', 0.0) * 4.0 / 1e6:.2f} "
                    f"{details}"
                )

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        metrics = self._metrics
        if metrics["data_load"]:
            means = {key: sum(values) / len(values) for key, values in metrics.items()}
            WorkflowTaskLogger.get_current().log(
                f"perf epoch {trainer.current_epoch}: "
                f"data_load mean={means['data_load']:.2f}ms "
                f"train_step mean={means['train_step']:.2f}ms"
            )
        for values in metrics.values():
            values.clear()


def _build_lightning_trainer(
    config: TrainingConfig,
    accelerator: str = "auto",
    devices: Any = "auto",
    logger: bool = False,
    log_every_n_steps: int = 50,
    enable_progress_bar: bool = False,
    enable_model_summary: bool = False,
    enable_checkpointing: bool = False,
    callbacks=None,
    **kwargs,
) -> L.Trainer:
    callbacks = list(callbacks or [])
    strategy = kwargs.pop("strategy", "auto")

    return L.Trainer(
        max_epochs=config.max_epochs,
        strategy=strategy,
        accelerator=accelerator,
        devices=devices,
        gradient_clip_val=config.grad_clip,
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=enable_model_summary,
        enable_checkpointing=enable_checkpointing,
        callbacks=callbacks,
        **kwargs,
    )


class Trainer:
    def __init__(
        self,
        config: TrainingConfig | None = None,
        *,
        accelerator: str = "auto",
        devices: Any = "auto",
        resources=None,
        logger: bool = False,
        log_every_n_steps: int = 50,
        enable_progress_bar: bool = False,
        enable_model_summary: bool = False,
        enable_checkpointing: bool = False,
        callbacks=None,
        **kwargs,
    ):
        self.config = (
            config
            if config is not None
            else TrainingConfig(
                max_epochs=100,
                grad_clip=1.0,
                batch_size=32,
                val_split=0.2,
            )
        )
        self.resources = resources
        self._callbacks = list(callbacks or [])
        self._elastic = self.config.elastic
        if self._elastic:
            if enable_checkpointing:
                raise ValueError(
                    "EasyScale uses internal epoch-boundary checkpoints; persistent "
                    "Lightning checkpointing is not supported."
                )
            if kwargs.get("accumulate_grad_batches", 1) != 1:
                raise ValueError("EasyScale requires accumulate_grad_batches=1.")
            self._lightning = None
        else:
            self._lightning = _build_lightning_trainer(
                self.config,
                accelerator=accelerator,
                devices=devices,
                logger=logger,
                log_every_n_steps=log_every_n_steps,
                enable_progress_bar=enable_progress_bar,
                enable_model_summary=enable_model_summary,
                enable_checkpointing=enable_checkpointing,
                callbacks=callbacks,
                **kwargs,
            )

    def train(
        self,
        model: SurrogateModel,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        model_tag: str | None = None,
    ) -> None:
        from dsbo.surrogates.surrogate_model import SurrogateModel

        assert isinstance(model, SurrogateModel), "model must be a SurrogateModel"
        if mp.current_process().daemon and (
            (not self._elastic and train_loader.num_workers > 0)
            or (val_loader is not None and val_loader.num_workers > 0)
        ):
            raise RuntimeError(
                "PyTorch DataLoader workers cannot be started from a daemonic "
                "Dask worker. Configure WorkerSpec(daemon=False) or set "
                "TrainingConfig(num_workers=0)."
            )
        if val_loader is not None and not model.implements_validation():
            raise ValueError(
                "A validation loader requires the model to implement validation_step."
            )
        if self.config.patience is not None and val_loader is None:
            raise ValueError("TrainingConfig.patience requires a validation loader.")

        if self._elastic:
            if not model.automatic_optimization:
                raise ValueError("EasyScale requires Lightning automatic optimization.")

            from distributed import worker_client
            from dsbo.surrogates.easyscale import EasyScaleController

            with worker_client(separate_thread=False) as client:
                EasyScaleController(client).fit(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    training_config=self.config,
                    resources=self.resources,
                    model_tag=model_tag or "surrogate",
                    callbacks=[
                        *self._callbacks,
                        MetricCallback(model_tag=model_tag),
                        PerfCallback(),
                    ],
                )
            return

        metric = MetricCallback(model_tag=model_tag)
        perf = PerfCallback()
        run_callbacks = [metric, perf]
        if self.config.patience is not None:
            from lightning.pytorch.callbacks import EarlyStopping

            run_callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.config.patience,
                    mode="min",
                )
            )

        assert self._lightning is not None
        base_callbacks = list(self._lightning.callbacks)
        self._lightning.callbacks = [*base_callbacks, *run_callbacks]
        try:
            self._lightning.fit(
                model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )
        finally:
            self._lightning.callbacks = base_callbacks


def save_checkpoint(state: dict, path) -> None:
    torch.save(state, path)
