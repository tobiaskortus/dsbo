
from __future__ import annotations

import time
import multiprocessing as mp
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import lightning as L
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from dsbo.logger import WorkflowTaskLogger
from dsbo.storage import get_current_storage

if TYPE_CHECKING:
    from dsbo.surrogates.surrogate_model import SurrogateModel


def is_distributed() -> bool:
    return dist.is_initialized() and dist.get_world_size() > 1

def is_rank_zero() -> bool:
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def maybe_distributed(loader: DataLoader, shuffle: bool = False) -> DataLoader:
    """Shard ``loader`` across DDP ranks via :class:`DistributedSampler`.‚
    No-op (returns ``loader`` unchanged) when not running distributed (ddp).
    """
    if not is_distributed():
        return loader
    sampler = DistributedSampler(
        loader.dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=shuffle,
    )
    kwargs = {key: getattr(loader, key) for key in (
        "batch_size", "num_workers", "collate_fn", "drop_last", "pin_memory",
        "timeout",
    ) if getattr(loader, key, None) is not None}
    num_workers = kwargs.get("num_workers", 0)
    if num_workers:
        kwargs["prefetch_factor"] = loader.prefetch_factor
        kwargs["persistent_workers"] = loader.persistent_workers
    return DataLoader(loader.dataset, sampler=sampler, **kwargs)


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


class DaskProcessDDPStrategy(L.pytorch.strategies.DDPStrategy):
    """A ``DDPStrategy`` that runs inside a process group owned by Dask.

    Under worker-per-rank dispatch each Dask worker is *already* one DDP rank
    and ``torch.distributed.init_process_group`` has been called by the Dask
    dispatch wrapper. This strategy therefore skips Lightning's subprocess
    launch (``setup_distributed``) while still wrapping the LightningModule in
    ``DistributedDataParallel`` so gradients are synchronized across ranks.
    """

    @property
    def root_device(self) -> torch.device:
        assert self.parallel_devices is not None
        return self.parallel_devices[0]

    @property
    def num_processes(self) -> int:
        if dist.is_initialized():
            return dist.get_world_size()
        return super().num_processes

    def setup_distributed(self) -> None:
        # The process group is already initialized by the Dask dispatch wrapper.
        # Set world ranks from the env and record the existing backend, but do
        # not call ``_init_dist_connection`` (which would spawn a new group).
        self.set_world_ranks()
        self._process_group_backend = self._get_process_group_backend()

    def _get_process_group_backend(self):
        if dist.is_initialized():
            return dist.get_backend()
        return super()._get_process_group_backend()



class MetricCallback(L.Callback):
    def __init__(self, model_tag: str | None = None):
        super().__init__()
        self.model_tag = model_tag

    def _log(self, trainer: L.Trainer, pl_module: L.LightningModule, epoch: int, name: str) -> None:
        metric = trainer.callback_metrics.get(f"{name}_loss")
        if metric is None:
            return
        value = float(metric)
        if is_rank_zero():
            #Log to WorkflowTaskLogger
            WorkflowTaskLogger.get_current().log(f"epoch {epoch} {name}_loss: {value:.6f}")

            #Log to intermediate storage
            storage = get_current_storage()
            if storage is not None and self.model_tag is not None:
                storage.intermediates.add_scalar(f"{self.model_tag}/{name}_loss", epoch, value)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self._log(trainer, pl_module, trainer.current_epoch, "train")

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if not trainer.sanity_checking:
            self._log(trainer, pl_module, trainer.current_epoch, "val")
    


class _PerfCallback(L.Callback):
    """Measures per-batch data loading, training step, and communication time,
    writing the breakdown to the running task's log file on the master rank.
    """
    LOG_EVERY_N = 100

    def __init__(self, log_every_n: int = LOG_EVERY_N):
        super().__init__()
        self.log_every_n = log_every_n
        self._batch_start = 0.0
        self._prev_batch_end = 0.0
        self._fit_start: float | None = None
        self._first_batch_logged = False
        self._comm = torch.zeros((1 << 20) // 4, dtype=torch.float32)  # 1 MiB
        self._metrics = {"data_load": [], "train_step": [], "comm": []}

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.perf_counter()
        if self._prev_batch_end:
            self._metrics["data_load"].append((now - self._prev_batch_end) * 1000.0)
        self._batch_start = now
        if not self._first_batch_logged and self._fit_start is not None:
            self._first_batch_logged = True
            if is_rank_zero():
                WorkflowTaskLogger.get_current().log(
                    f"perf: time to first training step = {(now - self._fit_start) * 1000.0:.1f}ms"
                )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = (time.perf_counter() - self._batch_start) * 1000.0
        self._prev_batch_end = time.perf_counter()
        self._metrics["train_step"].append(step)
        # all_reduce is collective: it must run on every rank. Logging is rank 0.
        comm = self._measure_comm()
        self._metrics["comm"].append(comm)
        if is_rank_zero() and self.log_every_n > 0 and (batch_idx + 1) % self.log_every_n == 0:
            WorkflowTaskLogger.get_current().log(
                f"perf step {batch_idx + 1}: train_step={step:.2f}ms comm={comm:.2f}ms"
            )

    def _measure_comm(self) -> float:
        if not is_distributed():
            return 0.0
        t0 = time.perf_counter()
        dist.all_reduce(self._comm)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    def on_train_epoch_end(self, trainer, pl_module):
        m = self._metrics
        if m["data_load"] and is_rank_zero():
            means = {k: sum(v) / len(v) for k, v in m.items()}
            WorkflowTaskLogger.get_current().log(
                f"perf epoch {trainer.current_epoch}: "
                f"data_load mean={means['data_load']:.2f}ms "
                f"train_step mean={means['train_step']:.2f}ms "
                f"comm mean={means['comm']:.2f}ms"
            )
        for key in m:
            m[key].clear()



def _build_lightning_trainer(
    config: TrainingConfig,
    accelerator: str = "auto",
    devices: Any = "auto",
    logger: bool = False,
    log_every_n_steps: int = 50,
    enable_progress_bar: bool = False,
    enable_model_summary: bool = False,
    enable_checkpointing: bool = False,
    resources=None,
    callbacks=None,
    **kwargs,
) -> L.Trainer:
    callbacks = list(callbacks or [])
    strategy = kwargs.pop("strategy", "auto")

    distributed = is_distributed()
    ranks = getattr(resources, "n_ranks", 1)
    if not distributed and ranks > 1:
        import warnings

        warnings.warn(
            f"Worker spec requests n_ranks={ranks} (DDP) but no process group is "
            "active in this process; falling back to single-device training. This "
            "usually means the module was not dispatched through the DDP rank "
            "wrapper.",
            RuntimeWarning,
            stacklevel=2,
        )

    if distributed:
        strategy = DaskProcessDDPStrategy()
        # DDP is unsupported on the MPS accelerator (Lightning rejects the DDP strategy on `mps`).
        if accelerator in ("auto", None) or accelerator == "mps":
            accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        devices = 1

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
        self._lightning = _build_lightning_trainer(
            self.config,
            accelerator=accelerator,
            devices=devices,
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
            enable_checkpointing=enable_checkpointing,
            resources=resources,
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
            train_loader.num_workers > 0
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

        train_loader = maybe_distributed(train_loader, shuffle=True)
        if val_loader is not None:
            val_loader = maybe_distributed(val_loader, shuffle=False)

        metric = MetricCallback(model_tag=model_tag)
        perf = _PerfCallback()
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
    if is_rank_zero():
        torch.save(state, path)
