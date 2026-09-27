"""Generic training loop shared by Stage 1 (pretrain.py) and Stage 2
(finetune.py). Neither stage existed at all in the original repository --
there was no training code whatsoever, only a live-inference `Main.py`.

Design goals, per the task brief: GPU support, optional mixed precision,
checkpoint/resume, and a training loop small enough to read in one sitting
rather than a heavyweight framework.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from ..utils.checkpoints import save_checkpoint, load_checkpoint
from ..utils.logging import MetricLogger

__all__ = ["Trainer"]


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step_fn: Callable[[torch.nn.Module, dict, torch.device, int], dict],
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        use_amp: bool = False,
        grad_accum_steps: int = 1,
        checkpoint_dir: str | Path = "checkpoints",
        log_dir: str | Path = "logs",
        config: dict | None = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.step_fn = step_fn
        self.device = torch.device(device)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.logger = MetricLogger(log_dir)
        self.config = config or {}
        self.start_epoch = 1
        self.best_metric = float("inf")

    def resume(self, checkpoint_path: str | Path):
        info = load_checkpoint(checkpoint_path, self.model, self.optimizer, self.scheduler,
                                map_location=str(self.device))
        self.start_epoch = info["epoch"] + 1
        self.best_metric = info["metrics"].get("best_metric", float("inf"))
        print(f"[Trainer] Resumed from {checkpoint_path} at epoch {self.start_epoch}, "
              f"best_metric={self.best_metric}")

    def _run_epoch(self, loader: DataLoader, epoch: int, train: bool) -> float:
        self.model.train(train)
        total_loss, n_batches = 0.0, 0
        self.optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            with torch.set_grad_enabled(train):
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    out = self.step_fn(self.model, batch, self.device, epoch)
                    loss = out["loss"]
            if train:
                self.scaler.scale(loss / self.grad_accum_steps).backward()
                if (i + 1) % self.grad_accum_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().item())
            n_batches += 1
        return total_loss / max(1, n_batches)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None, epochs: int,
            checkpoint_every: int = 1):
        for epoch in range(self.start_epoch, epochs + 1):
            train_loss = self._run_epoch(train_loader, epoch, train=True)
            val_loss = self._run_epoch(val_loader, epoch, train=False) if val_loader is not None else None
            if self.scheduler is not None:
                self.scheduler.step()

            log_kwargs = {"train_loss": train_loss}
            if val_loss is not None:
                log_kwargs["val_loss"] = val_loss
            self.logger.log(epoch, **log_kwargs)

            metric_for_best = val_loss if val_loss is not None else train_loss
            is_best = metric_for_best < self.best_metric
            self.best_metric = min(self.best_metric, metric_for_best)

            if epoch % checkpoint_every == 0:
                save_checkpoint(self.checkpoint_dir / f"epoch_{epoch}.pt", self.model, self.optimizer,
                                 self.scheduler, epoch, self.config, {"train_loss": train_loss,
                                                                       "val_loss": val_loss,
                                                                       "best_metric": self.best_metric})
            if is_best:
                save_checkpoint(self.checkpoint_dir / "best.pt", self.model, self.optimizer,
                                 self.scheduler, epoch, self.config, {"train_loss": train_loss,
                                                                       "val_loss": val_loss,
                                                                       "best_metric": self.best_metric})
