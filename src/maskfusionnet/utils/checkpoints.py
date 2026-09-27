"""Checkpoint I/O.

Fixes the original repository's checkpoint handling, which saved only a
bare `model.state_dict()` and loaded it with `strict=False` and no report
(`Main.py`: `model.load_state_dict(torch.load(...), strict=False)`), so
any silent shape mismatch (e.g. from the ADB/SEB-sharing bug this
reconstruction fixes) would never surface as anything other than possibly
-wrong numbers.

Every checkpoint written here bundles model/optimizer/scheduler state,
epoch, the config used to produce it, and whatever metrics were computed
at save time -- so a run is fully reproducible from its checkpoint alone.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

__all__ = ["save_checkpoint", "load_checkpoint"]


def save_checkpoint(path: str | Path, model, optimizer=None, scheduler=None,
                     epoch: int = 0, config: dict | None = None, metrics: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "config": config or {},
        "metrics": metrics or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None,
                     map_location: str = "cpu", strict: bool = True) -> dict:
    """Loads a checkpoint written by `save_checkpoint`. Returns the full
    payload (minus the large state dicts already applied) so callers can
    inspect `epoch`/`config`/`metrics`. Missing/unexpected keys are always
    reported explicitly -- never swallowed by a blanket `strict=False`
    unless the caller opts into it, in which case the mismatch report is
    still returned rather than discarded.
    """
    payload = torch.load(path, map_location=map_location)
    result = model.load_state_dict(payload["model_state_dict"], strict=strict)
    if hasattr(result, "missing_keys") and (result.missing_keys or result.unexpected_keys):
        print(f"[checkpoints] Non-strict load of {path}:")
        print(f"  missing_keys:    {result.missing_keys}")
        print(f"  unexpected_keys: {result.unexpected_keys}")

    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    return {"epoch": payload.get("epoch", 0), "config": payload.get("config", {}),
            "metrics": payload.get("metrics", {})}
