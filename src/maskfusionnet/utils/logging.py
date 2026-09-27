"""Minimal CSV + console training logger (no external experiment-tracking
service required -- keeps the stack small, per the task brief's guidance
against adding infrastructure the project doesn't need)."""
from __future__ import annotations

import csv
import time
from pathlib import Path

__all__ = ["MetricLogger"]


class MetricLogger:
    def __init__(self, log_dir: str | Path, filename: str = "metrics.csv"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / filename
        self._fieldnames: list[str] | None = None
        self._start = time.time()

    def log(self, step: int, **metrics):
        row = {"step": step, "elapsed_s": round(time.time() - self._start, 2), **metrics}
        new_file = not self.path.exists()
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        msg = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items())
        print(f"[step {step}] {msg}")
