"""YAML config loading with `${ENV_VAR}` expansion, so no script ever
needs a hard-coded local dataset path (per the task brief)."""
from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

__all__ = ["load_config"]

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _expand(value):
    if isinstance(value, str):
        def repl(m):
            var = m.group(1)
            resolved = os.environ.get(var)
            if resolved is None:
                return m.group(0)  # leave unresolved -- caller decides whether that's fatal
            return resolved
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _expand(raw)
