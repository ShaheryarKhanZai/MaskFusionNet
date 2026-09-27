"""Subject-level dataset splitting.

The original repository had no dataset/training code at all, so there was
no split logic to audit -- this is new. The one hard requirement from the
task brief is enforced here: splits are made over *subjects*, never over
individual clips or videos, so no subject appears in more than one split
(a per-clip random split would leak temporally-adjacent frames of the same
person's video across train/val/test, inflating reported accuracy).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

__all__ = ["subject_level_split", "SplitResult"]


@dataclass
class SplitResult:
    train: list
    val: list
    test: list

    def as_dict(self) -> dict:
        return {"train": self.train, "val": self.val, "test": self.test}


def subject_level_split(subject_ids: list, train_frac: float = 0.7, val_frac: float = 0.15,
                         seed: int = 42) -> SplitResult:
    """Shuffle subjects deterministically (given `seed`) and cut into three
    disjoint groups. `test_frac` is implied as 1 - train_frac - val_frac.

    Raises if fractions don't leave a positive test split, or if there are
    too few subjects to populate all three splits.
    """
    assert 0 < train_frac < 1 and 0 <= val_frac < 1
    test_frac = 1 - train_frac - val_frac
    assert test_frac > 0, "train_frac + val_frac must be < 1"

    subjects = sorted(set(subject_ids))  # de-dup, deterministic starting order
    if len(subjects) < 3:
        raise ValueError(
            f"Need at least 3 distinct subjects for a train/val/test split, got {len(subjects)}."
        )

    rng = random.Random(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = max(1, round(n * train_frac))
    n_val = max(1, round(n * val_frac)) if val_frac > 0 else 0
    n_train = min(n_train, n - 2)  # always leave >=1 for val slot's neighbor + test
    n_val = min(n_val, n - n_train - 1)

    train = subjects[:n_train]
    val = subjects[n_train:n_train + n_val]
    test = subjects[n_train + n_val:]

    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    assert len(test) > 0

    return SplitResult(train=train, val=val, test=test)
