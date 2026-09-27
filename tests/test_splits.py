import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maskfusionnet.data.splits import subject_level_split


def test_split_is_disjoint_and_covers_all_subjects():
    subjects = [f"subject{i}" for i in range(1, 21)]
    split = subject_level_split(subjects, train_frac=0.7, val_frac=0.15, seed=42)
    all_returned = split.train + split.val + split.test
    assert sorted(all_returned) == sorted(subjects)
    assert set(split.train).isdisjoint(split.val)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.val).isdisjoint(split.test)
    assert len(split.test) > 0


def test_split_is_deterministic_given_seed():
    subjects = [f"s{i}" for i in range(15)]
    a = subject_level_split(subjects, seed=7)
    b = subject_level_split(subjects, seed=7)
    assert a.as_dict() == b.as_dict()


def test_split_changes_with_different_seed():
    subjects = [f"s{i}" for i in range(15)]
    a = subject_level_split(subjects, seed=1)
    b = subject_level_split(subjects, seed=2)
    assert a.as_dict() != b.as_dict()


def test_split_raises_with_too_few_subjects():
    with pytest.raises(ValueError):
        subject_level_split(["only_one"], seed=0)
