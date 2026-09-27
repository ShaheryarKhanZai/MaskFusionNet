#!/usr/bin/env python3
"""Sanity-check a UBFC-rPPG directory and print the subject-level split
that training will use, WITHOUT copying/re-encoding any video (UBFC's
raw layout is already what `UBFCrPPGDataset` reads directly).

Usage:
    python scripts/prepare_ubfc.py --root /path/to/UBFC-rPPG
    # or:
    export MASKFUSIONNET_UBFC_ROOT=/path/to/UBFC-rPPG
    python scripts/prepare_ubfc.py
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maskfusionnet.data import discover_subjects, subject_level_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=os.environ.get("MASKFUSIONNET_UBFC_ROOT"))
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.root:
        raise SystemExit(
            "No dataset root given. Pass --root /path/to/UBFC-rPPG or set "
            "MASKFUSIONNET_UBFC_ROOT."
        )
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Path does not exist: {root}")

    subjects = discover_subjects(root)
    print(f"Found {len(subjects)} subject(s) with both vid.avi and ground_truth.txt under {root}")
    if len(subjects) < 3:
        print("WARNING: fewer than 3 subjects found -- cannot form a train/val/test split. "
              "Check that `root` points at the UBFC-rPPG directory whose immediate "
              "subdirectories are per-subject folders.")
        return

    split = subject_level_split(subjects, args.train_frac, args.val_frac, args.seed)
    print(f"train ({len(split.train)}): {split.train}")
    print(f"val   ({len(split.val)}): {split.val}")
    print(f"test  ({len(split.test)}): {split.test}")
    assert set(split.train).isdisjoint(split.val) and set(split.train).isdisjoint(split.test) \
        and set(split.val).isdisjoint(split.test)
    print("Split is subject-disjoint (no leakage). ")


if __name__ == "__main__":
    main()
