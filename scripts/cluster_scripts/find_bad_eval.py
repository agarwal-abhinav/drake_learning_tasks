#!/usr/bin/env python3
"""
Print paths of intermediate_snapshot.yaml files where
seeds_completed is missing, not a list, or length != 100.

Usage:
  python find_bad_seeds.py /path/to/root_dir
"""

from pathlib import Path
import sys
import yaml


def check_snapshot(path: Path) -> bool:
    """
    Returns True if seeds_completed is invalid.
    """
    try:
        with path.open("r") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Could not read {path}: {e}", file=sys.stderr)
        return True

    if not isinstance(data, dict):
        return True

    if "seeds_completed" not in data:
        return True

    seeds = data["seeds_completed"]

    if not isinstance(seeds, (list, tuple)):
        return True

    return len(seeds) != 100


def main(root: Path) -> None:
    if not root.exists():
        print(f"[ERROR] Path does not exist: {root}", file=sys.stderr)
        sys.exit(2)

    bad = False

    for path in root.rglob("intermediate_snapshot.yaml"):
        if check_snapshot(path):
            print(path)
            bad = True

    if not bad:
        print("All snapshots have seeds_completed length == 100")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python find_bad_seeds.py /path/to/root_dir", file=sys.stderr)
        sys.exit(1)

    main(Path(sys.argv[1]).expanduser().resolve())
