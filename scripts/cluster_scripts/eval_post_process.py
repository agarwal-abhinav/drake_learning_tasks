#!/usr/bin/env python3
"""
Analyze intermediate_snapshot.yaml stats across checkpoints for initial conditions 0 and 4.

Directory layout expected:

<root_dir>/
  0/
    <checkpoint_name_1>/intermediate_snapshot.yaml
    <checkpoint_name_2>/intermediate_snapshot.yaml
    ...
  4/
    <checkpoint_name_1>/intermediate_snapshot.yaml
    <checkpoint_name_2>/intermediate_snapshot.yaml
    ...

Each intermediate_snapshot.yaml should contain keys:
  total_mild_return_to_box, total_return_to_box, total_success, total_mild_success,
  total_final_area, total_mid_area, num_final_area, num_mid_area, seeds_completed

What it prints:
  1) Best checkpoints by metric when combining 0+4 (sums), with breakdown per condition.
     - For best total_mild_return_to_box: also report total_return_to_box for that checkpoint.
     - For best total_mild_success: also report total_success for that checkpoint.
     - For each best checkpoint, also report combined avg mid/final areas:
         avg_mid_area   = (total_mid_area_0 + total_mid_area_4) / (num_mid_area_0 + num_mid_area_4)
         avg_final_area = (total_final_area_0 + total_final_area_4) / (num_final_area_0 + num_final_area_4)

  2) The same “best checkpoint by metric” report, but separately for 0 only and 4 only.

Run:
  python analyze_snapshots.py /path/to/root_dir
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


REQUIRED_KEYS = [
    "total_mild_return_to_box",
    "total_return_to_box",
    "total_success",
    "total_mild_success",
    "total_final_area",
    "total_mid_area",
    "num_final_area",
    "num_mid_area",
]


PRIMARY_METRICS = [
    "total_mild_return_to_box",
    "total_return_to_box",
    "total_success",
    "total_mild_success",
]

PAIRED_METRICS = {
    "total_mild_return_to_box": ["total_mild_success"],
    "total_mild_success": ["total_mild_return_to_box"],
    "total_return_to_box": ["total_success"],
    "total_success": ["total_return_to_box"],
}



@dataclass(frozen=True)
class SnapshotStats:
    total_mild_return_to_box: float
    total_return_to_box: float
    total_success: float
    total_mild_success: float
    total_final_area: float
    total_mid_area: float
    num_final_area: float
    num_mid_area: float

    @staticmethod
    def from_yaml(path: Path) -> "SnapshotStats":
        with path.open("r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"{path}: YAML root is not a mapping/dict")

        # Ensure required numeric keys exist
        missing = [k for k in REQUIRED_KEYS if k not in data]
        if missing:
            raise KeyError(f"{path}: missing keys: {missing}")

        # Validate seeds_completed list length = 100
        if "seeds_completed" not in data:
            raise KeyError(f"{path}: missing key: 'seeds_completed'")

        seeds_completed = data["seeds_completed"]
        if not isinstance(seeds_completed, (list, tuple)):
            raise ValueError(f"{path}: seeds_completed must be a list/tuple, got {type(seeds_completed)}")

        if len(seeds_completed) != 100:
            raise ValueError(f"{path}: expected seeds_completed length 100, got {len(seeds_completed)}")

        def as_float(k: str) -> float:
            v = data[k]
            try:
                return float(v)
            except Exception as e:
                raise ValueError(f"{path}: key '{k}' value '{v}' not convertible to float") from e

        return SnapshotStats(
            total_mild_return_to_box=as_float("total_mild_return_to_box"),
            total_return_to_box=as_float("total_return_to_box"),
            total_success=as_float("total_success"),
            total_mild_success=as_float("total_mild_success"),
            total_final_area=as_float("total_final_area"),
            total_mid_area=as_float("total_mid_area"),
            num_final_area=as_float("num_final_area"),
            num_mid_area=as_float("num_mid_area"),
        )


    def get(self, key: str) -> float:
        return getattr(self, key)


def list_checkpoints(cond_dir: Path) -> List[str]:
    if not cond_dir.exists():
        return []
    return sorted([
        p.name for p in cond_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ])



def load_all_stats(root: Path, cond: str) -> Dict[str, SnapshotStats]:
    cond_dir = root / cond
    stats: Dict[str, SnapshotStats] = {}

    for ckpt in list_checkpoints(cond_dir):
        snap = cond_dir / ckpt / "intermediate_snapshot.yaml"
        if not snap.exists():
            print(f"[WARN] Missing snapshot: {snap}", file=sys.stderr)
            continue
        # try:
        #     stats[ckpt] = SnapshotStats.from_yaml(snap)
        # except Exception as e:
        #     print(f"[WARN] Failed to read {snap}: {e}", file=sys.stderr)
        
        stats[ckpt] = SnapshotStats.from_yaml(snap)

    return stats


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return float("nan")
    return num / den


def combined_avg_areas(s0: SnapshotStats, s4: SnapshotStats) -> Tuple[float, float]:
    total_mid = s0.total_mid_area + s4.total_mid_area
    total_final = s0.total_final_area + s4.total_final_area
    num_mid = s0.num_mid_area + s4.num_mid_area
    num_final = s0.num_final_area + s4.num_final_area
    return safe_div(total_mid, num_mid), safe_div(total_final, num_final)


def format_float(x: float) -> str:
    if x is None:
        return "None"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return str(x)
    # Keep it readable; tweak as needed.
    return f"{x:.6g}"


def pick_best_checkpoint(
    ckpts: List[str],
    stats_map: Dict[str, SnapshotStats],
    metric: str,
) -> Optional[str]:
    best_ckpt = None
    best_val = None
    for c in ckpts:
        if c not in stats_map:
            continue
        v = stats_map[c].get(metric)
        if best_val is None or v > best_val:
            best_val = v
            best_ckpt = c
    return best_ckpt


def print_best_combined(
    ckpts: List[str],
    stats0: Dict[str, SnapshotStats],
    stats4: Dict[str, SnapshotStats],
) -> None:
    print("\n==============================")
    print("BEST CHECKPOINTS (COMBINED 0+4)")
    print("==============================")

    usable = [c for c in ckpts if c in stats0 and c in stats4]
    if not usable:
        print("No common checkpoints with readable snapshots in both 0 and 4.")
        return

    # Precompute combined sums for each checkpoint
    combined: Dict[str, Dict[str, float]] = {}
    for c in usable:
        s0, s4 = stats0[c], stats4[c]
        combined[c] = {k: s0.get(k) + s4.get(k) for k in REQUIRED_KEYS}

    def best_by(metric: str) -> str:
        return max(usable, key=lambda c: combined[c][metric])

    best_ckpts = {m: best_by(m) for m in PRIMARY_METRICS}

    for metric in PRIMARY_METRICS:
        c = best_ckpts[metric]
        s0, s4 = stats0[c], stats4[c]
        total = combined[c][metric]

        print(f"\n--- Best by {metric} ---")
        print(f"checkpoint: {c}")
        print(f"{metric} (combined): {format_float(total)}")
        print(f"  breakdown: 0={format_float(s0.get(metric))}  4={format_float(s4.get(metric))}")

        # Also report paired metrics for this best checkpoint
        for other in PAIRED_METRICS.get(metric, []):
            other_total = s0.get(other) + s4.get(other)
            print(f"{other} at this checkpoint (combined): {format_float(other_total)}")
            print(f"  breakdown: 0={format_float(s0.get(other))}  4={format_float(s4.get(other))}")

        print(f"checkpoint: latest")
        print(f"{metric} (combined): {format_float(combined['latest'][metric])}")
        print(f"  breakdown: 0={format_float(stats0['latest'].get(metric))}  4={format_float(stats4['latest'].get(metric))}")

        # Also report paired metrics for this best checkpoint
        for other in PAIRED_METRICS.get(metric, []):
            other_total = stats0['latest'].get(other) + stats4['latest'].get(other)
            print(f"{other} at this checkpoint (combined): {format_float(other_total)}")
            print(f"  breakdown: 0={format_float(stats0['latest'].get(other))}  4={format_float(stats4['latest'].get(other))}")

        avg_mid, avg_final = combined_avg_areas(s0, s4)
        print(f"avg_mid_area (combined): {format_float(avg_mid)}")
        print(f"avg_final_area (combined): {format_float(avg_final)}")


def print_best_individual(
    cond: str,
    ckpts: List[str],
    stats: Dict[str, SnapshotStats],
) -> None:
    print("\n==============================")
    print(f"BEST CHECKPOINTS (CONDITION {cond} ONLY)")
    print("==============================")

    usable = [c for c in ckpts if c in stats]
    if not usable:
        print(f"No readable checkpoints found for condition {cond}.")
        return

    def best_by(metric: str) -> str:
        return max(usable, key=lambda c: stats[c].get(metric))

    for metric in PRIMARY_METRICS:
        c = best_by(metric)
        s = stats[c]

        print(f"\n--- Best by {metric} ---")
        print(f"checkpoint: {c}")
        print(f"{metric}: {format_float(s.get(metric))}")

        # Also report paired metrics for this best checkpoint
        for other in PAIRED_METRICS.get(metric, []):
            print(f"{other} at this checkpoint: {format_float(s.get(other))}")


        avg_mid = safe_div(s.total_mid_area, s.num_mid_area)
        avg_final = safe_div(s.total_final_area, s.num_final_area)
        print(f"avg_mid_area: {format_float(avg_mid)}")
        print(f"avg_final_area: {format_float(avg_final)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find best checkpoints across conditions 0 and 4 from intermediate_snapshot.yaml files."
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Directory containing subdirectories '0' and '4'.",
    )
    args = parser.parse_args()

    root = Path(args.root_dir).expanduser().resolve()
    d0 = root / "0"
    d4 = root / "4"

    if not root.exists():
        print(f"[ERROR] Root directory does not exist: {root}", file=sys.stderr)
        return 2
    if not d0.exists() or not d4.exists():
        print(f"[ERROR] Expected directories {d0} and {d4} to exist.", file=sys.stderr)
        return 2

    stats0 = load_all_stats(root, "0")
    stats4 = load_all_stats(root, "4")

    ckpts0 = set(stats0.keys())
    ckpts4 = set(stats4.keys())
    common = sorted(ckpts0 & ckpts4)
    only0 = sorted(ckpts0 - ckpts4)
    only4 = sorted(ckpts4 - ckpts0)

    if only0 or only4:
        print("[WARN] Checkpoint sets differ between 0 and 4.", file=sys.stderr)
        if only0:
            print(f"[WARN] Present only in 0: {only0}", file=sys.stderr)
        if only4:
            print(f"[WARN] Present only in 4: {only4}", file=sys.stderr)
        print("[WARN] Proceeding with intersection for combined analysis.", file=sys.stderr)

    # Combined report uses intersection, as you requested they should match.
    print_best_combined(common, stats0, stats4)

    # Individual reports use each condition's own list (but we pass union sorted for stable ordering)
    union_ckpts = sorted(ckpts0 | ckpts4)
    print_best_individual("0", union_ckpts, stats0)
    print_best_individual("4", union_ckpts, stats4)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
