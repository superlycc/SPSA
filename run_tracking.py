"""Replay cached detections without loading ground-truth identities."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from persistent_lineage import LineageConfig, PersistentLineage
from practical_lineage_ablation import PracticalAblation


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--extension", default=".jpg")
    parser.add_argument("--control", choices=("spsa", "no_ab", "no_dummy"), default="spsa")
    args = parser.parse_args()
    if args.detections.suffix == ".npz":
        with np.load(args.detections, allow_pickle=False) as cache:
            rows = cache["detections"].copy()
            frames = int(cache["frames"])
    else:
        rows = np.loadtxt(args.detections, delimiter=",", ndmin=2)
        if args.frames is None:
            parser.error("--frames is required for CSV detections.")
        frames = args.frames
    if rows.ndim != 2 or rows.shape[1] != 6 or not np.isfinite(rows).all():
        raise ValueError("Expected rows: frame,x1,y1,x2,y2,confidence.")
    if np.any(rows[:, 0] != np.floor(rows[:, 0])) or np.any(
            (rows[:, 0] < 1) | (rows[:, 0] > frames)):
        raise ValueError("Invalid detection frame indices.")
    if np.any(rows[:, 3:5] <= rows[:, 1:3]) or np.any(
            (rows[:, 5] < 0) | (rows[:, 5] > 1)):
        raise ValueError("Invalid boxes or detection scores.")
    image = cv2.imread(str(args.images / f"000001{args.extension}"))
    if image is None:
        raise ValueError("Cannot read first frame.")
    model = PersistentLineage() if args.control == "spsa" else PracticalAblation(args.control)
    results = []
    start = time.perf_counter()
    for frame in range(1, frames + 1):
        detections = rows[rows[:, 0] == frame, 1:].copy()
        tracked = model.update(detections, image.shape[:2],
            args.images / f"{frame:06d}{args.extension}",
            args.images / f"{frame-1:06d}{args.extension}")
        if not np.isfinite(tracked).all() or len(set(tracked[:, 4])) != len(tracked):
            raise ValueError("Invalid or duplicate output identities.")
        for x1, y1, x2, y2, identity in tracked:
            if x2 > x1 and y2 > y1:
                results.append([frame, identity, x1, y1, x2-x1, y2-y1, 1, -1, -1, -1])
    elapsed = time.perf_counter() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, np.asarray(results).reshape(-1, 10), delimiter=",",
               fmt=["%d", "%d"] + ["%.6f"]*5 + ["%d"]*3)
    metadata = dict(configuration=asdict(LineageConfig()), control=args.control,
        frames=frames, seconds=elapsed, detections_sha256=digest(args.detections),
        predictions_sha256=digest(args.output), point_counters=model.counters)
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
