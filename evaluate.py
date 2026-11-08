"""Evaluate MOT-format whole-object boxes with pinned TrackEval metrics."""

import argparse
import csv
import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np

from run_tracking import digest
from zef_evaluate_baselines import box_iou


def load_rows(path, columns):
    if path.stat().st_size == 0:
        return np.empty((0, columns))
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    if rows.shape[1] < columns or not np.isfinite(rows).all():
        raise ValueError(f"Invalid rows: {path}")
    return rows


def evaluation_data(gt_path, prediction, frames):
    ground_truth = load_rows(gt_path, 7)
    ground_truth = ground_truth[(ground_truth[:, 6] != 0) &
                                (ground_truth[:, 4] > 0) & (ground_truth[:, 5] > 0)]
    tracks = load_rows(prediction, 10)
    for rows in (ground_truth, tracks):
        if len(rows) and (np.any(rows[:, 0] < 1) or np.any(rows[:, 0] > frames)
                          or np.any(rows[:, :2] != np.floor(rows[:, :2]))):
            raise ValueError("Invalid frame or identity indices")
        if len({tuple(row[:2]) for row in rows}) != len(rows):
            raise ValueError("Duplicate frame/identity rows")
    gt_mapping = {v: i for i, v in enumerate(np.unique(ground_truth[:, 1]))}
    tr_mapping = {v: i for i, v in enumerate(np.unique(tracks[:, 1]))}
    data = dict(num_timesteps=frames, num_gt_ids=len(gt_mapping),
        num_tracker_ids=len(tr_mapping), num_gt_dets=len(ground_truth),
        num_tracker_dets=len(tracks), gt_ids=[], tracker_ids=[], similarity_scores=[])
    for frame in range(1, frames + 1):
        gt = ground_truth[ground_truth[:, 0] == frame]
        tr = tracks[tracks[:, 0] == frame]
        data["gt_ids"].append(np.array([gt_mapping[v] for v in gt[:, 1]], dtype=int))
        data["tracker_ids"].append(np.array([tr_mapping[v] for v in tr[:, 1]], dtype=int))
        gt_boxes, tr_boxes = gt[:, 2:6].copy(), tr[:, 2:6].copy()
        gt_boxes[:, 2:] += gt_boxes[:, :2]
        tr_boxes[:, 2:] += tr_boxes[:, :2]
        data["similarity_scores"].append(box_iou(gt_boxes, tr_boxes))
    return data


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def summarize(results):
    hota, identity, clear = results["HOTA"], results["Identity"], results["CLEAR"]
    return dict(HOTA=float(np.mean(hota["HOTA"])*100),
        DetA=float(np.mean(hota["DetA"])*100), AssA=float(np.mean(hota["AssA"])*100),
        IDF1=float(identity["IDF1"]*100), IDSW=int(clear["IDSW"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--lengths", type=Path, required=True)
    parser.add_argument("--trackeval", type=Path, default=Path("external/TrackEval"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # The pinned upstream metrics use NumPy aliases removed in NumPy 1.24.
    for alias, value in (("float", float), ("int", int)):
        if alias not in np.__dict__:
            setattr(np, alias, value)
    namespace = types.ModuleType("trackeval")
    namespace.__path__ = [str(args.trackeval.resolve() / "trackeval")]
    sys.modules["trackeval"] = namespace
    classes = [importlib.import_module(f"trackeval.metrics.{name}").__dict__[label]
               for name, label in (("hota", "HOTA"), ("clear", "CLEAR"), ("identity", "Identity"))]
    metrics = [cls({"PRINT_CONFIG": False}) for cls in classes]
    lengths = json.loads(args.lengths.read_text())
    raw, hashes = {}, {}
    for sequence, frames in lengths.items():
        gt = args.gt_root / sequence / "gt/gt.txt"
        pred = args.predictions / f"{sequence}.txt"
        data = evaluation_data(gt, pred, int(frames))
        raw[sequence] = {m.get_name(): m.eval_sequence(data) for m in metrics}
        hashes[sequence] = dict(ground_truth=digest(gt), predictions=digest(pred))
    combined = {m.get_name(): m.combine_sequences(
        {seq: result[m.get_name()] for seq, result in raw.items()}) for m in metrics}
    rows = [dict(sequence=seq, **summarize(result)) for seq, result in raw.items()]
    rows.append(dict(sequence="COMBINED", **summarize(combined)))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "metrics.json").write_text(
        json.dumps(jsonable(dict(sequences=raw, combined=combined, hashes=hashes)), indent=2) + "\n")
    print(rows[-1])


if __name__ == "__main__":
    main()
