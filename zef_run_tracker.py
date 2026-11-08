"""Load the pinned Hybrid-SORT host without detector or training dependencies."""

import importlib
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import numpy as np


def scalar_prediction_compatibility(module):
    tracker_class = module.KalmanBoxTracker
    if getattr(tracker_class, "_zef_scalar_compatible", False):
        return
    original = tracker_class.predict

    def predict(self):
        boxes, kalman_score, simple_score = original(self)
        return (boxes, float(np.asarray(kalman_score).item()),
                float(np.asarray(simple_score).item()))

    tracker_class.predict = predict
    tracker_class._zef_scalar_compatible = True


def build_tracker(name):
    if name != "HybridSORT":
        raise ValueError("This release uses the Hybrid-SORT host.")
    for alias, value in (("float", float), ("int", int)):
        if alias not in np.__dict__:
            setattr(np, alias, value)
    source = Path(os.environ.get(
        "SPSA_HYBRIDSORT", Path(__file__).parent / "external/HybridSORT")).resolve()
    if not (source / "trackers/hybrid_sort_tracker/hybrid_sort.py").is_file():
        raise FileNotFoundError("Install the pinned HybridSORT source; see README.md.")
    namespace = types.ModuleType("trackers")
    namespace.__path__ = [str(source / "trackers")]
    sys.modules["trackers"] = namespace
    module = importlib.import_module("trackers.hybrid_sort_tracker.hybrid_sort")
    scalar_prediction_compatibility(module)
    common = dict(track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20=False,
                  TCM_first_step=True, TCM_byte_step=True, TCM_first_step_weight=1.0,
                  TCM_byte_step_weight=1.0)
    args = SimpleNamespace(**common)
    host = module.Hybrid_Sort(args, det_thresh=0.5, max_age=30, min_hits=3,
                             iou_threshold=0.3, delta_t=3, inertia=0.2, use_byte=True)
    return host, dict(**common, iou_threshold=0.3, inertia=0.2, use_byte=True)
