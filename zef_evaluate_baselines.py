"""Continuous-coordinate IoU used by point-pool overlap checks."""

import numpy as np


def box_iou(first, second):
    lower = np.maximum(first[:, None, :2], second[None, :, :2])
    upper = np.minimum(first[:, None, 2:4], second[None, :, 2:4])
    intersection = np.maximum(upper - lower, 0).prod(axis=2)
    first_area = np.maximum(first[:, 2:4] - first[:, :2], 0).prod(axis=1)
    second_area = np.maximum(second[:, 2:4] - second[:, :2], 0).prod(axis=1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
