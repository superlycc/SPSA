"""Decision-only controls without modifying the frozen third-round source."""

from itertools import product
from unittest.mock import patch

import numpy as np

import persistent_lineage as frozen


def solve_no_ab(claims, dummy, margin):
    pooled = np.repeat(claims.mean(axis=0, keepdims=True), 2, axis=0)
    return frozen.solve_claims(pooled, dummy, margin)


def solve_no_dummy(claims, dummy, margin):
    _, detections, owners = claims.shape
    if not owners or detections < owners:
        return None
    choices = []
    for order in product(range(detections), repeat=owners):
        if len(set(order)) != owners:
            continue
        values = np.array([[claims[g, d, t] for t, d in enumerate(order)]
                           for g in (0, 1)])
        choices.append((order, values.mean(axis=1)))
    # A sole feasible assignment has no competing real assignment.
    if len(choices) == 1:
        return choices[0][0]
    best = []
    for g in (0, 1):
        ranked = sorted(choices, key=lambda c: c[1][g], reverse=True)
        if ranked[0][1][g] - ranked[1][1][g] < margin:
            return None
        best.append(ranked[0][0])
    return best[0] if best[0] == best[1] else None


class PracticalAblation(frozen.PersistentLineage):
    def __init__(self, control):
        if control not in ("no_ab", "no_dummy"):
            raise ValueError("Unknown decision control")
        super().__init__(frozen.LineageConfig())
        self.control = control

    def associate(self, *args, **kwargs):
        solver = solve_no_ab if self.control == "no_ab" else solve_no_dummy
        original = frozen.solve_claims
        if self.control == "no_ab":
            # Capture the frozen solver before temporarily replacing its global name.
            def solver(claims, dummy, margin):
                pooled = np.repeat(claims.mean(axis=0, keepdims=True), 2, axis=0)
                return original(pooled, dummy, margin)
        with patch.object(frozen, "solve_claims", solver):
            return super().associate(*args, **kwargs)
