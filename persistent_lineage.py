"""Keep identity-tagged points independent of disputed detector associations."""

from dataclasses import dataclass
from itertools import product

import cv2
import numpy as np

from compositional_birth import BirthConfig, CompositionalBirth
from pastfish import components
from zef_evaluate_baselines import box_iou


@dataclass
class LineageConfig:
    mode: str = "persistent"
    margin: float = .10
    overlap: float = .10
    ttl: int = 40
    points: int = 80
    minimum_group: int = 4
    dummy: float = .25
    birth_score: float = .85


def solve_claims(claims, dummy, margin):
    """Enumerate injective assignments including a private missing-observation choice."""
    _, detections, owners = claims.shape
    if not detections or not owners:
        return None
    choices = []
    for order in product(range(-1, detections), repeat=owners):
        real = [i for i in order if i >= 0]
        if len(real) != len(set(real)):
            continue
        values = np.array([[dummy if i < 0 else claims[g, i, t]
                            for t, i in enumerate(order)] for g in (0, 1)])
        choices.append((order, values.mean(axis=1)))
    group_best = []
    for g in (0, 1):
        ranked = sorted(choices, key=lambda c: c[1][g], reverse=True)
        if ranked[0][1][g] - ranked[1][1][g] < margin:
            return None
        group_best.append(ranked[0][0])
    return group_best[0] if group_best[0] == group_best[1] else None


def point_claims(points, labels, halves, boxes, owners, minimum=4):
    """Exclusive containment votes; these are support proxies, not instance masks."""
    claims = np.zeros((2, len(boxes), len(owners)))
    if not len(points) or not len(boxes):
        return claims
    scaled = boxes[:, :4] * .5
    centers = (scaled[:, :2]+scaled[:, 2:])/2
    sizes = np.maximum(scaled[:, 2:]-scaled[:, :2], 1)
    inside = ((points[:, None] >= scaled[None, :, :2]).all(axis=2) &
              (points[:, None] <= scaled[None, :, 2:]).all(axis=2))
    distance = (((points[:, None]-centers[None])/sizes[None])**2).sum(axis=2)
    distance[~inside] = np.inf
    assigned = distance.argmin(axis=1)
    assigned[~inside.any(axis=1)] = -1
    for t, identity in enumerate(owners):
        for g in (0, 1):
            mask = (labels == identity) & (halves == g)
            if mask.sum() >= minimum:
                for d in range(len(boxes)):
                    claims[g, d, t] = np.mean(assigned[mask] == d)
    return claims


class PersistentLineage:
    def __init__(self, config=None):
        self.config = config or LineageConfig()
        self.extension = CompositionalBirth(BirthConfig(
            mode="score", score_threshold=self.config.birth_score))
        self.host = self.extension.host
        self.module = self.extension.module
        self.original_associate = self.module.associate_4_points_with_score
        self.frame = 0
        self.points = np.empty((0, 2), dtype=np.float32)
        self.labels = np.empty(0, dtype=int)
        self.halves = np.empty(0, dtype=int)
        self.start = 0
        self.events = []
        self.counters = dict(image_pairs=0, seeded=0, transported=0, promotions=0,
                             components=0, stable=0, changed=0, expired=0)

    def seed(self, gray, tracks):
        points, labels, halves = [], [], []
        boxes = np.array([t.last_observation[:4] for t in tracks]) * .5
        h, w = gray.shape
        for t, box in zip(tracks, boxes):
            x1, y1, x2, y2 = np.rint(box).astype(int)
            x1, x2 = np.clip([x1, x2], 0, w)
            y1, y2 = np.clip([y1, y2], 0, h)
            if x2 <= x1 or y2 <= y1:
                continue
            mask = np.zeros_like(gray, dtype=np.uint8)
            roi = gray[y1:y2, x1:x2]
            mask[y1:y2, x1:x2] = (roi < np.median(roi)-6)*255
            for other in tracks:
                if other.id == t.id:
                    continue
                a, b, c, d = np.rint(other.last_observation[:4]*.5).astype(int)
                a, c = np.clip([a, c], 0, w)
                b, d = np.clip([b, d], 0, h)
                mask[b:d, a:c] = 0
            found = cv2.goodFeaturesToTrack(gray, self.config.points, .01, 3, mask=mask)
            if found is None:
                continue
            found = found.reshape(-1, 2)
            points.extend(found)
            labels.extend([t.id]*len(found))
            halves.extend(np.arange(len(found)) % 2)
        self.points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        self.labels = np.asarray(labels, dtype=int)
        self.halves = np.asarray(halves, dtype=int)
        self.start = self.frame
        self.counters["seeded"] += len(points)

    def prepare(self, current_image, previous_image):
        alive = {t.id for t in self.host.trackers}
        keep = np.isin(self.labels, list(alive))
        self.points, self.labels, self.halves = self.points[keep], self.labels[keep], self.halves[keep]
        if self.frame-self.start >= self.config.ttl:
            self.points = self.points[:0]
            self.labels = self.labels[:0]
            self.halves = self.halves[:0]
            self.counters["expired"] += 1
        tracks = [t for t in self.host.trackers if t.time_since_update == 0
                  and t.hits >= 5 and t.last_observation[4] > .5]
        crowd = []
        if len(tracks) >= 2:
            overlaps = box_iou(np.array([t.last_observation[:4] for t in tracks]),
                               np.array([t.last_observation[:4] for t in tracks]))
            np.fill_diagonal(overlaps, 0)
            crowd = [t for i, t in enumerate(tracks) if overlaps[i].max() > self.config.overlap]
        reseed = len(crowd) >= 2 and (not len(self.points) or self.config.mode == "rolling")
        if not reseed and not len(self.points):
            return
        previous = self.extension.gray(previous_image)
        current = self.extension.gray(current_image)
        self.counters["image_pairs"] += 1
        if reseed:
            self.seed(previous, crowd)
        if not len(self.points):
            return
        source = self.points.reshape(-1, 1, 2)
        target, a, _ = cv2.calcOpticalFlowPyrLK(previous, current, source, None,
                                             winSize=(21, 21), maxLevel=3)
        if target is None:
            self.points, self.labels, self.halves = self.points[:0], self.labels[:0], self.halves[:0]
            return
        returned, b, _ = cv2.calcOpticalFlowPyrLK(current, previous, target, None,
                                               winSize=(21, 21), maxLevel=3)
        if returned is None:
            self.points, self.labels, self.halves = self.points[:0], self.labels[:0], self.halves[:0]
            return
        moved = target.reshape(-1, 2)
        valid = ((a.ravel() == 1) & (b.ravel() == 1) &
                 (np.linalg.norm((returned-source).reshape(-1, 2), axis=1) < 1.5) &
                 (moved >= 0).all(axis=1) &
                 (moved < np.array(current.shape[::-1])).all(axis=1))
        self.points, self.labels, self.halves = moved[valid], self.labels[valid], self.halves[valid]
        self.counters["transported"] += int(valid.sum())

    def associate(self, dets, trks, *args, **kwargs):
        matched, unmatched_dets, unmatched_trks = self.original_associate(dets, trks, *args, **kwargs)
        if self.config.mode == "promotion" or not len(self.points):
            return matched, unmatched_dets, unmatched_trks
        owners = [t.id for t in self.host.trackers]
        claims = point_claims(self.points, self.labels, self.halves, dets, owners,
                              self.config.minimum_group)
        allowed = claims.mean(axis=0) > .05
        cloud = set(np.where(allowed.any(axis=0))[0])
        # A detector already assigned outside this evidence pool cannot be stolen.
        for d, t in matched:
            if t not in cloud:
                allowed[d] = False
        pairs = {int(t): int(d) for d, t in matched}
        for ds, ts in components(allowed):
            if not (2 <= len(ts) <= 4 and 1 <= len(ds) <= 4):
                continue
            if any(t in pairs and pairs[t] not in ds for t in ts):
                continue
            self.counters["components"] += 1
            local = claims[:, ds][:, :, ts]
            order = solve_claims(local, self.config.dummy, self.config.margin)
            event = dict(frame=self.frame, owners=[owners[t]+1 for t in ts],
                         boxes=dets[ds].tolist(), claims=local.tolist(),
                         point_counts={str(owners[t]+1): int((self.labels == owners[t]).sum()) for t in ts},
                         stable=order is not None, changed=False)
            if order is not None:
                self.counters["stable"] += 1
                proposal = {t: int(ds[d]) for t, d in zip(ts, order) if d >= 0}
                original = {t: pairs[t] for t in ts if t in pairs}
                if proposal != original:
                    self.counters["changed"] += 1
                    event["changed"] = True
                for t in ts:
                    pairs.pop(t, None)
                pairs.update(proposal)
                event["before"] = original
                event["after"] = proposal
            self.events.append(event)
        result = np.array([(d, t) for t, d in pairs.items()], dtype=int).reshape(-1, 2)
        if len(set(result[:, 0])) != len(result):
            raise ValueError("Non-injective lineage association")
        return (result, np.setdiff1d(np.arange(len(dets)), result[:, 0]),
                np.setdiff1d(np.arange(len(trks)), result[:, 1]))

    def update(self, detections, info, current_image=None, previous_image=None):
        self.frame += 1
        if self.config.mode == "disabled":
            return self.extension.update(detections, info, current_image, previous_image)
        self.prepare(current_image, previous_image)
        boxes = detections.copy()
        scores = {tuple(b[:4]): float(b[4]) for b in boxes}
        owners = [t.id for t in self.host.trackers]
        claims = point_claims(self.points, self.labels, self.halves, boxes, owners,
                              self.config.minimum_group)
        if len(boxes) and len(owners):
            promote = ((boxes[:, 4] > .1) & (boxes[:, 4] < .5) &
                       (claims.mean(axis=0).max(axis=1) > self.config.dummy))
            boxes[promote, 4] = .50001
            self.counters["promotions"] += int(promote.sum())
        original_update = self.module.KalmanBoxTracker.update
        original_associate = self.module.associate_4_points_with_score

        def restore_score(track, bbox):
            if bbox is not None:
                bbox = bbox.copy()
                bbox[4] = scores[tuple(bbox[:4])]
            return original_update(track, bbox)

        self.module.KalmanBoxTracker.update = restore_score
        self.module.associate_4_points_with_score = self.associate
        try:
            return self.extension.update(boxes, info, current_image, previous_image)
        finally:
            self.module.KalmanBoxTracker.update = original_update
            self.module.associate_4_points_with_score = original_associate
