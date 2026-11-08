"""Negative and positive controls before the third validation search."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from compositional_birth import BirthConfig, CompositionalBirth
from persistent_lineage import LineageConfig, PersistentLineage, point_claims, solve_claims


class LineageTests(unittest.TestCase):
    def test_unique_pair(self):
        claims = np.array([[[.9, .1], [.1, .9]]]*2)
        self.assertEqual(solve_claims(claims, .25, .1), (0, 1))

    def test_dummy_is_real_choice(self):
        claims = np.array([[[.9, 0.0]]]*2)
        self.assertEqual(solve_claims(claims, .25, .1), (0, -1))

    def test_shared_detection_tie_falls_back(self):
        claims = np.array([[[.9, .9]]]*2)
        self.assertIsNone(solve_claims(claims, .25, .1))

    def test_ab_conflict_falls_back(self):
        claims = np.array([[[.9, .1], [.1, .9]], [[.1, .9], [.9, .1]]])
        self.assertIsNone(solve_claims(claims, .25, .1))

    def test_empty_claims(self):
        self.assertIsNone(solve_claims(np.zeros((2, 0, 2)), .25, .1))

    def test_low_support_not_forced(self):
        self.assertEqual(solve_claims(np.zeros((2, 2, 2)), .25, .1), (-1, -1))

    def test_points_vote_once(self):
        points = np.array([[5, 5], [5, 5], [5, 5], [5, 5]], np.float32)
        claims = point_claims(points, np.ones(4), np.array([0, 0, 1, 1]),
                              np.array([[0, 0, 20, 20], [0, 0, 40, 40]]), [1], 2)
        np.testing.assert_array_equal(claims[:, :, 0], [[1, 0], [1, 0]])

    def test_independent_new_detection_has_no_claim(self):
        claims = point_claims(np.array([[5, 5]]*8, np.float32), np.ones(8),
                              np.arange(8) % 2, np.array([[100, 100, 120, 120]]), [1])
        self.assertEqual(claims.sum(), 0)

    def test_disabled_exact_birth_host(self):
        sequence = [np.empty((0, 5))] + [np.array([[20+i, 20, 50+i, 50, .9]])
                                        for i in range(12)]
        model = PersistentLineage(LineageConfig(mode="disabled"))
        actual = [model.update(b, (200, 200)) for b in sequence]
        host = CompositionalBirth(BirthConfig(mode="score", score_threshold=.85))
        expected = [host.update(b, (200, 200), None, None) for b in sequence]
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(model.counters["image_pairs"], 0)

    def test_empty_points_no_mutation_hooks_restore(self):
        model = PersistentLineage()
        associate = model.module.associate_4_points_with_score
        update = model.module.KalmanBoxTracker.update
        boxes = np.array([[20, 20, 40, 40, .95]])
        original = boxes.copy()
        with patch.object(model, "prepare"):
            model.update(boxes, (200, 200))
        np.testing.assert_array_equal(boxes, original)
        self.assertIs(model.module.associate_4_points_with_score, associate)
        self.assertIs(model.module.KalmanBoxTracker.update, update)

    def test_exception_restores_all_hooks(self):
        model = PersistentLineage()
        associate = model.module.associate_4_points_with_score
        update = model.module.KalmanBoxTracker.update
        initialize = model.module.KalmanBoxTracker.__init__
        with patch.object(model, "prepare"), patch.object(model.host, "update",
                                                         side_effect=RuntimeError("control")):
            with self.assertRaises(RuntimeError):
                model.update(np.empty((0, 5)), (200, 200))
        self.assertIs(model.module.associate_4_points_with_score, associate)
        self.assertIs(model.module.KalmanBoxTracker.update, update)
        self.assertIs(model.module.KalmanBoxTracker.__init__, initialize)

    def test_persistent_sources_not_rebound_to_current_boxes(self):
        model = PersistentLineage()
        model.frame = 10
        model.start = 9
        model.points = np.array([[10, 10], [12, 12]]*4, np.float32)
        model.labels = np.array([1, 2]*4)
        model.halves = np.array([0, 0, 1, 1]*2)
        model.host.trackers = [
            SimpleNamespace(id=i, time_since_update=0, hits=20,
                            last_observation=np.array([0, 0, 50, 50, .9]))
            for i in (1, 2)]
        a = np.ones((8, 1), np.uint8)
        image = np.zeros((100, 100), np.uint8)
        with patch.object(model, "seed") as seed, patch.object(
                model.extension, "gray", return_value=image) as gray, patch(
                "persistent_lineage.cv2.calcOpticalFlowPyrLK",
                side_effect=[(model.points.reshape(-1, 1, 2)+1, a, None),
                             (model.points.reshape(-1, 1, 2), a, None)]):
            model.prepare("frame10", "frame9")
        seed.assert_not_called()
        self.assertEqual([c.args[0] for c in gray.call_args_list], ["frame9", "frame10"])
        np.testing.assert_array_equal(model.labels, [1, 2]*4)

    def test_promoted_observation_restores_original_score(self):
        model = PersistentLineage(LineageConfig(mode="promotion"))
        for _ in range(6):
            model.update(np.array([[20, 20, 60, 60, .95]]), (200, 200))
        identity = model.host.trackers[0].id
        model.points = np.array([[20, 20]]*8, np.float32)
        model.labels = np.full(8, identity)
        model.halves = np.arange(8) % 2
        with patch.object(model, "prepare"):
            model.update(np.array([[20, 20, 60, 60, .4]]), (200, 200))
        self.assertEqual(model.counters["promotions"], 1)
        self.assertAlmostEqual(model.host.trackers[0].last_observation[4], .4)

    def test_positive_atomic_reassignment_preserves_other_track(self):
        model = PersistentLineage()
        model.host.trackers = [SimpleNamespace(id=i) for i in (1, 2, 3)]
        model.points = np.array([[5, 5]]*16 + [[25, 5]]*4 +
                                [[5, 5]]*4 + [[25, 5]]*16, np.float32)
        model.labels = np.array([1]*20 + [2]*20)
        model.halves = np.arange(40) % 2
        model.original_associate = lambda *a, **kw: (
            np.array([[0, 1], [1, 0], [2, 2]]), np.array([], int), np.array([], int))
        dets = np.array([[0, 0, 20, 20, .9], [40, 0, 60, 20, .9],
                         [100, 100, 120, 120, .9]])
        matched, ds, ts = model.associate(dets, np.zeros((3, 6)))
        self.assertEqual(set(map(tuple, matched)), {(0, 0), (1, 1), (2, 2)})
        self.assertEqual(len(ds)+len(ts), 0)
        self.assertEqual(model.counters["changed"], 1)


if __name__ == "__main__":
    unittest.main()
