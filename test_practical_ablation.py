"""Check exactly which decision protections the two controls remove."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import persistent_lineage as frozen
from practical_lineage_ablation import PracticalAblation, solve_no_ab, solve_no_dummy


class PracticalTests(unittest.TestCase):
    def test_no_ab_agreement_matches_original(self):
        c = np.array([[[.9, .1], [.1, .9]]] * 2)
        self.assertEqual(solve_no_ab(c, .25, .10), frozen.solve_claims(c, .25, .10))

    def test_no_ab_can_accept_conflicting_groups(self):
        c = np.array([[[1., 0.], [0., 1.]], [[.3, .7], [.7, .3]]])
        self.assertIsNone(frozen.solve_claims(c, .25, .10))
        self.assertEqual(solve_no_ab(c, .25, .10), (0, 1))

    def test_no_ab_keeps_pooled_margin(self):
        self.assertIsNone(solve_no_ab(np.full((2, 2, 2), .5), .25, .10))

    def test_no_ab_keeps_dummy(self):
        self.assertEqual(solve_no_ab(np.array([[[.9, 0.]]] * 2), .25, .10), (0, -1))

    def test_no_dummy_falls_back_when_real_boxes_insufficient(self):
        c = np.array([[[.9, 0.]]] * 2)
        self.assertEqual(frozen.solve_claims(c, .25, .10), (0, -1))
        self.assertIsNone(solve_no_dummy(c, .25, .10))

    def test_no_dummy_keeps_group_agreement(self):
        c = np.array([[[.9, .1], [.1, .9]], [[.1, .9], [.9, .1]]])
        self.assertIsNone(solve_no_dummy(c, .25, .10))

    def test_no_dummy_keeps_group_margin(self):
        self.assertIsNone(solve_no_dummy(np.full((2, 2, 2), .5), .25, .10))

    def test_no_dummy_assignment_is_injective(self):
        c = np.array([[[.9, .1], [.1, .9]]] * 2)
        self.assertEqual(solve_no_dummy(c, .25, .10), (0, 1))

    def test_empty_candidates(self):
        for solver in (solve_no_ab, solve_no_dummy):
            self.assertIsNone(solver(np.zeros((2, 0, 2)), .25, .10))

    def test_control_only_changes_solver_and_restores_hook(self):
        original = frozen.solve_claims
        for mode in ("no_ab", "no_dummy"):
            model = PracticalAblation(mode)
            self.assertEqual(model.config, frozen.LineageConfig())
            model.host.trackers = [SimpleNamespace(id=1), SimpleNamespace(id=2)]
            model.points = np.array([[5, 5]] * 16 + [[25, 5]] * 4 +
                                    [[5, 5]] * 4 + [[25, 5]] * 16, np.float32)
            model.labels = np.array([1] * 20 + [2] * 20)
            model.halves = np.arange(40) % 2
            model.original_associate = lambda *a, **kw: (
                np.array([[0, 1], [1, 0]]), np.array([], int), np.array([], int))
            result, _, _ = model.associate(
                np.array([[0, 0, 20, 20, .9], [40, 0, 60, 20, .9]]),
                np.zeros((2, 6)))
            self.assertEqual(set(map(tuple, result)), {(0, 0), (1, 1)})
            self.assertIs(frozen.solve_claims, original)

    def test_exception_restores_solver(self):
        original = frozen.solve_claims
        model = PracticalAblation("no_ab")
        with patch.object(frozen.PersistentLineage, "associate", side_effect=RuntimeError):
            with self.assertRaises(RuntimeError):
                model.associate(None, None)
        self.assertIs(frozen.solve_claims, original)


if __name__ == "__main__":
    unittest.main()
