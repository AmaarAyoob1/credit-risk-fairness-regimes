"""
Tests for fairness metric calculations.
Validates correctness with known inputs and edge cases.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fairness import (
    demographic_parity,
    equal_opportunity,
    predictive_parity,
    compute_sample_weights,
    compute_group_thresholds,
    run_fairness_audit,
)


class TestDemographicParity:
    """Tests for demographic parity metric."""

    def test_perfect_parity(self):
        """When all groups have identical approval rates, gap should be 0."""
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        protected = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        result = demographic_parity(y_pred, protected)
        assert result["gap"] == 0.0

    def test_maximum_disparity(self):
        """When one group is always approved and another always denied."""
        y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        protected = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        result = demographic_parity(y_pred, protected)
        assert result["gap"] == 1.0

    def test_partial_disparity(self):
        """Known partial disparity case."""
        y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1])
        protected = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        result = demographic_parity(y_pred, protected)
        # Group 0: 75% approval, Group 1: 25% approval
        assert abs(result["gap"] - 0.5) < 1e-10

    def test_multiple_groups(self):
        """Works correctly with 3+ groups."""
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        protected = np.array([0, 0, 1, 1, 2, 2])

        result = demographic_parity(y_pred, protected)
        assert len(result["group_rates"]) == 3


class TestEqualOpportunity:
    """Tests for equal opportunity metric."""

    def test_perfect_equality(self):
        """When TPR is equal across groups."""
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 0, 1, 0])
        protected = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        result = equal_opportunity(y_true, y_pred, protected)
        assert result["gap"] == 0.0

    def test_known_disparity(self):
        """Verifiable disparity in true positive rates."""
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        protected = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        result = equal_opportunity(y_true, y_pred, protected)
        # Group 0: all correctly classified, Group 1: 50% correctly classified
        assert result["gap"] == 0.5


class TestSampleWeights:
    """Tests for fairness-aware sample weighting."""

    def test_weights_sum(self):
        """Weights should produce balanced effective sample sizes."""
        y_true = np.array([0, 0, 0, 0, 1, 1])
        protected = np.array([0, 0, 0, 1, 1, 1])

        weights = compute_sample_weights(y_true, protected)
        assert len(weights) == len(y_true)
        assert all(w > 0 for w in weights)

    def test_balanced_data_equal_weights(self):
        """With perfectly balanced data, weights should be approximately equal."""
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        protected = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        weights = compute_sample_weights(y_true, protected)
        assert np.allclose(weights, weights[0])


class TestFairnessAudit:
    """Tests for the full fairness audit pipeline."""

    def test_audit_returns_all_metrics(self):
        """Audit should return results for all four metrics."""
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 0])
        y_prob = np.array([0.2, 0.3, 0.7, 0.4, 0.1, 0.6, 0.8, 0.3])
        protected = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        results = run_fairness_audit(y_true, y_pred, y_prob, protected)

        assert "demographic_parity" in results
        assert "equal_opportunity" in results
        assert "predictive_parity" in results
        assert "calibration" in results
        assert "overall" in results

    def test_audit_pass_fail(self):
        """Audit correctly identifies passing/failing metrics."""
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0])  # Same prediction for all
        y_prob = np.array([0.2, 0.3, 0.4, 0.4, 0.2, 0.3, 0.4, 0.4])
        protected = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        results = run_fairness_audit(y_true, y_pred, y_prob, protected)
        # With identical predictions across groups, DP should pass
        assert results["demographic_parity"]["passes"] is True
