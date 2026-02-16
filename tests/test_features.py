"""Tests for feature engineering functions."""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from features import engineer_features


@pytest.fixture
def sample_loan_data():
    """Create minimal sample data for testing."""
    return pd.DataFrame({
        "loan_amnt": [10000, 20000, 5000],
        "int_rate": [10.5, 15.0, 7.5],
        "installment": [330, 700, 160],
        "annual_inc": [60000, 45000, 80000],
        "dti": [15.0, 25.0, 8.0],
        "open_acc": [8, 5, 12],
        "pub_rec": [0, 1, 0],
        "revol_bal": [5000, 15000, 2000],
        "revol_util": [45.0, 80.0, 20.0],
        "total_acc": [20, 15, 30],
        "mort_acc": [1, 0, 3],
        "pub_rec_bankruptcies": [0, 0, 0],
    })


class TestFeatureEngineering:
    def test_creates_new_columns(self, sample_loan_data):
        """Engineer features should add new columns."""
        result = engineer_features(sample_loan_data)
        expected_new = [
            "debt_to_income_ratio", "credit_utilization",
            "income_to_loan_ratio", "monthly_burden_ratio",
        ]
        for col in expected_new:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_missing_values(self, sample_loan_data):
        """Engineered features should not introduce NaN values."""
        result = engineer_features(sample_loan_data)
        new_cols = [c for c in result.columns if c not in sample_loan_data.columns]
        for col in new_cols:
            assert result[col].isna().sum() == 0, f"NaN found in {col}"

    def test_derogatory_flag(self, sample_loan_data):
        """Derogatory flag should be 1 when pub_rec > 0."""
        result = engineer_features(sample_loan_data)
        assert result.iloc[1]["derogatory_flag"] == 1  # pub_rec = 1
        assert result.iloc[0]["derogatory_flag"] == 0  # pub_rec = 0

    def test_zero_income_handling(self, sample_loan_data):
        """Should handle zero income without division errors."""
        sample_loan_data.loc[0, "annual_inc"] = 0
        result = engineer_features(sample_loan_data)
        assert np.isfinite(result.iloc[0]["debt_to_income_ratio"])
        assert np.isfinite(result.iloc[0]["monthly_burden_ratio"])

    def test_preserves_original_columns(self, sample_loan_data):
        """Original columns should still be present."""
        result = engineer_features(sample_loan_data)
        for col in sample_loan_data.columns:
            assert col in result.columns
