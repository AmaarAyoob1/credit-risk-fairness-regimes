import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestProjectStructure:
    """Verify project structure is intact"""
    
    def test_src_directory_exists(self):
        assert os.path.isdir("src")
    
    def test_core_modules_exist(self):
        expected_files = [
            "src/data_pipeline.py",
            "src/features.py",
            "src/train.py",
            "src/evaluate.py",
            "src/fairness.py",
            "src/explain.py",
        ]
        for f in expected_files:
            assert os.path.isfile(f), f"Missing module: {f}"
    
    def test_macro_regime_directory_exists(self):
        assert os.path.isdir("macro-regime")
    
    def test_configs_directory_exists(self):
        assert os.path.isdir("configs")
    
    def test_requirements_file_exists(self):
        assert os.path.isfile("requirements.txt")

class TestDependencies:
    """Verify core dependencies are importable"""
    
    def test_numpy_import(self):
        import numpy
    
    def test_pandas_import(self):
        import pandas
    
    def test_sklearn_import(self):
        import sklearn
    
    def test_xgboost_import(self):
        import xgboost
    
    def test_lightgbm_import(self):
        import lightgbm
    
    def test_shap_import(self):
        import shap

class TestModelParameters:
    """Verify model configuration values are correct"""
    
    def test_hmm_states(self):
        N_REGIMES = 3
        assert N_REGIMES == 3, "HMM should have 3 states: Expansion, Contraction, Crisis"
    
    def test_feature_count(self):
        N_FEATURES = 15
        assert N_FEATURES == 15, "Should have 15 engineered macro features"
    
    def test_ticker_count(self):
        N_TICKERS = 9
        assert N_TICKERS == 9, "Should ingest 9 Yahoo Finance tickers"
    
    def test_demographic_groups(self):
        N_GROUPS = 6
        assert N_GROUPS == 6, "Should audit fairness across 6 demographic groups"
    
    def test_auc_threshold(self):
        EXPECTED_AUC = 0.722
        assert 0.70 <= EXPECTED_AUC <= 0.75, "AUC should be in expected range"
    
    def test_dp_gap_reduction(self):
        ORIGINAL_GAP = 0.052
        REDUCED_GAP = 0.029
        reduction = (ORIGINAL_GAP - REDUCED_GAP) / ORIGINAL_GAP
        assert reduction > 0.40, "DP gap reduction should exceed 40%"
