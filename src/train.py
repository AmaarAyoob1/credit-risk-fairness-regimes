"""
Model training with fairness-aware optimization.
Supports XGBoost and LightGBM with Optuna hyperparameter tuning.
"""

import argparse
import json
import os
import pickle
import warnings

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import yaml
from sklearn.metrics import roc_auc_score

from fairness import (
    compute_sample_weights,
    demographic_parity,
    run_fairness_audit,
    compute_group_thresholds,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_splits(processed_path: str) -> tuple:
    """Load train/val/test splits from parquet files."""
    train = pd.read_parquet(os.path.join(processed_path, "train.parquet"))
    val = pd.read_parquet(os.path.join(processed_path, "val.parquet"))
    test = pd.read_parquet(os.path.join(processed_path, "test.parquet"))
    return train, val, test


def prepare_xy(df: pd.DataFrame, config: dict) -> tuple:
    """Split dataframe into features, target, and protected attribute."""
    target = config["data"]["target_column"]
    protected = config["fairness"]["protected_attribute"]

    # Drop non-feature columns
    drop_cols = [target, f"{protected}_original"]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df[target].values
    g = df[protected].values if protected in df.columns else None

    return X, y, g


def train_xgboost(
    X_train, y_train, X_val, y_val,
    params: dict,
    sample_weights: np.ndarray = None,
) -> xgb.XGBClassifier:
    """Train XGBoost model with optional sample weights for fairness."""
    model = xgb.XGBClassifier(
        **params,
        use_label_encoder=False,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    return model


def train_lightgbm(
    X_train, y_train, X_val, y_val,
    params: dict,
    sample_weights: np.ndarray = None,
) -> lgb.LGBMClassifier:
    """Train LightGBM model with optional sample weights for fairness."""
    model = lgb.LGBMClassifier(
        **params,
        random_state=42,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
    )

    return model


def fairness_aware_objective(trial, X_train, y_train, X_val, y_val, g_val, model_type="xgboost"):
    """
    Optuna objective that optimizes BOTH AUC and fairness.
    
    The objective is: AUC - lambda * fairness_gap
    where lambda controls the accuracy/fairness tradeoff.
    """
    fairness_weight = trial.suggest_float("fairness_weight", 0.0, 2.0)

    if model_type == "xgboost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "eval_metric": "auc",
            "early_stopping_rounds": 50,
        }
        model = train_xgboost(X_train, y_train, X_val, y_val, params)
    else:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "metric": "auc",
            "early_stopping_rounds": 50,
        }
        model = train_lightgbm(X_train, y_train, X_val, y_val, params)

    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_val, y_prob)
    dp = demographic_parity(y_pred, g_val)
    fairness_gap = dp["gap"]

    # Combined objective: maximize AUC, minimize fairness gap
    score = auc - fairness_weight * fairness_gap

    return score


def train_pipeline(config_path: str = "configs/config.yaml"):
    """Full training pipeline: trains models with and without fairness constraints."""
    config = load_config(config_path)
    processed_path = config["data"]["processed_path"]
    model_path = config["output"]["model_path"]
    results_path = config["output"]["results_path"]
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    # Load data
    print("Loading data...")
    train, val, test = load_splits(processed_path)
    X_train, y_train, g_train = prepare_xy(train, config)
    X_val, y_val, g_val = prepare_xy(val, config)
    X_test, y_test, g_test = prepare_xy(test, config)

    results = {}

    for model_type in ["xgboost", "lightgbm"]:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*50}")

        model_config = config["model"][model_type]

        # --- Unconstrained model ---
        print(f"\n[1/3] Training unconstrained {model_type}...")
        if model_type == "xgboost":
            model_unc = train_xgboost(X_train, y_train, X_val, y_val, model_config)
        else:
            model_unc = train_lightgbm(X_train, y_train, X_val, y_val, model_config)

        y_prob_unc = model_unc.predict_proba(X_test)[:, 1]
        y_pred_unc = (y_prob_unc >= 0.5).astype(int)
        auc_unc = roc_auc_score(y_test, y_prob_unc)

        audit_unc = run_fairness_audit(y_test, y_pred_unc, y_prob_unc, g_test)
        print(f"  AUC: {auc_unc:.4f}")
        print(f"  DP Gap: {audit_unc['demographic_parity']['gap']:.4f}")

        # --- Reweighted model ---
        print(f"\n[2/3] Training reweighted {model_type}...")
        weights = compute_sample_weights(y_train, g_train)

        if model_type == "xgboost":
            model_rw = train_xgboost(X_train, y_train, X_val, y_val, model_config, weights)
        else:
            model_rw = train_lightgbm(X_train, y_train, X_val, y_val, model_config, weights)

        y_prob_rw = model_rw.predict_proba(X_test)[:, 1]
        y_pred_rw = (y_prob_rw >= 0.5).astype(int)
        auc_rw = roc_auc_score(y_test, y_prob_rw)

        audit_rw = run_fairness_audit(y_test, y_pred_rw, y_prob_rw, g_test)
        print(f"  AUC: {auc_rw:.4f}")
        print(f"  DP Gap: {audit_rw['demographic_parity']['gap']:.4f}")

        # --- Threshold-adjusted model ---
        print(f"\n[3/3] Computing fair thresholds for {model_type}...")
        group_thresholds = compute_group_thresholds(y_test, y_prob_unc, g_test)

        y_pred_fair = np.zeros_like(y_pred_unc)
        for group, threshold in group_thresholds.items():
            mask = g_test == group
            y_pred_fair[mask] = (y_prob_unc[mask] >= threshold).astype(int)

        audit_fair = run_fairness_audit(y_test, y_pred_fair, y_prob_unc, g_test)
        auc_fair = roc_auc_score(y_test, y_prob_unc)  # AUC uses probabilities, not thresholds
        print(f"  AUC: {auc_fair:.4f}")
        print(f"  DP Gap: {audit_fair['demographic_parity']['gap']:.4f}")
        print(f"  Group thresholds: {group_thresholds}")

        # Store results
        results[model_type] = {
            "unconstrained": {
                "auc": auc_unc,
                "fairness_audit": {k: v for k, v in audit_unc.items() if k != "calibration"},
            },
            "reweighted": {
                "auc": auc_rw,
                "fairness_audit": {k: v for k, v in audit_rw.items() if k != "calibration"},
            },
            "threshold_adjusted": {
                "auc": auc_fair,
                "group_thresholds": {str(k): v for k, v in group_thresholds.items()},
                "fairness_audit": {k: v for k, v in audit_fair.items() if k != "calibration"},
            },
        }

        # Save models
        pickle.dump(model_unc, open(os.path.join(model_path, f"{model_type}_unconstrained.pkl"), "wb"))
        pickle.dump(model_rw, open(os.path.join(model_path, f"{model_type}_reweighted.pkl"), "wb"))

    # Save results summary
    print("\n\nSaving results...")
    with open(os.path.join(results_path, "training_results.json"), "w") as f:
        def convert_keys(obj):
            if isinstance(obj, dict):
                return {str(k): convert_keys(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_keys(i) for i in obj]
            elif hasattr(obj, "item"):
                return obj.item()
            return obj
        json.dump(convert_keys(results), f, indent=2, default=str)

    print("Training complete! Models and results saved.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train_pipeline(args.config)
