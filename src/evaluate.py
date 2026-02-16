"""
Model evaluation with comprehensive performance and fairness reporting.
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from fairness import run_fairness_audit, print_fairness_report


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    g_test: np.ndarray,
    model_name: str = "Model",
    thresholds: dict = None,
) -> dict:
    """
    Comprehensive model evaluation: performance + fairness.
    
    Args:
        model: Trained sklearn-compatible classifier
        X_test: Test features
        y_test: True labels
        g_test: Protected attribute values
        model_name: Name for display
        thresholds: Optional fairness thresholds from config
    
    Returns:
        Dict with all metrics.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Performance metrics
    performance = {
        "auc": roc_auc_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    performance["confusion_matrix"] = cm.tolist()
    performance["true_negatives"] = int(cm[0, 0])
    performance["false_positives"] = int(cm[0, 1])
    performance["false_negatives"] = int(cm[1, 0])
    performance["true_positives"] = int(cm[1, 1])

    # ROC curve data
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    performance["roc_curve"] = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }

    # Fairness audit
    fairness_results = run_fairness_audit(y_test, y_pred, y_prob, g_test, thresholds)

    # Print report
    print(f"\n{'='*50}")
    print(f"EVALUATION: {model_name}")
    print(f"{'='*50}")
    print(f"AUC:       {performance['auc']:.4f}")
    print(f"Accuracy:  {performance['accuracy']:.4f}")
    print(f"Precision: {performance['precision']:.4f}")
    print(f"Recall:    {performance['recall']:.4f}")
    print(f"F1 Score:  {performance['f1']:.4f}")

    print_fairness_report(fairness_results)

    return {
        "model_name": model_name,
        "performance": performance,
        "fairness": fairness_results,
    }


def compare_models(results_list: list[dict]) -> pd.DataFrame:
    """
    Create comparison table across multiple model evaluations.
    
    Args:
        results_list: List of dicts from evaluate_model()
    
    Returns:
        DataFrame comparing all models.
    """
    rows = []
    for result in results_list:
        row = {
            "Model": result["model_name"],
            "AUC": result["performance"]["auc"],
            "Accuracy": result["performance"]["accuracy"],
            "Precision": result["performance"]["precision"],
            "Recall": result["performance"]["recall"],
            "F1": result["performance"]["f1"],
            "DP Gap": result["fairness"]["demographic_parity"]["gap"],
            "EO Gap": result["fairness"]["equal_opportunity"]["gap"],
            "PP Gap": result["fairness"]["predictive_parity"]["gap"],
            "DP Pass": result["fairness"]["demographic_parity"]["passes"],
            "EO Pass": result["fairness"]["equal_opportunity"]["passes"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(df.to_string(index=False))
    return df
