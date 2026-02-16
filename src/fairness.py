"""
Fairness metrics and bias auditing for credit risk models.

Implements demographic parity, equal opportunity, predictive parity,
and calibration metrics aligned with CFPB and EU AI Act requirements.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import Optional


def demographic_parity(
    y_pred: np.ndarray,
    protected: np.ndarray,
    favorable_label: int = 0,
) -> dict:
    """
    Demographic Parity: Are approval rates equal across groups?
    
    Measures: P(Y_hat = favorable | G = g) for each group g.
    A fair model has equal approval rates regardless of group membership.
    
    Args:
        y_pred: Predicted labels (0 = approved/paid, 1 = denied/default)
        protected: Protected attribute group labels
        favorable_label: Which label is "favorable" (0 = no default = approved)
    
    Returns:
        Dict with per-group rates and overall gap.
    """
    groups = np.unique(protected)
    rates = {}

    for g in groups:
        mask = protected == g
        rates[g] = np.mean(y_pred[mask] == favorable_label)

    gap = max(rates.values()) - min(rates.values())

    return {
        "metric": "demographic_parity",
        "group_rates": rates,
        "gap": gap,
        "max_group": max(rates, key=rates.get),
        "min_group": min(rates, key=rates.get),
    }


def equal_opportunity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray,
    favorable_label: int = 0,
) -> dict:
    """
    Equal Opportunity: Are true positive rates equal across groups?
    
    Measures: P(Y_hat = favorable | Y = favorable, G = g)
    Among people who actually repay, are approval rates equal?
    
    This is often considered the most important fairness metric for lending
    because it asks: "Are creditworthy people from all groups equally likely
    to be approved?"
    """
    groups = np.unique(protected)
    rates = {}

    for g in groups:
        mask = (protected == g) & (y_true == favorable_label)
        if mask.sum() > 0:
            rates[g] = np.mean(y_pred[mask] == favorable_label)
        else:
            rates[g] = np.nan

    valid_rates = {k: v for k, v in rates.items() if not np.isnan(v)}
    gap = max(valid_rates.values()) - min(valid_rates.values()) if valid_rates else 0

    return {
        "metric": "equal_opportunity",
        "group_rates": rates,
        "gap": gap,
        "max_group": max(valid_rates, key=valid_rates.get) if valid_rates else None,
        "min_group": min(valid_rates, key=valid_rates.get) if valid_rates else None,
    }


def predictive_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray,
    favorable_label: int = 0,
) -> dict:
    """
    Predictive Parity: Is precision equal across groups?
    
    Measures: P(Y = favorable | Y_hat = favorable, G = g)
    Among approved applicants, are actual repayment rates equal?
    """
    groups = np.unique(protected)
    rates = {}

    for g in groups:
        mask = (protected == g) & (y_pred == favorable_label)
        if mask.sum() > 0:
            rates[g] = np.mean(y_true[mask] == favorable_label)
        else:
            rates[g] = np.nan

    valid_rates = {k: v for k, v in rates.items() if not np.isnan(v)}
    gap = max(valid_rates.values()) - min(valid_rates.values()) if valid_rates else 0

    return {
        "metric": "predictive_parity",
        "group_rates": rates,
        "gap": gap,
    }


def calibration_by_group(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    protected: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Calibration: Are predicted probabilities accurate across groups?
    
    For each group, bins predictions and compares predicted vs actual rates.
    A well-calibrated model means P(Y=1 | score=s, G=g) ≈ s for all groups.
    """
    groups = np.unique(protected)
    group_calibration = {}

    for g in groups:
        mask = protected == g
        probs = y_prob[mask]
        true = y_true[mask]

        bins = np.linspace(0, 1, n_bins + 1)
        bin_means_pred = []
        bin_means_true = []

        for i in range(n_bins):
            bin_mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if bin_mask.sum() > 0:
                bin_means_pred.append(probs[bin_mask].mean())
                bin_means_true.append(true[bin_mask].mean())

        if bin_means_pred:
            calibration_error = np.mean(
                np.abs(np.array(bin_means_pred) - np.array(bin_means_true))
            )
        else:
            calibration_error = np.nan

        group_calibration[g] = {
            "calibration_error": calibration_error,
            "predicted_means": bin_means_pred,
            "actual_means": bin_means_true,
        }

    errors = [v["calibration_error"] for v in group_calibration.values() if not np.isnan(v["calibration_error"])]
    gap = max(errors) - min(errors) if errors else 0

    return {
        "metric": "calibration",
        "group_calibration": group_calibration,
        "gap": gap,
    }


def run_fairness_audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    protected: np.ndarray,
    thresholds: Optional[dict] = None,
) -> dict:
    """
    Run complete fairness audit across all metrics.
    
    Returns a comprehensive report with per-metric results,
    pass/fail status, and recommendations.
    """
    if thresholds is None:
        thresholds = {
            "demographic_parity_gap": 0.05,
            "equal_opportunity_gap": 0.05,
            "predictive_parity_gap": 0.05,
            "calibration_gap": 0.03,
        }

    results = {}

    # Run each metric
    dp = demographic_parity(y_pred, protected)
    eo = equal_opportunity(y_true, y_pred, protected)
    pp = predictive_parity(y_true, y_pred, protected)
    cal = calibration_by_group(y_true, y_prob, protected)

    results["demographic_parity"] = {
        **dp,
        "passes": dp["gap"] <= thresholds["demographic_parity_gap"],
        "threshold": thresholds["demographic_parity_gap"],
    }
    results["equal_opportunity"] = {
        **eo,
        "passes": eo["gap"] <= thresholds["equal_opportunity_gap"],
        "threshold": thresholds["equal_opportunity_gap"],
    }
    results["predictive_parity"] = {
        **pp,
        "passes": pp["gap"] <= thresholds["predictive_parity_gap"],
        "threshold": thresholds["predictive_parity_gap"],
    }
    results["calibration"] = {
        **cal,
        "passes": cal["gap"] <= thresholds["calibration_gap"],
        "threshold": thresholds["calibration_gap"],
    }

    # Overall assessment
    all_pass = all(r["passes"] for r in results.values())
    results["overall"] = {
        "passes_all": all_pass,
        "failed_metrics": [k for k, v in results.items() if k != "overall" and not v["passes"]],
    }

    return results


def compute_group_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    protected: np.ndarray,
    target_rate: Optional[float] = None,
) -> dict:
    """
    Compute per-group decision thresholds to achieve demographic parity.
    
    For each group, finds the threshold that produces an approval rate
    closest to the target rate. If no target is specified, uses the
    overall approval rate.
    
    This is the simplest fairness intervention — it doesn't retrain the model,
    just adjusts the decision boundary per group.
    """
    if target_rate is None:
        target_rate = 1 - np.mean(y_true)  # Overall non-default rate

    groups = np.unique(protected)
    group_thresholds = {}

    for g in groups:
        mask = protected == g
        probs = y_prob[mask]

        # Search for threshold that produces approval rate closest to target
        best_threshold = 0.5
        best_diff = float("inf")

        for t in np.arange(0.1, 0.9, 0.01):
            approval_rate = np.mean(probs < t)  # prob < threshold = approved (no default)
            diff = abs(approval_rate - target_rate)
            if diff < best_diff:
                best_diff = diff
                best_threshold = t

        group_thresholds[g] = best_threshold

    return group_thresholds


def compute_sample_weights(
    y_true: np.ndarray,
    protected: np.ndarray,
) -> np.ndarray:
    """
    Compute reweighting sample weights for fairness.
    
    Weights are inversely proportional to group size × label frequency,
    so underrepresented group-label combinations get higher weight.
    This helps the model pay equal attention to all groups during training.
    """
    n = len(y_true)
    groups = np.unique(protected)
    labels = np.unique(y_true)
    weights = np.ones(n)

    for g in groups:
        for l in labels:
            mask = (protected == g) & (y_true == l)
            count = mask.sum()
            if count > 0:
                # Weight = N / (n_groups × n_labels × count_in_cell)
                weights[mask] = n / (len(groups) * len(labels) * count)

    return weights


def print_fairness_report(audit_results: dict, group_names: Optional[dict] = None):
    """Pretty-print fairness audit results."""
    print("\n" + "=" * 60)
    print("FAIRNESS AUDIT REPORT")
    print("=" * 60)

    for metric_name, result in audit_results.items():
        if metric_name == "overall":
            continue

        status = "PASS ✓" if result["passes"] else "FAIL ✗"
        print(f"\n{metric_name.upper().replace('_', ' ')}: [{status}]")
        print(f"  Gap: {result['gap']:.4f} (threshold: {result['threshold']})")

        if "group_rates" in result:
            for group, rate in result["group_rates"].items():
                name = group_names.get(group, group) if group_names else group
                print(f"  Group '{name}': {rate:.4f}")

    overall = audit_results["overall"]
    print(f"\n{'=' * 60}")
    if overall["passes_all"]:
        print("OVERALL: ALL METRICS PASS ✓")
    else:
        failed = ", ".join(overall["failed_metrics"])
        print(f"OVERALL: FAILED METRICS: {failed}")
    print("=" * 60)
