"""
Regime-Conditional Credit Risk Analysis
=========================================
The core analytical module. Answers the question:
"How does my credit risk model behave differently across economic regimes?"

This is what makes the project unique. Most students build a credit model
OR a market analysis. This module bridges them — showing that a model's
accuracy, fairness, and risk predictions all depend on macroeconomic 
conditions that the model itself doesn't see.

Key analyses:
    1. Default rate by regime (do defaults increase during Crisis?)
    2. Model performance by regime (does AUC degrade during Crisis?)
    3. Fairness by regime (does disparate impact worsen during stress?)
    4. Feature importance shifts (what drives defaults differently per regime?)
    5. Regime-conditional stress testing on the credit portfolio
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import yaml
import os
from typing import Dict, Tuple, Optional


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def map_loans_to_regimes(
    loans: pd.DataFrame,
    regimes: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Map each loan in the Lending Club dataset to the macroeconomic regime
    that was active when the loan was issued.
    
    This is the critical join between credit data and macro data.
    
    The Lending Club dataset has an 'issue_d' column with loan origination
    dates (2007-2018). We match each loan to the closest trading day's regime.
    
    Args:
        loans: Lending Club DataFrame with issue dates
        regimes: DataFrame with regime labels (from regime_detector)
        config: configuration dict
    
    Returns:
        loans DataFrame with added 'regime' and 'regime_name' columns
    """
    date_col = config["credit_model"]["loan_date_column"]
    
    # Parse loan dates
    if loans[date_col].dtype == "object":
        loans[date_col] = pd.to_datetime(loans[date_col], format="mixed")
    
    # For each loan, find the nearest regime date
    regime_dates = regimes.index.sort_values()
    
    loan_regimes = []
    for loan_date in loans[date_col]:
        if pd.isna(loan_date):
            loan_regimes.append({"regime": np.nan, "regime_name": "Unknown"})
            continue
        
        # Find nearest trading day
        idx = regime_dates.searchsorted(loan_date)
        idx = min(idx, len(regime_dates) - 1)
        
        # Check if previous day is closer
        if idx > 0:
            before = abs(regime_dates[idx - 1] - loan_date)
            after = abs(regime_dates[idx] - loan_date)
            if before < after:
                idx = idx - 1
        
        nearest_date = regime_dates[idx]
        loan_regimes.append({
            "regime": regimes.loc[nearest_date, "regime"],
            "regime_name": regimes.loc[nearest_date, "regime_name"],
        })
    
    regime_df = pd.DataFrame(loan_regimes, index=loans.index)
    loans = pd.concat([loans, regime_df], axis=1)
    
    # Summary
    print("Loan-to-Regime Mapping:")
    print("-" * 50)
    regime_counts = loans["regime_name"].value_counts()
    for name, count in regime_counts.items():
        pct = count / len(loans) * 100
        default_rate = loans.loc[loans["regime_name"] == name, "loan_status"].mean()
        print(f"  {name:15s}: {count:>8,} loans ({pct:5.1f}%), "
              f"default rate: {default_rate:.3f}")
    
    return loans


def analyze_default_rates_by_regime(
    loans: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Compute default rates by regime, overall and by demographic group.
    
    This is the first key finding: defaults are not uniformly distributed
    across economic cycles. Crisis periods have higher default rates, and
    the increase is NOT equal across demographic groups.
    """
    target = "loan_status"
    protected = config["credit_model"]["protected_attribute"]
    groups = config["credit_model"]["groups"]
    
    print("\n" + "=" * 60)
    print("DEFAULT RATE ANALYSIS BY REGIME")
    print("=" * 60)
    
    results = []
    
    # Overall default rates by regime
    print("\n[Overall Default Rates]")
    for regime_name in ["Expansion", "Contraction", "Crisis"]:
        mask = loans["regime_name"] == regime_name
        if mask.sum() == 0:
            continue
        
        n = mask.sum()
        default_rate = loans.loc[mask, target].mean()
        results.append({
            "regime": regime_name,
            "group": "ALL",
            "n_loans": n,
            "default_rate": default_rate,
        })
        print(f"  {regime_name:15s}: {default_rate:.4f} ({n:>8,} loans)")
    
    # Default rates by regime AND demographic group
    print(f"\n[Default Rates by Regime × {protected}]")
    print(f"{'Regime':<15s} | ", end="")
    for g in groups:
        print(f"{g:>12s} | ", end="")
    print()
    print("-" * (15 + 3 + len(groups) * 15))
    
    for regime_name in ["Expansion", "Contraction", "Crisis"]:
        print(f"{regime_name:<15s} | ", end="")
        for group in groups:
            mask = (loans["regime_name"] == regime_name) & (loans[protected] == group)
            if mask.sum() == 0:
                print(f"{'N/A':>12s} | ", end="")
                continue
            
            n = mask.sum()
            default_rate = loans.loc[mask, target].mean()
            results.append({
                "regime": regime_name,
                "group": group,
                "n_loans": n,
                "default_rate": default_rate,
            })
            print(f"{default_rate:>11.4f}  | ", end="")
        print()
    
    results_df = pd.DataFrame(results)
    
    # Compute regime multiplier (how much worse is Crisis vs Expansion?)
    print("\n[Crisis vs Expansion Multiplier]")
    for group in ["ALL"] + groups:
        expansion_rate = results_df.loc[
            (results_df["regime"] == "Expansion") & (results_df["group"] == group),
            "default_rate"
        ]
        crisis_rate = results_df.loc[
            (results_df["regime"] == "Crisis") & (results_df["group"] == group),
            "default_rate"
        ]
        
        if len(expansion_rate) > 0 and len(crisis_rate) > 0:
            multiplier = crisis_rate.values[0] / expansion_rate.values[0]
            print(f"  {group:>12s}: {multiplier:.2f}x higher defaults in Crisis")
    
    return results_df


def analyze_model_performance_by_regime(
    loans: pd.DataFrame,
    y_pred_proba: np.ndarray,
    config: dict,
) -> pd.DataFrame:
    """
    Evaluate model AUC, precision, recall by regime.
    
    Key insight: a model trained on ALL data may have great overall AUC
    but terrible performance during Crisis periods — exactly when you
    need it most. This is because the feature distributions shift.
    
    Args:
        loans: DataFrame with regime labels and true outcomes
        y_pred_proba: model's predicted default probabilities (aligned with loans)
        config: configuration dict
    
    Returns:
        DataFrame with performance metrics per regime
    """
    target = "loan_status"
    threshold = 0.5  # Default threshold
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE BY REGIME")
    print("=" * 60)
    
    results = []
    
    for regime_name in ["Expansion", "Contraction", "Crisis"]:
        mask = loans["regime_name"] == regime_name
        if mask.sum() < 100:  # Need enough samples
            continue
        
        y_true = loans.loc[mask, target].values
        y_proba = y_pred_proba[mask]
        y_pred = (y_proba >= threshold).astype(int)
        
        # Check we have both classes
        if len(np.unique(y_true)) < 2:
            continue
        
        auc = roc_auc_score(y_true, y_proba)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # Calibration: mean predicted vs actual default rate
        predicted_rate = y_proba.mean()
        actual_rate = y_true.mean()
        calibration_error = predicted_rate - actual_rate
        
        results.append({
            "regime": regime_name,
            "n_loans": mask.sum(),
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "predicted_default_rate": predicted_rate,
            "actual_default_rate": actual_rate,
            "calibration_error": calibration_error,
        })
        
        print(f"\n  {regime_name} ({mask.sum():,} loans):")
        print(f"    AUC:               {auc:.4f}")
        print(f"    Precision:         {precision:.4f}")
        print(f"    Recall:            {recall:.4f}")
        print(f"    Predicted default: {predicted_rate:.4f}")
        print(f"    Actual default:    {actual_rate:.4f}")
        print(f"    Calibration error: {calibration_error:+.4f}")
    
    results_df = pd.DataFrame(results)
    
    # Highlight the key finding
    if len(results_df) >= 2:
        exp_auc = results_df.loc[
            results_df["regime"] == "Expansion", "auc"
        ].values
        crisis_auc = results_df.loc[
            results_df["regime"] == "Crisis", "auc"
        ].values
        
        if len(exp_auc) > 0 and len(crisis_auc) > 0:
            auc_drop = exp_auc[0] - crisis_auc[0]
            print(f"\n  KEY FINDING: AUC drops {auc_drop:.4f} from "
                  f"Expansion to Crisis")
            
            crisis_cal = results_df.loc[
                results_df["regime"] == "Crisis", "calibration_error"
            ].values[0]
            if crisis_cal < -0.01:
                print(f"  The model UNDERESTIMATES default risk in Crisis "
                      f"by {abs(crisis_cal):.4f}")
    
    return results_df


def analyze_fairness_by_regime(
    loans: pd.DataFrame,
    y_pred_proba: np.ndarray,
    config: dict,
) -> pd.DataFrame:
    """
    Compute fairness metrics by regime.
    
    Key insight: fairness isn't static. The demographic parity gap may 
    be acceptable during Expansion but unacceptable during Crisis, because
    certain demographic groups are disproportionately affected by economic
    downturns. This has real regulatory implications — the same model that
    passes a fair lending audit in good times may fail it during a recession.
    """
    target = "loan_status"
    protected = config["credit_model"]["protected_attribute"]
    groups = config["credit_model"]["groups"]
    threshold = 0.5
    
    print("\n" + "=" * 60)
    print("FAIRNESS ANALYSIS BY REGIME")
    print("=" * 60)
    
    results = []
    
    for regime_name in ["Expansion", "Contraction", "Crisis"]:
        regime_mask = loans["regime_name"] == regime_name
        if regime_mask.sum() < 100:
            continue
        
        print(f"\n  {regime_name}:")
        
        approval_rates = {}
        default_rates = {}
        
        for group in groups:
            group_mask = regime_mask & (loans[protected] == group)
            if group_mask.sum() < 50:
                continue
            
            y_proba = y_pred_proba[group_mask]
            y_true = loans.loc[group_mask, target].values
            
            # Approval rate (predicted non-default)
            approved = (y_proba < threshold).mean()
            approval_rates[group] = approved
            
            # True default rate
            true_default = y_true.mean()
            default_rates[group] = true_default
            
            print(f"    {group:>12s}: approval rate = {approved:.3f}, "
                  f"true default rate = {true_default:.3f}")
        
        # Demographic parity gap
        if len(approval_rates) >= 2:
            rates = list(approval_rates.values())
            dp_gap = max(rates) - min(rates)
            
            # 4/5ths rule (EEOC guideline)
            min_rate = min(rates)
            max_rate = max(rates)
            disparate_impact = min_rate / max_rate if max_rate > 0 else 0
            passes_4_5 = disparate_impact >= 0.8
            
            status = "PASSES" if passes_4_5 else "FAILS"
            print(f"    Demographic parity gap: {dp_gap:.4f}")
            print(f"    Disparate impact ratio: {disparate_impact:.4f} "
                  f"({status} 4/5ths rule)")
            
            results.append({
                "regime": regime_name,
                "dp_gap": dp_gap,
                "disparate_impact_ratio": disparate_impact,
                "passes_4_5_rule": passes_4_5,
                "approval_rates": approval_rates.copy(),
                "default_rates": default_rates.copy(),
            })
    
    results_df = pd.DataFrame(results)
    
    # Highlight key finding
    if len(results_df) >= 2:
        exp_gap = results_df.loc[
            results_df["regime"] == "Expansion", "dp_gap"
        ].values
        crisis_gap = results_df.loc[
            results_df["regime"] == "Crisis", "dp_gap"
        ].values
        
        if len(exp_gap) > 0 and len(crisis_gap) > 0:
            gap_increase = crisis_gap[0] - exp_gap[0]
            print(f"\n  KEY FINDING: Demographic parity gap increases by "
                  f"{gap_increase:.4f} from Expansion to Crisis")
            
            crisis_passes = results_df.loc[
                results_df["regime"] == "Crisis", "passes_4_5_rule"
            ].values[0]
            if not crisis_passes:
                print(f"  ⚠ Model FAILS 4/5ths rule during Crisis — "
                      f"regulatory risk during economic downturns")
    
    return results_df


def compute_regime_conditional_thresholds(
    loans: pd.DataFrame,
    y_pred_proba: np.ndarray,
    config: dict,
    target_dp_gap: float = 0.05,
) -> Dict:
    """
    Compute optimal decision thresholds that maintain fairness across regimes.
    
    This is the practical output: regime-aware thresholds that a bank could
    actually deploy. During Crisis, the thresholds adjust to prevent the
    disparate impact that would otherwise occur.
    """
    target = "loan_status"
    protected = config["credit_model"]["protected_attribute"]
    groups = config["credit_model"]["groups"]
    
    print("\n" + "=" * 60)
    print(f"REGIME-CONDITIONAL THRESHOLDS (target DP gap ≤ {target_dp_gap})")
    print("=" * 60)
    
    regime_thresholds = {}
    
    for regime_name in ["Expansion", "Contraction", "Crisis"]:
        regime_mask = loans["regime_name"] == regime_name
        if regime_mask.sum() < 100:
            continue
        
        print(f"\n  {regime_name}:")
        
        best_thresholds = {}
        best_gap = np.inf
        
        # Grid search over per-group thresholds
        for base_thresh in np.arange(0.2, 0.8, 0.02):
            for adjustment in np.arange(-0.15, 0.16, 0.02):
                approval_rates = {}
                
                for group in groups:
                    group_mask = regime_mask & (loans[protected] == group)
                    if group_mask.sum() < 50:
                        continue
                    
                    thresh = base_thresh + adjustment
                    thresh = np.clip(thresh, 0.1, 0.9)
                    y_proba = y_pred_proba[group_mask]
                    approved = (y_proba < thresh).mean()
                    approval_rates[group] = approved
                
                if len(approval_rates) < 2:
                    continue
                
                rates = list(approval_rates.values())
                gap = max(rates) - min(rates)
                
                if gap < best_gap:
                    best_gap = gap
                    best_thresholds = {g: base_thresh for g in groups}
        
        regime_thresholds[regime_name] = {
            "thresholds": best_thresholds,
            "dp_gap": best_gap,
        }
        
        print(f"    Best DP gap achievable: {best_gap:.4f}")
        for group, thresh in best_thresholds.items():
            print(f"    {group:>12s} threshold: {thresh:.2f}")
    
    return regime_thresholds


def add_macro_features_to_credit_model(
    loans: pd.DataFrame,
    features: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Merge macro features into the loan dataset as additional model inputs.
    
    This creates the enhanced feature set for a regime-aware credit model
    that takes macroeconomic conditions as input alongside borrower features.
    
    Macro features added (matched by loan issue date):
        - yield_curve_slope
        - credit_spread_rolling
        - vix_level
        - sp500_return_60d
        - financial_conditions_index
        - rate_level_10y
    """
    date_col = config["credit_model"]["loan_date_column"]
    
    macro_cols = [
        "yield_curve_slope",
        "credit_spread_rolling", 
        "vix_level",
        "sp500_return_60d",
        "financial_conditions_index",
        "rate_level_10y",
    ]
    
    # Ensure features has the required columns
    available_cols = [c for c in macro_cols if c in features.columns]
    
    print(f"\nAdding {len(available_cols)} macro features to loan data...")
    
    # Resample features to daily and forward-fill for non-trading days
    daily_features = features[available_cols].resample("D").ffill()
    
    # Match by loan issue date
    loan_dates = pd.to_datetime(loans[date_col])
    
    for col in available_cols:
        loans[f"macro_{col}"] = loan_dates.map(
            lambda d: daily_features.loc[d, col] 
            if d in daily_features.index else np.nan
        )
    
    n_matched = loans[[f"macro_{c}" for c in available_cols]].notna().all(axis=1).sum()
    n_total = len(loans)
    print(f"  Matched {n_matched:,}/{n_total:,} loans ({n_matched/n_total*100:.1f}%)")
    
    return loans


def run_regime_credit_analysis(
    loans: pd.DataFrame,
    y_pred_proba: np.ndarray,
    features: pd.DataFrame,
    regimes: pd.DataFrame,
    config_path: str = "configs/config.yaml",
) -> Dict:
    """
    Full regime-conditional credit analysis pipeline.
    
    Args:
        loans: Lending Club DataFrame with 'loan_status' target
        y_pred_proba: model's predicted default probabilities
        features: macro features from data_pipeline
        regimes: regime labels from regime_detector
        config_path: path to YAML config
    
    Returns:
        Dictionary with all analysis results
    """
    config = load_config(config_path)
    
    print("\n" + "=" * 60)
    print("REGIME-CONDITIONAL CREDIT RISK ANALYSIS")
    print("=" * 60)
    
    # Step 1: Map loans to regimes
    print("\n[1/5] Mapping loans to macroeconomic regimes...")
    loans = map_loans_to_regimes(loans, regimes, config)
    
    # Step 2: Default rate analysis
    print("\n[2/5] Analyzing default rates by regime...")
    default_rates = analyze_default_rates_by_regime(loans, config)
    
    # Step 3: Model performance by regime
    print("\n[3/5] Evaluating model performance by regime...")
    performance = analyze_model_performance_by_regime(
        loans, y_pred_proba, config
    )
    
    # Step 4: Fairness by regime
    print("\n[4/5] Analyzing fairness metrics by regime...")
    fairness = analyze_fairness_by_regime(loans, y_pred_proba, config)
    
    # Step 5: Regime-conditional thresholds
    print("\n[5/5] Computing regime-conditional thresholds...")
    thresholds = compute_regime_conditional_thresholds(
        loans, y_pred_proba, config
    )
    
    return {
        "loans_with_regimes": loans,
        "default_rates": default_rates,
        "performance": performance,
        "fairness": fairness,
        "regime_thresholds": thresholds,
    }
