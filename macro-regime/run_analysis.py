"""
Macro-Regime Credit Risk Analysis — Main Pipeline
====================================================
Run the full analysis from Yahoo Finance data pull through
regime detection to credit model evaluation.

Usage:
    python run_analysis.py                    # Full pipeline
    python run_analysis.py --skip-download    # Use cached data
    python run_analysis.py --regimes-only     # Only detect regimes

Requirements:
    pip install yfinance hmmlearn xgboost scikit-learn pandas numpy 
                matplotlib seaborn pyyaml joblib
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import os
import sys

from src.data_pipeline import run_pipeline, load_config
from src.regime_detector import detect_regimes
from src.regime_credit_analysis import (
    map_loans_to_regimes,
    analyze_default_rates_by_regime,
    analyze_model_performance_by_regime,
    analyze_fairness_by_regime,
    compute_regime_conditional_thresholds,
    add_macro_features_to_credit_model,
)
from src.visualizations import (
    generate_all_plots,
    plot_regime_timeline,
    plot_financial_conditions_index,
)


def load_credit_model_and_data(config: dict):
    """
    Load the existing Loan Default model and Lending Club data.
    
    This connects to your existing loan-default-fairness project.
    Adjust paths in config.yaml to match your local setup.
    """
    model_path = config["credit_model"]["model_path"]
    
    print(f"\nLoading credit model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"  ⚠ Model not found at {model_path}")
        print(f"  Please update 'credit_model.model_path' in configs/config.yaml")
        print(f"  to point to your xgboost_unconstrained.pkl file")
        return None, None, None
    
    model = joblib.load(model_path)
    print(f"  Model loaded: {type(model).__name__}")
    
    # Load Lending Club data
    # Try common locations
    data_paths = [
        "../loan-default-fairness/data/lending_club_cleaned.csv",
        "../loan-default-fairness/data/lending_club.csv",
        "data/lending_club_cleaned.csv",
    ]
    
    loans = None
    for path in data_paths:
        if os.path.exists(path):
            print(f"  Loading loan data from: {path}")
            loans = pd.read_csv(path)
            print(f"  Loaded {len(loans):,} loans")
            break
    
    if loans is None:
        print(f"  ⚠ Lending Club data not found. Tried: {data_paths}")
        print(f"  Place your cleaned CSV in one of the above locations.")
        return model, None, None
    
    return model, loans, None


def generate_predictions(model, loans: pd.DataFrame) -> np.ndarray:
    """
    Generate predictions from the existing credit model.
    
    Uses the same feature engineering as your original model.
    Adjust feature list if your model uses different columns.
    """
    # Feature columns used by the original XGBoost model
    # These should match what you trained on
    feature_cols = [
        "loan_amnt", "term", "int_rate", "installment", "annual_inc",
        "dti", "open_acc", "pub_rec", "revol_bal", "revol_util",
        "total_acc", "mort_acc", "pub_rec_bankruptcies",
        # Engineered features (if present)
        "credit_utilization", "dti_x_int_rate", "income_to_loan",
    ]
    
    # Use whatever features are available
    available = [c for c in feature_cols if c in loans.columns]
    print(f"\n  Using {len(available)} features for prediction")
    
    X = loans[available].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Generate predictions
    y_pred_proba = model.predict_proba(X)[:, 1]  # P(default)
    
    print(f"  Predictions: mean={y_pred_proba.mean():.4f}, "
          f"std={y_pred_proba.std():.4f}")
    
    return y_pred_proba


def run_full_pipeline(args):
    """Execute the complete analysis pipeline."""
    config = load_config()
    
    print("=" * 70)
    print("  MACRO-REGIME CREDIT RISK ANALYSIS")
    print("  Connecting Credit Risk Models to Economic Cycles")
    print("=" * 70)
    
    # =====================================================================
    # STAGE 1: Macro Data Pipeline
    # =====================================================================
    if args.skip_download:
        print("\n[Stage 1] Loading cached macro data...")
        features = pd.read_csv(
            config["data"]["features_path"], index_col=0, parse_dates=True
        )
        print(f"  Loaded {features.shape[0]} days × {features.shape[1]} features")
    else:
        print("\n[Stage 1] Pulling macro data from Yahoo Finance...")
        features = run_pipeline()
    
    # =====================================================================
    # STAGE 2: Regime Detection
    # =====================================================================
    print("\n[Stage 2] Detecting macroeconomic regimes...")
    regimes = detect_regimes(features)
    
    if args.regimes_only:
        print("\n  --regimes-only flag set. Generating regime plots only.")
        plot_regime_timeline(features, regimes)
        plot_financial_conditions_index(features, regimes)
        print("\nDone! Check figures/ directory.")
        return
    
    # =====================================================================
    # STAGE 3: Credit Model Integration
    # =====================================================================
    print("\n[Stage 3] Loading credit model and loan data...")
    model, loans, _ = load_credit_model_and_data(config)
    
    if loans is None:
        print("\n  Cannot proceed without loan data.")
        print("  Running regime analysis and visualizations only...")
        plot_regime_timeline(features, regimes)
        plot_financial_conditions_index(features, regimes)
        print("\nPartial analysis complete. Add loan data to run full pipeline.")
        return
    
    # Generate predictions
    print("\n  Generating model predictions...")
    y_pred_proba = generate_predictions(model, loans)
    
    # =====================================================================
    # STAGE 4: Regime-Conditional Analysis
    # =====================================================================
    print("\n[Stage 4] Running regime-conditional credit analysis...")
    
    # Map loans to regimes
    loans = map_loans_to_regimes(loans, regimes, config)
    
    # Default rates by regime
    default_rates = analyze_default_rates_by_regime(loans, config)
    
    # Model performance by regime
    performance = analyze_model_performance_by_regime(
        loans, y_pred_proba, config
    )
    
    # Fairness by regime
    fairness = analyze_fairness_by_regime(loans, y_pred_proba, config)
    
    # Regime-conditional thresholds
    thresholds = compute_regime_conditional_thresholds(
        loans, y_pred_proba, config
    )
    
    # =====================================================================
    # STAGE 5: Enhanced Model with Macro Features
    # =====================================================================
    print("\n[Stage 5] Adding macro features to credit model...")
    loans_enhanced = add_macro_features_to_credit_model(
        loans, features, config
    )
    
    # Save enhanced dataset
    os.makedirs("data", exist_ok=True)
    loans_enhanced.to_csv("data/loans_with_regimes_and_macro.csv", index=False)
    print(f"  Saved enhanced dataset: data/loans_with_regimes_and_macro.csv")
    
    # =====================================================================
    # STAGE 6: Visualizations
    # =====================================================================
    analysis_results = {
        "default_rates": default_rates,
        "performance": performance,
        "fairness": fairness,
    }
    generate_all_plots(features, regimes, analysis_results)
    
    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE — KEY FINDINGS")
    print("=" * 70)
    
    # Default rate shift
    if len(default_rates) > 0:
        overall = default_rates[default_rates["group"] == "ALL"]
        if len(overall) >= 2:
            exp_rate = overall.loc[
                overall["regime"] == "Expansion", "default_rate"
            ].values
            crisis_rate = overall.loc[
                overall["regime"] == "Crisis", "default_rate"
            ].values
            if len(exp_rate) > 0 and len(crisis_rate) > 0:
                mult = crisis_rate[0] / exp_rate[0]
                print(f"\n  1. Default rate is {mult:.1f}× higher during "
                      f"Crisis vs Expansion")
    
    # AUC degradation
    if len(performance) >= 2:
        exp_auc = performance.loc[
            performance["regime"] == "Expansion", "auc"
        ].values
        crisis_auc = performance.loc[
            performance["regime"] == "Crisis", "auc"
        ].values
        if len(exp_auc) > 0 and len(crisis_auc) > 0:
            print(f"  2. Model AUC drops from {exp_auc[0]:.3f} to "
                  f"{crisis_auc[0]:.3f} during Crisis")
    
    # Fairness degradation
    if len(fairness) >= 2:
        exp_pass = fairness.loc[
            fairness["regime"] == "Expansion", "passes_4_5_rule"
        ].values
        crisis_pass = fairness.loc[
            fairness["regime"] == "Crisis", "passes_4_5_rule"
        ].values
        if len(crisis_pass) > 0 and not crisis_pass[0]:
            print(f"  3. Model FAILS 4/5ths rule during Crisis — "
                  f"regulatory risk under stress")
    
    print(f"\n  Outputs:")
    print(f"    figures/regime_timeline.png")
    print(f"    figures/default_rates_by_regime.png")
    print(f"    figures/model_performance_by_regime.png")
    print(f"    figures/fairness_by_regime.png")
    print(f"    figures/financial_conditions_index.png")
    print(f"    data/loans_with_regimes_and_macro.csv")
    print(f"    data/regime_labels.csv")
    print(f"    data/hmm_model.pkl")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Macro-Regime Credit Risk Analysis"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Use cached macro data instead of pulling from Yahoo Finance"
    )
    parser.add_argument(
        "--regimes-only", action="store_true",
        help="Only run regime detection (no credit model analysis)"
    )
    
    args = parser.parse_args()
    run_full_pipeline(args)
