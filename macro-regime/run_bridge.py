"""
Bridge script: Connect Loan Default model to Macro-Regime analysis.
Recovers issue_d from raw data and aligns with processed features.
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import yaml

# Paths
RAW_DATA = "/Users/ayoobamaar/Desktop/loan-default-fairness/data/raw/lending_club.csv"
PROCESSED_DIR = "/Users/ayoobamaar/Desktop/loan-default-fairness/data/processed"
MODEL_PATH = "/Users/ayoobamaar/Desktop/loan-default-fairness/models/xgboost_unconstrained.pkl"
MACRO_FEATURES = "/Users/ayoobamaar/Desktop/macro-regime/data/macro_features.csv"
REGIME_LABELS = "/Users/ayoobamaar/Desktop/macro-regime/data/regime_labels.csv"
CONFIG_PATH = "/Users/ayoobamaar/Desktop/macro-regime/configs/config.yaml"

print("=" * 60)
print("BRIDGE: Connecting Credit Model to Macro Regimes")
print("=" * 60)

# ---- Step 1: Load processed test data ----
print("\n[1/6] Loading processed test data...")
test = pd.read_parquet(os.path.join(PROCESSED_DIR, "test.parquet"))
train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
val = pd.read_parquet(os.path.join(PROCESSED_DIR, "val.parquet"))

# Combine all splits for full analysis
all_data = pd.concat([train, val, test], axis=0)
print(f"  Total processed loans: {len(all_data):,}")

# ---- Step 2: Recover issue_d from raw data ----
print("\n[2/6] Recovering issue dates from raw data...")
print("  (This takes a moment — reading 1.6GB file...)")

# Read only the columns we need from raw data
raw_cols = ["issue_d", "loan_status", "loan_amnt", "int_rate", "installment", 
            "annual_inc", "home_ownership"]
raw = pd.read_csv(RAW_DATA, usecols=raw_cols, low_memory=False)
print(f"  Raw rows: {len(raw):,}")

# Apply same filters as data_pipeline.py
# Filter to completed loans (same logic)
raw = raw[raw["loan_status"].isin([
    "Fully Paid", "Charged Off", "Default"
])].copy()
print(f"  After filtering completed loans: {len(raw):,}")

# Parse dates
raw["issue_d"] = pd.to_datetime(raw["issue_d"], format="mixed")
print(f"  Date range: {raw['issue_d'].min()} to {raw['issue_d'].max()}")

# We can't perfectly align raw to processed (rows were dropped during cleaning).
# Instead, assign dates proportionally — the processed data maintains row order.
# Reset index on both to align.
raw_clean = raw.dropna(subset=["issue_d"]).reset_index(drop=True)
all_data_reset = all_data.reset_index(drop=True)

# Match by loan_amnt + int_rate + installment (unique enough combination)
# But simpler: since data_pipeline preserves order, we can use the date distribution
# to assign dates to the processed data.

# Strategy: create a date lookup from raw data using loan characteristics
raw_clean["_key"] = (
    raw_clean["loan_amnt"].astype(str) + "_" + 
    raw_clean["int_rate"].astype(str) + "_" + 
    raw_clean["installment"].astype(str)
)

all_data_reset["_key"] = (
    all_data_reset["loan_amnt"].astype(str) + "_" + 
    all_data_reset["int_rate"].astype(str) + "_" + 
    all_data_reset["installment"].astype(str)
)

# Build date lookup (first match wins)
date_lookup = raw_clean.drop_duplicates(subset="_key", keep="first").set_index("_key")["issue_d"]

# Map dates
all_data_reset["issue_d"] = all_data_reset["_key"].map(date_lookup)
matched = all_data_reset["issue_d"].notna().sum()
total = len(all_data_reset)
print(f"  Matched {matched:,}/{total:,} loans to dates ({matched/total*100:.1f}%)")

# Drop unmatched
loans = all_data_reset[all_data_reset["issue_d"].notna()].copy()
print(f"  Loans with dates: {len(loans):,}")

# ---- Step 3: Load model and generate predictions ----
print("\n[3/6] Loading model and generating predictions...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Get feature columns (everything except target and helper columns)
drop_cols = ["loan_status", "issue_d", "_key", "home_ownership_original"]
drop_cols = [c for c in drop_cols if c in loans.columns]
feature_cols = [c for c in loans.columns if c not in drop_cols]

X = loans[feature_cols].copy()
y = loans["loan_status"].values

# Handle any missing values
X = X.fillna(X.median())

y_pred_proba = model.predict_proba(X)[:, 1]
print(f"  Predictions: mean={y_pred_proba.mean():.4f}, std={y_pred_proba.std():.4f}")

# ---- Step 4: Load regime data ----
print("\n[4/6] Loading regime labels...")
regimes = pd.read_csv(REGIME_LABELS, index_col=0, parse_dates=True)
features = pd.read_csv(MACRO_FEATURES, index_col=0, parse_dates=True)
print(f"  Regime data: {len(regimes):,} trading days")
print(f"  Date range: {regimes.index.min().date()} to {regimes.index.max().date()}")

# ---- Step 5: Map loans to regimes ----
print("\n[5/6] Mapping loans to macroeconomic regimes...")

regime_dates = regimes.index.sort_values()
loan_dates = loans["issue_d"].values

regime_labels = []
for ld in loan_dates:
    ld_ts = pd.Timestamp(ld)
    idx = regime_dates.searchsorted(ld_ts)
    idx = min(idx, len(regime_dates) - 1)
    if idx > 0:
        before = abs(regime_dates[idx - 1] - ld_ts)
        after = abs(regime_dates[idx] - ld_ts)
        if before < after:
            idx = idx - 1
    regime_labels.append(regimes.iloc[idx]["regime_name"])

loans["regime_name"] = regime_labels

# Print distribution
print("\n  Loan Distribution by Regime:")
print("  " + "-" * 50)
for regime in ["Expansion", "Contraction", "Crisis"]:
    mask = loans["regime_name"] == regime
    n = mask.sum()
    pct = n / len(loans) * 100
    default_rate = loans.loc[mask, "loan_status"].mean()
    print(f"  {regime:15s}: {n:>10,} loans ({pct:5.1f}%), default rate: {default_rate:.4f}")

# ---- Step 6: Run regime-conditional analysis ----
print("\n[6/6] Running regime-conditional analysis...")

from sklearn.metrics import roc_auc_score

print("\n  MODEL PERFORMANCE BY REGIME:")
print("  " + "-" * 60)
regime_results = []
for regime in ["Expansion", "Contraction", "Crisis"]:
    mask = loans["regime_name"] == regime
    if mask.sum() < 100:
        continue
    
    y_true = loans.loc[mask, "loan_status"].values
    y_proba = y_pred_proba[mask.values]
    y_pred = (y_proba >= 0.5).astype(int)
    
    if len(np.unique(y_true)) < 2:
        continue
    
    auc = roc_auc_score(y_true, y_proba)
    predicted_rate = y_proba.mean()
    actual_rate = y_true.mean()
    calibration_error = predicted_rate - actual_rate
    
    regime_results.append({
        "regime": regime,
        "n_loans": mask.sum(),
        "auc": auc,
        "predicted_default_rate": predicted_rate,
        "actual_default_rate": actual_rate,
        "calibration_error": calibration_error,
    })
    
    print(f"\n  {regime} ({mask.sum():,} loans):")
    print(f"    AUC:               {auc:.4f}")
    print(f"    Predicted default: {predicted_rate:.4f}")
    print(f"    Actual default:    {actual_rate:.4f}")
    print(f"    Calibration error: {calibration_error:+.4f}")

# Key findings
if len(regime_results) >= 2:
    exp = [r for r in regime_results if r["regime"] == "Expansion"]
    crisis = [r for r in regime_results if r["regime"] == "Crisis"]
    
    if exp and crisis:
        auc_drop = exp[0]["auc"] - crisis[0]["auc"]
        default_mult = crisis[0]["actual_default_rate"] / exp[0]["actual_default_rate"]
        cal_crisis = crisis[0]["calibration_error"]
        
        print("\n" + "=" * 60)
        print("  KEY FINDINGS")
        print("=" * 60)
        print(f"  1. Default rate is {default_mult:.2f}x higher during Crisis vs Expansion")
        print(f"  2. AUC drops {auc_drop:.4f} from Expansion to Crisis")
        if cal_crisis < -0.01:
            print(f"  3. Model UNDERESTIMATES default risk in Crisis by {abs(cal_crisis):.4f}")
        elif cal_crisis > 0.01:
            print(f"  3. Model OVERESTIMATES default risk in Crisis by {abs(cal_crisis):.4f}")

# Fairness by regime
print("\n  FAIRNESS BY REGIME (home_ownership groups):")
print("  " + "-" * 60)

# home_ownership is encoded — find the column
ho_col = "home_ownership"
if ho_col in loans.columns:
    for regime in ["Expansion", "Contraction", "Crisis"]:
        regime_mask = loans["regime_name"] == regime
        if regime_mask.sum() < 100:
            continue
        
        print(f"\n  {regime}:")
        approval_rates = {}
        for group_val in loans[ho_col].unique():
            group_mask = regime_mask & (loans[ho_col] == group_val)
            if group_mask.sum() < 50:
                continue
            y_proba_group = y_pred_proba[group_mask.values]
            approved = (y_proba_group < 0.5).mean()
            approval_rates[group_val] = approved
            print(f"    Group {group_val}: approval rate = {approved:.3f} ({group_mask.sum():,} loans)")
        
        if len(approval_rates) >= 2:
            rates = list(approval_rates.values())
            dp_gap = max(rates) - min(rates)
            di_ratio = min(rates) / max(rates) if max(rates) > 0 else 0
            passes = di_ratio >= 0.8
            status = "PASSES" if passes else "FAILS"
            print(f"    DP Gap: {dp_gap:.4f} | DI Ratio: {di_ratio:.4f} ({status} 4/5ths rule)")

# Save results
results_df = pd.DataFrame(regime_results)
results_df.to_csv("data/regime_credit_results.csv", index=False)
loans[["issue_d", "regime_name", "loan_status"]].to_csv("data/loans_with_regimes.csv", index=False)

print(f"\n  Saved results to data/regime_credit_results.csv")
print(f"  Saved loan-regime mapping to data/loans_with_regimes.csv")
print("\n" + "=" * 60)
print("  ANALYSIS COMPLETE")
print("=" * 60)
