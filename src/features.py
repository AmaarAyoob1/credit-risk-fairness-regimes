"""
Feature engineering for loan default prediction.
Transforms raw loan attributes into predictive features.
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features from raw loan data.
    
    Returns DataFrame with new feature columns added.
    """
    df = df.copy()

    # --- Financial Ratio Features ---

    # Debt-to-income ratio (enhanced — accounts for the new loan)
    df["debt_to_income_ratio"] = np.where(
        df["annual_inc"] > 0,
        (df["installment"] * 12 + df["revol_bal"]) / df["annual_inc"],
        0,
    )

    # Credit utilization ratio
    df["credit_utilization"] = np.where(
        df["revol_util"].notna(), df["revol_util"] / 100, 0
    )

    # Income-to-loan ratio — how much of annual income this loan represents
    df["income_to_loan_ratio"] = np.where(
        df["loan_amnt"] > 0, df["annual_inc"] / df["loan_amnt"], 0
    )

    # Monthly payment burden — installment as % of monthly income
    monthly_income = df["annual_inc"] / 12
    df["monthly_burden_ratio"] = np.where(
        monthly_income > 0, df["installment"] / monthly_income, 0
    )

    # --- Credit History Features ---

    # Credit history length proxy — total accounts / open accounts
    df["credit_history_length"] = np.where(
        df["open_acc"] > 0, df["total_acc"] / df["open_acc"], 0
    )

    # Derogatory records flag — any public records or bankruptcies
    df["derogatory_flag"] = (
        (df["pub_rec"] > 0) | (df["pub_rec_bankruptcies"] > 0)
    ).astype(int)

    # High revolving balance flag — above 75th percentile
    revol_75 = df["revol_bal"].quantile(0.75)
    df["high_revolving_flag"] = (df["revol_bal"] > revol_75).astype(int)

    # --- Risk Indicator Features ---

    # Interest rate bucket — higher rates indicate lender-assessed risk
    df["rate_bucket"] = pd.cut(
        df["int_rate"],
        bins=[0, 8, 12, 16, 20, 35],
        labels=[0, 1, 2, 3, 4],
    ).astype(float)

    # Account utilization — open accounts vs total accounts
    df["account_utilization"] = np.where(
        df["total_acc"] > 0, df["open_acc"] / df["total_acc"], 0
    )

    # Loan amount per open account — concentration of debt
    df["loan_per_account"] = np.where(
        df["open_acc"] > 0, df["loan_amnt"] / df["open_acc"], 0
    )

    # --- Interaction Features ---

    # High DTI + High interest rate interaction
    df["high_dti_high_rate"] = (
        (df["dti"] > df["dti"].quantile(0.75))
        & (df["int_rate"] > df["int_rate"].quantile(0.75))
    ).astype(int)

    # Low income + High loan amount interaction
    df["low_income_high_loan"] = (
        (df["annual_inc"] < df["annual_inc"].quantile(0.25))
        & (df["loan_amnt"] > df["loan_amnt"].quantile(0.75))
    ).astype(int)

    # Cap extreme values in engineered features
    for col in ["debt_to_income_ratio", "income_to_loan_ratio", "credit_history_length"]:
        cap = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=cap)

    print(f"Engineered {12} new features")
    return df


def get_feature_names(config: dict) -> list[str]:
    """Return list of all feature names (numerical + engineered + encoded categorical)."""
    return (
        config["features"]["numerical"]
        + config["features"]["engineered"]
    )
