"""
Data pipeline for Lending Club loan data.
Handles downloading, cleaning, feature engineering, and train/test splitting.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from features import engineer_features

warnings.filterwarnings("ignore")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_data(output_path: str) -> str:
    """
    Download Lending Club dataset from Kaggle.
    Requires: kaggle API credentials in ~/.kaggle/kaggle.json
    
    Alternative: Download manually from
    https://www.kaggle.com/datasets/wordsforthewise/lending-club
    and place in data/raw/
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            "wordsforthewise/lending-club",
            path=os.path.dirname(output_path),
            unzip=True,
        )
        print(f"Data downloaded to {output_path}")
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print("\nAlternative: Download manually from:")
        print("https://www.kaggle.com/datasets/wordsforthewise/lending-club")
        print(f"Place the CSV in: {output_path}")
        sys.exit(1)

    return output_path


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw Lending Club data.
    
    Steps:
    - Filter to completed loans (Fully Paid or Charged Off)
    - Remove columns with >40% missing values
    - Handle missing values in remaining columns
    - Convert data types
    - Remove outliers using IQR method on key numerical columns
    """
    print(f"Raw data shape: {df.shape}")

    # Filter to completed loans only — we need known outcomes
    valid_statuses = ["Fully Paid", "Charged Off"]
    df = df[df["loan_status"].isin(valid_statuses)].copy()
    print(f"After filtering to completed loans: {df.shape}")

    # Create binary target: 1 = default (Charged Off), 0 = paid
    df["loan_status"] = (df["loan_status"] == "Charged Off").astype(int)

    # Drop columns with >40% missing
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > 0.4].index.tolist()
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} columns with >40% missing values")

    # Convert interest rate from string to float
    if df["int_rate"].dtype == object:
        df["int_rate"] = df["int_rate"].str.rstrip("%").astype(float)

    # Convert term to numeric
    if df["term"].dtype == object:
        df["term"] = df["term"].str.strip().str.split().str[0].astype(int)

    # Clean employment length
    if "emp_length" in df.columns:
        emp_map = {
            "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
            "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
            "8 years": 8, "9 years": 9, "10+ years": 10,
        }
        df["emp_length"] = df["emp_length"].map(emp_map)

    # Fill numerical missing values with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values with mode
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Remove extreme outliers in annual income (top 1%)
    income_cap = df["annual_inc"].quantile(0.99)
    df = df[df["annual_inc"] <= income_cap]

    print(f"Cleaned data shape: {df.shape}")
    print(f"Default rate: {df['loan_status'].mean():.3f}")

    return df


def select_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Select and prepare features specified in config."""
    numerical = config["features"]["numerical"]
    categorical = config["features"]["categorical"]

    # Keep only columns that exist in the dataframe
    available_numerical = [c for c in numerical if c in df.columns]
    available_categorical = [c for c in categorical if c in df.columns]

    target = config["data"]["target_column"]
    keep_cols = available_numerical + available_categorical + [target]

    df_selected = df[keep_cols].copy()

    # One-hot encode categoricals (keep protected attribute separate)
    protected = config["fairness"]["protected_attribute"]
    encode_cols = [c for c in available_categorical if c != protected]

    df_encoded = pd.get_dummies(df_selected, columns=encode_cols, drop_first=True)

    # Label encode protected attribute (keep as-is for fairness analysis)
    if protected in df_encoded.columns:
        df_encoded[f"{protected}_original"] = df_encoded[protected].copy()
        df_encoded[protected] = df_encoded[protected].astype("category").cat.codes

    print(f"Final feature matrix shape: {df_encoded.shape}")
    return df_encoded


def split_data(
    df: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test with stratification on both
    target and protected attribute for fair representation.
    """
    target = config["data"]["target_column"]
    protected = config["fairness"]["protected_attribute"]
    test_size = config["data"]["test_size"]
    val_size = config["data"]["val_size"]
    seed = config["data"]["random_state"]

    # Create stratification key combining target and protected attribute
    df["_strat_key"] = df[target].astype(str) + "_" + df[protected].astype(str)

    # First split: train+val vs test
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["_strat_key"], random_state=seed
    )

    # Second split: train vs val
    adjusted_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=adjusted_val_size,
        stratify=train_val["_strat_key"],
        random_state=seed,
    )

    # Drop stratification key
    for split in [train, val, test]:
        split.drop(columns=["_strat_key"], inplace=True)

    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
    print(f"Train default rate: {train[target].mean():.3f}")
    print(f"Test default rate:  {test[target].mean():.3f}")

    return train, val, test


def run_pipeline(config_path: str = "configs/config.yaml", download: bool = False):
    """Execute the full data pipeline."""
    config = load_config(config_path)

    raw_path = config["data"]["raw_path"]
    processed_path = config["data"]["processed_path"]
    os.makedirs(processed_path, exist_ok=True)

    # Step 1: Download data if needed
    if download or not os.path.exists(raw_path):
        download_data(raw_path)

    # Step 2: Load raw data
    print("\n--- Loading raw data ---")
    df = pd.read_csv(raw_path, low_memory=False)
    print(f"Loaded {len(df):,} rows")

    # Step 3: Clean
    print("\n--- Cleaning data ---")
    df = clean_data(df)

    # Step 4: Feature engineering
    print("\n--- Engineering features ---")
    df = engineer_features(df)

    # Step 5: Select and encode features
    print("\n--- Selecting features ---")
    df = select_features(df, config)

    # Step 6: Split
    print("\n--- Splitting data ---")
    train, val, test = split_data(df, config)

    # Step 7: Save
    train.to_parquet(os.path.join(processed_path, "train.parquet"), index=False)
    val.to_parquet(os.path.join(processed_path, "val.parquet"), index=False)
    test.to_parquet(os.path.join(processed_path, "test.parquet"), index=False)
    print(f"\nSaved processed data to {processed_path}")

    return train, val, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Config path")
    parser.add_argument("--download", action="store_true", help="Download data from Kaggle")
    args = parser.parse_args()

    run_pipeline(config_path=args.config, download=args.download)
