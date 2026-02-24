"""
Macro Data Pipeline — Yahoo Finance
=====================================
Pulls macro indicators, engineers features, and outputs a clean
daily-frequency dataset ready for regime detection.

Features engineered:
    - yield_curve_slope:     10Y Treasury minus 13W T-Bill
    - yield_curve_mid:       10Y minus 5Y (mid-curve flattening)
    - credit_spread_rolling: Rolling spread between HY and IG bond returns
    - vix_level:             VIX index level (smoothed)
    - sp500_return_60d:      60-day rolling return on S&P 500
    - sp500_volatility:      60-day rolling realized volatility
    - gold_momentum:         60-day gold return (flight to safety signal)
    - dollar_momentum:       60-day USD index return
"""

import pandas as pd
import numpy as np
import yfinance as yf
import yaml
import os
from pathlib import Path
from datetime import datetime


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def pull_raw_data(config: dict) -> pd.DataFrame:
    """
    Pull raw daily data from Yahoo Finance for all configured tickers.
    
    Returns:
        DataFrame with MultiIndex columns (ticker, OHLCV field)
    """
    tickers_map = config["data"]["tickers"]
    ticker_symbols = list(tickers_map.values())
    start = config["data"]["start_date"]
    end = config["data"]["end_date"]
    
    print(f"Pulling data for {len(ticker_symbols)} tickers: {ticker_symbols}")
    print(f"Date range: {start} to {end}")
    
    # Download all tickers at once (more efficient)
    raw = yf.download(
        tickers=ticker_symbols,
        start=start,
        end=end,
        auto_adjust=True,
        threads=True,
    )
    
    print(f"Raw data shape: {raw.shape}")
    print(f"Date range received: {raw.index.min()} to {raw.index.max()}")
    
    return raw


def engineer_features(raw: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Transform raw OHLCV data into macro indicator features.
    
    This is where domain knowledge matters. Every feature has an
    economic rationale for why it predicts credit conditions.
    """
    tickers = config["data"]["tickers"]
    window = config["regime"]["rolling_window"]  # 21 trading days ≈ 1 month
    
    features = pd.DataFrame(index=raw.index)
    
    # -------------------------------------------------------------------------
    # 1. YIELD CURVE SLOPE (most important recession predictor)
    #    10Y Treasury minus 13W T-Bill
    #    Positive = normal economy, Negative = inverted = recession coming
    #    Economic logic: banks borrow short and lend long. When curve inverts,
    #    banking becomes unprofitable → credit tightening → defaults rise.
    # -------------------------------------------------------------------------
    tnx = raw["Close"][tickers["treasury_10y"]]
    irx = raw["Close"][tickers["treasury_13w"]]
    features["yield_curve_slope"] = tnx - irx
    
    # Also compute mid-curve (10Y - 5Y) for nuance
    fvx = raw["Close"][tickers["treasury_5y"]]
    features["yield_curve_mid"] = tnx - fvx
    
    # -------------------------------------------------------------------------
    # 2. CREDIT SPREAD (direct measure of default risk pricing)
    #    Spread between high-yield and investment-grade bond ETF returns.
    #    When this widens, the market is pricing in higher default probability.
    #    Economic logic: HYG holds junk bonds. When default fear rises,
    #    junk bonds sell off harder than IG bonds → spread widens.
    # -------------------------------------------------------------------------
    hyg_price = raw["Close"][tickers["high_yield"]]
    lqd_price = raw["Close"][tickers["inv_grade"]]
    
    # Daily return spread
    hyg_ret = hyg_price.pct_change()
    lqd_ret = lqd_price.pct_change()
    daily_spread = hyg_ret - lqd_ret
    
    # Rolling cumulative spread (smoothed signal)
    features["credit_spread_rolling"] = daily_spread.rolling(window).sum()
    
    # Also store raw price ratio for visualization
    features["hyg_lqd_ratio"] = hyg_price / lqd_price
    
    # -------------------------------------------------------------------------
    # 3. VIX (market fear gauge)
    #    Higher VIX = more uncertainty = tighter credit conditions
    #    Economic logic: high VIX means options market expects large moves.
    #    Banks respond by tightening lending standards → harder for borrowers
    #    to refinance → existing loans default more.
    # -------------------------------------------------------------------------
    vix = raw["Close"][tickers["vix"]]
    features["vix_level"] = vix.rolling(window).mean()  # Smoothed
    features["vix_spike"] = vix / vix.rolling(60).mean()  # Relative to 3-month avg
    
    # -------------------------------------------------------------------------
    # 4. S&P 500 (equity market health)
    #    Falling equity market → wealth effect → reduced consumer spending
    #    and harder to service debt.
    # -------------------------------------------------------------------------
    spy = raw["Close"][tickers["sp500"]]
    features["sp500_return_60d"] = spy.pct_change(60)  # 3-month momentum
    features["sp500_volatility"] = spy.pct_change().rolling(60).std() * np.sqrt(252)
    features["sp500_drawdown"] = spy / spy.rolling(252).max() - 1  # Distance from 1Y high
    
    # -------------------------------------------------------------------------
    # 5. GOLD & DOLLAR (flight to safety indicators)
    #    Rising gold + rising dollar = risk-off environment
    #    Economic logic: when investors flee to safety, credit conditions
    #    tighten for risky borrowers.
    # -------------------------------------------------------------------------
    gld = raw["Close"][tickers["gold"]]
    features["gold_momentum"] = gld.pct_change(60)  # 3-month gold return
    
    dxy = raw["Close"][tickers["dollar"]]
    features["dollar_momentum"] = dxy.pct_change(60)  # 3-month dollar return
    
    # -------------------------------------------------------------------------
    # 6. COMPOSITE INDICATORS
    # -------------------------------------------------------------------------
    # Financial Conditions Index (simple equal-weighted z-score composite)
    # Combines yield curve, credit spread, VIX, and equity momentum
    fci_components = ["yield_curve_slope", "credit_spread_rolling", 
                      "vix_level", "sp500_return_60d"]
    
    fci_z = features[fci_components].apply(
        lambda x: (x - x.rolling(252).mean()) / x.rolling(252).std()
    )
    # Flip signs so that higher = tighter conditions (worse for borrowers)
    fci_z["yield_curve_slope"] *= -1    # Inverted curve = tight
    fci_z["sp500_return_60d"] *= -1     # Falling market = tight
    # credit_spread_rolling and vix already directionally correct
    
    features["financial_conditions_index"] = fci_z.mean(axis=1)
    
    # -------------------------------------------------------------------------
    # 7. RATE ENVIRONMENT
    # -------------------------------------------------------------------------
    features["rate_level_10y"] = tnx.rolling(window).mean()
    features["rate_change_6m"] = tnx - tnx.shift(126)  # 6-month rate change
    
    # Drop NaN rows from rolling computations
    features = features.dropna()
    
    print(f"\nEngineered {features.shape[1]} features over {features.shape[0]} trading days")
    print(f"Date range: {features.index.min().date()} to {features.index.max().date()}")
    print(f"\nFeature summary:")
    print(features.describe().round(4).to_string())
    
    return features


def save_data(raw: pd.DataFrame, features: pd.DataFrame, config: dict):
    """Save raw data and features to CSV."""
    os.makedirs("data", exist_ok=True)
    
    raw_path = config["data"]["raw_data_path"]
    feat_path = config["data"]["features_path"]
    
    raw.to_csv(raw_path)
    features.to_csv(feat_path)
    
    print(f"\nSaved raw data to {raw_path}")
    print(f"Saved features to {feat_path}")


def run_pipeline(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """
    Full pipeline: pull data → engineer features → save.
    
    Returns:
        DataFrame of engineered macro features (daily frequency).
    """
    config = load_config(config_path)
    
    print("=" * 60)
    print("MACRO DATA PIPELINE")
    print("=" * 60)
    
    # Step 1: Pull raw data from Yahoo Finance
    print("\n[1/3] Pulling raw data from Yahoo Finance...")
    raw = pull_raw_data(config)
    
    # Step 2: Engineer features
    print("\n[2/3] Engineering macro features...")
    features = engineer_features(raw, config)
    
    # Step 3: Save
    print("\n[3/3] Saving data...")
    save_data(raw, features, config)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    return features


if __name__ == "__main__":
    features = run_pipeline()
