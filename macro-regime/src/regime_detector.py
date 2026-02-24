"""
Regime Detector — Hidden Markov Models
========================================
Identifies macroeconomic regimes (Expansion, Contraction, Crisis)
from financial market data using Gaussian Hidden Markov Models.

Why HMM:
    Markets don't announce when regimes change. The economy transitions
    gradually from expansion to contraction, and HMMs are designed to 
    infer hidden states from observable signals. The "hidden" state is 
    the economic regime; the "observable" signals are yield curves, 
    credit spreads, VIX, and equity momentum.

Validation approach:
    The detected regimes MUST align with known economic events:
    - 2001 Dot-Com recession
    - 2008-2009 Global Financial Crisis
    - 2020 COVID crash
    - 2022 rate shock
    If they don't, the model is wrong and needs retuning.
"""

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import joblib
import yaml
import os
import warnings
from typing import Tuple, Dict

warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_hmm_features(
    features: pd.DataFrame, config: dict
) -> Tuple[np.ndarray, StandardScaler, pd.DatetimeIndex]:
    """
    Select and standardize features for HMM input.
    
    Returns:
        X_scaled: (n_days, n_features) standardized feature matrix
        scaler:   fitted StandardScaler (needed for new data)
        dates:    DatetimeIndex aligned with X_scaled rows
    """
    hmm_feature_cols = config["regime"]["hmm_features"]
    
    # Verify all required features exist
    missing = [c for c in hmm_feature_cols if c not in features.columns]
    if missing:
        raise ValueError(f"Missing features for HMM: {missing}")
    
    X = features[hmm_feature_cols].copy()
    
    # Drop any remaining NaN rows
    X = X.dropna()
    dates = X.index
    
    # Standardize — HMM is sensitive to feature scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    
    print(f"HMM input: {X_scaled.shape[0]} days × {X_scaled.shape[1]} features")
    print(f"Features used: {hmm_feature_cols}")
    
    return X_scaled, scaler, dates


def fit_hmm(
    X: np.ndarray, config: dict, n_fits: int = 10
) -> GaussianHMM:
    """
    Fit Gaussian HMM with multiple random initializations.
    
    HMMs are sensitive to initialization, so we fit multiple times
    and keep the best (highest log-likelihood) model.
    
    Args:
        X: standardized feature matrix
        config: configuration dict
        n_fits: number of random restarts
    
    Returns:
        Best-fit GaussianHMM model
    """
    n_regimes = config["regime"]["n_regimes"]
    cov_type = config["regime"]["covariance_type"]
    n_iter = config["regime"]["n_iter"]
    seed = config["regime"]["random_state"]
    
    print(f"\nFitting {n_regimes}-regime Gaussian HMM ({n_fits} random restarts)...")
    
    best_model = None
    best_score = -np.inf
    
    for i in range(n_fits):
        model = GaussianHMM(
            n_components=n_regimes,
            covariance_type=cov_type,
            n_iter=n_iter,
            random_state=seed + i,
            verbose=False,
        )
        
        try:
            model.fit(X)
            score = model.score(X)
            
            if score > best_score:
                best_score = score
                best_model = model
                
        except Exception as e:
            # Some initializations may not converge
            continue
    
    if best_model is None:
        raise RuntimeError("HMM failed to converge on all restarts")
    
    print(f"Best log-likelihood: {best_score:.2f}")
    print(f"Converged: {best_model.monitor_.converged}")
    
    return best_model


def label_regimes(
    model: GaussianHMM,
    X: np.ndarray,
    dates: pd.DatetimeIndex,
    features: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Assign regime labels to each trading day and reorder regimes
    so they have economically meaningful names.
    
    The HMM assigns arbitrary labels (0, 1, 2). We reorder them
    based on the mean VIX level in each regime:
        - Lowest VIX mean  → "Expansion"
        - Middle VIX mean  → "Contraction"  
        - Highest VIX mean → "Crisis"
    
    This ensures regime labels are interpretable.
    """
    # Predict most likely regime sequence
    raw_labels = model.predict(X)
    
    # Get regime probabilities for each day
    raw_probs = model.predict_proba(X)
    
    # Build temporary DataFrame to compute regime statistics
    temp = pd.DataFrame({"raw_regime": raw_labels}, index=dates)
    temp["vix_level"] = features.loc[dates, "vix_level"].values
    
    # Compute mean VIX per raw regime
    vix_by_regime = temp.groupby("raw_regime")["vix_level"].mean().sort_values()
    
    # Create mapping: lowest VIX → 0 (Expansion), highest → 2 (Crisis)
    regime_order = {old: new for new, old in enumerate(vix_by_regime.index)}
    regime_labels = config["regime"]["regime_labels"]
    
    print("\nRegime identification (ordered by mean VIX):")
    print("-" * 50)
    for old_label, new_label in sorted(regime_order.items(), key=lambda x: x[1]):
        mean_vix = vix_by_regime[old_label]
        name = regime_labels[new_label]
        n_days = (raw_labels == old_label).sum()
        pct = n_days / len(raw_labels) * 100
        print(f"  {name}: mean VIX = {mean_vix:.1f}, {n_days} days ({pct:.1f}%)")
    
    # Apply mapping
    mapped_labels = np.array([regime_order[r] for r in raw_labels])
    
    # Build output DataFrame
    regimes = pd.DataFrame(index=dates)
    regimes["regime"] = mapped_labels
    regimes["regime_name"] = regimes["regime"].map(regime_labels)
    
    # Add regime probabilities (reordered)
    for new_label in sorted(regime_labels.keys()):
        name = regime_labels[new_label]
        # Find which original column maps to this new label
        old_label = [k for k, v in regime_order.items() if v == new_label][0]
        regimes[f"prob_{name.lower()}"] = raw_probs[:, old_label]
    
    return regimes


def validate_regimes(regimes: pd.DataFrame, config: dict) -> Dict:
    """
    Validate that detected regimes align with known economic events.
    
    This is the critical sanity check. If the model puts 2008 in
    "Expansion" or 2020 COVID in "Expansion", something is wrong.
    """
    scenarios = config["stress_testing"]["scenarios"]
    regime_labels = config["regime"]["regime_labels"]
    
    print("\n" + "=" * 60)
    print("REGIME VALIDATION — Known Historical Events")
    print("=" * 60)
    
    validation_results = {}
    
    for scenario_key, scenario in scenarios.items():
        start = pd.Timestamp(scenario["start"])
        end = pd.Timestamp(scenario["end"])
        name = scenario["name"]
        
        # Get regimes during this period
        mask = (regimes.index >= start) & (regimes.index <= end)
        period_regimes = regimes.loc[mask]
        
        if len(period_regimes) == 0:
            print(f"\n  {name}: No data in range (may predate available data)")
            continue
        
        # Dominant regime during this period
        regime_counts = period_regimes["regime_name"].value_counts()
        dominant = regime_counts.index[0]
        dominant_pct = regime_counts.iloc[0] / len(period_regimes) * 100
        
        # For crisis events, we expect "Crisis" or "Contraction" to dominate
        expected_stress = scenario_key in ["gfc_2008", "covid_2020", "rate_shock_2022", "dot_com_2001"]
        is_valid = dominant in ["Crisis", "Contraction"] if expected_stress else True
        
        validation_results[scenario_key] = {
            "name": name,
            "dominant_regime": dominant,
            "dominant_pct": dominant_pct,
            "regime_distribution": regime_counts.to_dict(),
            "valid": is_valid,
        }
        
        status = "✓" if is_valid else "✗ WARNING"
        print(f"\n  {status} {name} ({start.date()} to {end.date()}):")
        for regime_name, count in regime_counts.items():
            pct = count / len(period_regimes) * 100
            bar = "█" * int(pct / 2)
            print(f"      {regime_name:15s}: {pct:5.1f}% {bar}")
    
    n_valid = sum(1 for v in validation_results.values() if v["valid"])
    n_total = len(validation_results)
    print(f"\n  Validation: {n_valid}/{n_total} events correctly classified")
    
    if n_valid < n_total:
        print("  ⚠ Some events not correctly classified. Consider adjusting")
        print("    HMM features or number of regimes.")
    
    return validation_results


def compute_transition_matrix(regimes: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Compute empirical regime transition probabilities.
    
    This answers: "Given we're in Expansion today, what's the probability
    we transition to Contraction tomorrow?"
    
    Critical for stress testing — if you're in Expansion and want to 
    simulate a path to Crisis, the transition matrix tells you the 
    likelihood and typical path.
    """
    regime_labels = config["regime"]["regime_labels"]
    labels = list(regime_labels.values())
    n = len(labels)
    
    transitions = np.zeros((n, n))
    regime_vals = regimes["regime"].values
    
    for i in range(len(regime_vals) - 1):
        transitions[regime_vals[i], regime_vals[i + 1]] += 1
    
    # Normalize rows to probabilities
    row_sums = transitions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    trans_probs = transitions / row_sums
    
    trans_df = pd.DataFrame(trans_probs, index=labels, columns=labels)
    
    print("\nEmpirical Transition Matrix (daily):")
    print("-" * 50)
    print(trans_df.round(4).to_string())
    
    # Compute average regime duration
    print("\nAverage Regime Duration:")
    for i, label in enumerate(labels):
        # Expected duration = 1 / (1 - self-transition probability)
        self_prob = trans_probs[i, i]
        if self_prob < 1:
            avg_duration = 1 / (1 - self_prob)
            print(f"  {label}: {avg_duration:.0f} trading days ({avg_duration/21:.1f} months)")
    
    return trans_df


def save_regime_results(
    model: GaussianHMM,
    scaler: StandardScaler,
    regimes: pd.DataFrame,
    config: dict,
):
    """Save fitted model and regime labels."""
    os.makedirs("data", exist_ok=True)
    
    model_path = config["regime"]["model_path"]
    regimes_path = config["regime"]["regimes_path"]
    
    joblib.dump({"model": model, "scaler": scaler}, model_path)
    regimes.to_csv(regimes_path)
    
    print(f"\nSaved HMM model to {model_path}")
    print(f"Saved regime labels to {regimes_path}")


def detect_regimes(
    features: pd.DataFrame, config_path: str = "configs/config.yaml"
) -> pd.DataFrame:
    """
    Full regime detection pipeline.
    
    Args:
        features: DataFrame from data_pipeline.engineer_features()
        config_path: path to YAML config
    
    Returns:
        DataFrame with regime labels and probabilities for each trading day.
    """
    config = load_config(config_path)
    
    print("\n" + "=" * 60)
    print("REGIME DETECTION — Hidden Markov Model")
    print("=" * 60)
    
    # Step 1: Prepare features
    print("\n[1/5] Preparing HMM features...")
    X, scaler, dates = prepare_hmm_features(features, config)
    
    # Step 2: Fit HMM
    print("\n[2/5] Fitting HMM...")
    model = fit_hmm(X, config)
    
    # Step 3: Label regimes
    print("\n[3/5] Labeling regimes...")
    regimes = label_regimes(model, X, dates, features, config)
    
    # Step 4: Validate
    print("\n[4/5] Validating against known events...")
    validation = validate_regimes(regimes, config)
    
    # Step 5: Transition matrix
    print("\n[5/5] Computing transition matrix...")
    trans_matrix = compute_transition_matrix(regimes, config)
    
    # Save
    save_regime_results(model, scaler, regimes, config)
    
    print("\n" + "=" * 60)
    print("REGIME DETECTION COMPLETE")
    print("=" * 60)
    
    return regimes


if __name__ == "__main__":
    config = load_config()
    features = pd.read_csv(
        config["data"]["features_path"], index_col=0, parse_dates=True
    )
    regimes = detect_regimes(features)
