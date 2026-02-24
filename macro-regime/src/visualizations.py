"""
Visualizations — Regime Analysis
==================================
Publication-quality charts for regime detection and credit risk analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
import os
from typing import Dict, Optional

# Style
plt.rcParams.update({
    "figure.figsize": (14, 6),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,

})

REGIME_COLORS = {
    "Expansion": "#2ecc71",    # Green
    "Contraction": "#f39c12",  # Orange
    "Crisis": "#e74c3c",       # Red
}


def ensure_fig_dir(path: str = "figures/"):
    os.makedirs(path, exist_ok=True)
    return path


def plot_regime_timeline(
    features: pd.DataFrame,
    regimes: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """
    Plot macro indicators with regime shading overlay.
    This is the hero chart — it shows regimes align with known events.
    """
    fig_dir = ensure_fig_dir()
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    
    dates = regimes.index
    regime_vals = regimes["regime_name"].values
    
    # Helper: shade regimes on axis
    def shade_regimes(ax):
        prev_regime = regime_vals[0]
        start_idx = 0
        for i in range(1, len(regime_vals)):
            if regime_vals[i] != prev_regime or i == len(regime_vals) - 1:
                color = REGIME_COLORS.get(prev_regime, "#cccccc")
                ax.axvspan(dates[start_idx], dates[i], 
                          alpha=0.15, color=color, linewidth=0)
                prev_regime = regime_vals[i]
                start_idx = i
    
    # Panel 1: Yield Curve Slope
    ax = axes[0]
    shade_regimes(ax)
    ax.plot(features.index, features["yield_curve_slope"], 
            color="#2c3e50", linewidth=0.8)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("Yield Curve Slope\n(10Y − 13W)")
    ax.set_title("Macroeconomic Regimes Detected by Hidden Markov Model", 
                 fontsize=14, fontweight="bold")
    ax.annotate("Inversion →\nRecession Signal", xy=(0.02, 0.05),
                xycoords="axes fraction", fontsize=8, color="red", alpha=0.7)
    
    # Panel 2: VIX
    ax = axes[1]
    shade_regimes(ax)
    ax.plot(features.index, features["vix_level"], 
            color="#8e44ad", linewidth=0.8)
    ax.axhline(y=20, color="gray", linestyle=":", linewidth=0.5)
    ax.axhline(y=30, color="orange", linestyle=":", linewidth=0.5)
    ax.set_ylabel("VIX\n(Smoothed)")
    
    # Panel 3: Credit Spread
    ax = axes[2]
    shade_regimes(ax)
    ax.plot(features.index, features["credit_spread_rolling"], 
            color="#e67e22", linewidth=0.8)
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.5)
    ax.set_ylabel("Credit Spread\n(HY − IG Rolling)")
    
    # Panel 4: S&P 500 Drawdown
    ax = axes[3]
    shade_regimes(ax)
    ax.fill_between(features.index, features["sp500_drawdown"], 0,
                    color="#3498db", alpha=0.4)
    ax.plot(features.index, features["sp500_drawdown"],
            color="#2c3e50", linewidth=0.5)
    ax.set_ylabel("S&P 500\nDrawdown")
    ax.set_xlabel("Date")
    
    # Add known events annotations on bottom panel
    events = {
        "2008-09-15": "Lehman\nBrothers",
        "2020-03-11": "COVID\nDeclared",
        "2022-03-16": "First Rate\nHike",
    }
    for date_str, label in events.items():
        try:
            dt = pd.Timestamp(date_str)
            if dt in features.index or dt >= features.index.min():
                axes[3].annotate(
                    label, xy=(dt, -0.05), fontsize=7,
                    ha="center", va="top", color="#7f8c8d",
                    arrowprops=dict(arrowstyle="-", color="#bdc3c7", lw=0.5),
                )
        except:
            pass
    
    # Legend
    legend_patches = [mpatches.Patch(color=c, alpha=0.3, label=n) 
                      for n, c in REGIME_COLORS.items()]
    axes[0].legend(handles=legend_patches, loc="upper right", 
                   fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    path = save_path or f"{fig_dir}regime_timeline.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_default_rates_by_regime(
    default_rates: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """Bar chart of default rates across regimes and demographic groups."""
    fig_dir = ensure_fig_dir()
    
    # Filter to group-level data (exclude "ALL")
    group_data = default_rates[default_rates["group"] != "ALL"]
    overall_data = default_rates[default_rates["group"] == "ALL"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Overall default rates by regime
    regimes = overall_data["regime"].values
    rates = overall_data["default_rate"].values
    colors = [REGIME_COLORS.get(r, "#95a5a6") for r in regimes]
    
    bars = ax1.bar(regimes, rates, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("Default Rate")
    ax1.set_title("Default Rate by Economic Regime", fontweight="bold")
    
    # Add value labels
    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{rate:.3f}", ha="center", va="bottom", fontsize=10)
    
    # Add multiplier annotation
    if len(rates) >= 2:
        multiplier = rates[-1] / rates[0] if rates[0] > 0 else 0
        ax1.annotate(f"{multiplier:.1f}× higher", 
                    xy=(2, rates[-1]), xytext=(1.5, rates[-1] * 1.15),
                    fontsize=10, fontweight="bold", color="#e74c3c",
                    arrowprops=dict(arrowstyle="->", color="#e74c3c"))
    
    # Right: Default rates by regime and group
    if len(group_data) > 0:
        pivot = group_data.pivot(index="regime", columns="group", values="default_rate")
        # Reorder regimes
        regime_order = ["Expansion", "Contraction", "Crisis"]
        pivot = pivot.reindex([r for r in regime_order if r in pivot.index])
        
        pivot.plot(kind="bar", ax=ax2, edgecolor="white", linewidth=0.5)
        ax2.set_ylabel("Default Rate")
        ax2.set_title("Default Rate by Regime × Demographic Group", fontweight="bold")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
        ax2.legend(title="Group", fontsize=9)
    
    plt.tight_layout()
    path = save_path or f"{fig_dir}default_rates_by_regime.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_model_performance_by_regime(
    performance: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """AUC and calibration error across regimes."""
    fig_dir = ensure_fig_dir()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    regimes = performance["regime"].values
    colors = [REGIME_COLORS.get(r, "#95a5a6") for r in regimes]
    
    # Left: AUC by regime
    bars = ax1.bar(regimes, performance["auc"], color=colors,
                   edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("AUC")
    ax1.set_title("Model AUC by Economic Regime", fontweight="bold")
    ax1.set_ylim(0.5, max(performance["auc"]) * 1.1)
    
    for bar, auc in zip(bars, performance["auc"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{auc:.3f}", ha="center", va="bottom", fontsize=10)
    
    # AUC drop annotation
    if len(performance) >= 2:
        drop = performance["auc"].iloc[0] - performance["auc"].iloc[-1]
        ax1.annotate(f"AUC drops {drop:.3f}\nduring Crisis",
                    xy=(len(regimes) - 1, performance["auc"].iloc[-1]),
                    xytext=(len(regimes) - 1.5, performance["auc"].iloc[-1] - 0.03),
                    fontsize=9, color="#e74c3c", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#e74c3c"))
    
    # Right: Calibration (predicted vs actual default rate)
    x = np.arange(len(regimes))
    width = 0.35
    
    ax2.bar(x - width / 2, performance["predicted_default_rate"], width,
            label="Predicted", color="#3498db", alpha=0.8)
    ax2.bar(x + width / 2, performance["actual_default_rate"], width,
            label="Actual", color="#e74c3c", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(regimes)
    ax2.set_ylabel("Default Rate")
    ax2.set_title("Model Calibration by Regime", fontweight="bold")
    ax2.legend()
    
    # Annotate calibration error
    for i, (_, row) in enumerate(performance.iterrows()):
        err = row["calibration_error"]
        if abs(err) > 0.005:
            ax2.annotate(f"{'Under' if err < 0 else 'Over'}-predicts\nby {abs(err):.3f}",
                        xy=(i + width / 2, row["actual_default_rate"]),
                        xytext=(i + 0.5, row["actual_default_rate"] + 0.02),
                        fontsize=8, color="#e74c3c" if err < 0 else "#f39c12")
    
    plt.tight_layout()
    path = save_path or f"{fig_dir}model_performance_by_regime.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_fairness_by_regime(
    fairness: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """Fairness metrics across regimes."""
    fig_dir = ensure_fig_dir()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    regimes = fairness["regime"].values
    colors = [REGIME_COLORS.get(r, "#95a5a6") for r in regimes]
    
    # Left: Demographic parity gap
    bars = ax1.bar(regimes, fairness["dp_gap"], color=colors,
                   edgecolor="white", linewidth=0.5)
    ax1.axhline(y=0.05, color="red", linestyle="--", linewidth=1,
                label="5% threshold")
    ax1.set_ylabel("Demographic Parity Gap")
    ax1.set_title("Fairness Gap by Economic Regime", fontweight="bold")
    ax1.legend()
    
    for bar, gap in zip(bars, fairness["dp_gap"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{gap:.3f}", ha="center", va="bottom", fontsize=10)
    
    # Right: Disparate impact ratio
    bars = ax2.bar(regimes, fairness["disparate_impact_ratio"], color=colors,
                   edgecolor="white", linewidth=0.5)
    ax2.axhline(y=0.8, color="red", linestyle="--", linewidth=1,
                label="4/5ths Rule (0.80)")
    ax2.set_ylabel("Disparate Impact Ratio")
    ax2.set_title("4/5ths Rule Compliance by Regime", fontweight="bold")
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    
    for bar, ratio, passes in zip(bars, fairness["disparate_impact_ratio"],
                                   fairness["passes_4_5_rule"]):
        color = "#2ecc71" if passes else "#e74c3c"
        label = "PASS" if passes else "FAIL"
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{ratio:.3f}\n{label}", ha="center", va="bottom",
                fontsize=9, color=color, fontweight="bold")
    
    plt.tight_layout()
    path = save_path or f"{fig_dir}fairness_by_regime.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_regime_transition_heatmap(
    trans_matrix: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """Heatmap of regime transition probabilities."""
    fig_dir = ensure_fig_dir()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(trans_matrix, annot=True, fmt=".3f", cmap="YlOrRd",
                ax=ax, vmin=0, vmax=1, linewidths=0.5,
                cbar_kws={"label": "Transition Probability"})
    
    ax.set_title("Regime Transition Matrix (Daily)", fontweight="bold")
    ax.set_ylabel("From Regime")
    ax.set_xlabel("To Regime")
    
    plt.tight_layout()
    path = save_path or f"{fig_dir}transition_matrix.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_financial_conditions_index(
    features: pd.DataFrame,
    regimes: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """Financial Conditions Index over time with regime overlay."""
    fig_dir = ensure_fig_dir()
    
    fig, ax = plt.subplots(figsize=(16, 5))
    
    # Shade regimes
    dates = regimes.index
    regime_vals = regimes["regime_name"].values
    prev = regime_vals[0]
    start = 0
    for i in range(1, len(regime_vals)):
        if regime_vals[i] != prev or i == len(regime_vals) - 1:
            color = REGIME_COLORS.get(prev, "#ccc")
            ax.axvspan(dates[start], dates[i], alpha=0.15, color=color)
            prev = regime_vals[i]
            start = i
    
    ax.plot(features.index, features["financial_conditions_index"],
            color="#2c3e50", linewidth=0.8)
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.5)
    ax.fill_between(features.index, features["financial_conditions_index"], 0,
                    where=features["financial_conditions_index"] > 0,
                    color="#e74c3c", alpha=0.2, label="Tight conditions")
    ax.fill_between(features.index, features["financial_conditions_index"], 0,
                    where=features["financial_conditions_index"] <= 0,
                    color="#2ecc71", alpha=0.2, label="Loose conditions")
    
    ax.set_title("Financial Conditions Index (Higher = Tighter = Worse for Borrowers)",
                 fontweight="bold")
    ax.set_ylabel("FCI (z-score composite)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    
    plt.tight_layout()
    path = save_path or f"{fig_dir}financial_conditions_index.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def generate_all_plots(
    features: pd.DataFrame,
    regimes: pd.DataFrame,
    analysis_results: Dict,
):
    """Generate all visualization charts."""
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_regime_timeline(features, regimes)
    plot_financial_conditions_index(features, regimes)
    
    if "default_rates" in analysis_results:
        plot_default_rates_by_regime(analysis_results["default_rates"])
    
    if "performance" in analysis_results:
        plot_model_performance_by_regime(analysis_results["performance"])
    
    if "fairness" in analysis_results:
        plot_fairness_by_regime(analysis_results["fairness"])
    
    print("\nAll visualizations saved to figures/")
