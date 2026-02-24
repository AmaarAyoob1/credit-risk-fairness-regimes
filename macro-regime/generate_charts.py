"""
Generate all charts for the Macro-Regime Credit Risk Analysis.
Run after run_bridge.py to create publication-quality visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

plt.rcParams.update({
    "figure.figsize": (14, 6),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,
})

REGIME_COLORS = {
    "Expansion": "#2ecc71",
    "Contraction": "#f39c12",
    "Crisis": "#e74c3c",
}

# Load data
print("Loading data...")
results = pd.read_csv("data/regime_credit_results.csv")
loans = pd.read_csv("data/loans_with_regimes.csv", parse_dates=["issue_d"])
regimes = pd.read_csv("data/regime_labels.csv", index_col=0, parse_dates=True)
features = pd.read_csv("data/macro_features.csv", index_col=0, parse_dates=True)

import os
os.makedirs("figures", exist_ok=True)

# ================================================================
# CHART 1: Default Rate by Regime
# ================================================================
print("Generating Chart 1: Default Rates by Regime...")
fig, ax = plt.subplots(figsize=(10, 6))

regimes_order = ["Expansion", "Contraction", "Crisis"]
default_rates = []
n_loans = []
for r in regimes_order:
    mask = loans["regime_name"] == r
    default_rates.append(loans.loc[mask, "loan_status"].mean())
    n_loans.append(mask.sum())

colors = [REGIME_COLORS[r] for r in regimes_order]
bars = ax.bar(regimes_order, default_rates, color=colors, edgecolor="white", linewidth=1.5)

for bar, rate, n in zip(bars, default_rates, n_loans):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{rate:.3f}\n({n:,} loans)", ha="center", va="bottom", fontsize=11)

multiplier = default_rates[2] / default_rates[0]
ax.annotate(f"{multiplier:.2f}× higher", xy=(2, default_rates[2]),
            xytext=(1.5, default_rates[2] + 0.02), fontsize=12, fontweight="bold",
            color="#e74c3c", arrowprops=dict(arrowstyle="->", color="#e74c3c"))

ax.set_ylabel("Default Rate", fontsize=12)
ax.set_title("Loan Default Rate by Economic Regime", fontsize=14, fontweight="bold")
ax.set_ylim(0, max(default_rates) * 1.25)
plt.tight_layout()
plt.savefig("figures/default_rates_by_regime.png")
plt.close()
print("  Saved: figures/default_rates_by_regime.png")

# ================================================================
# CHART 2: Model AUC by Regime
# ================================================================
print("Generating Chart 2: Model AUC by Regime...")
fig, ax = plt.subplots(figsize=(10, 6))

aucs = results["auc"].values
bars = ax.bar(regimes_order, aucs, color=colors, edgecolor="white", linewidth=1.5)

for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{auc:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

auc_drop = aucs[0] - aucs[2]
ax.annotate(f"AUC drops {auc_drop:.4f}\nduring Crisis", xy=(2, aucs[2]),
            xytext=(1.3, aucs[2] - 0.02), fontsize=11, fontweight="bold",
            color="#e74c3c", arrowprops=dict(arrowstyle="->", color="#e74c3c"))

ax.set_ylabel("AUC (Area Under ROC Curve)", fontsize=12)
ax.set_title("Model Discriminative Power by Economic Regime", fontsize=14, fontweight="bold")
ax.set_ylim(0.65, max(aucs) * 1.05)
ax.axhline(y=0.7, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
ax.text(2.5, 0.701, "AUC = 0.70 baseline", fontsize=8, color="gray")
plt.tight_layout()
plt.savefig("figures/model_auc_by_regime.png")
plt.close()
print("  Saved: figures/model_auc_by_regime.png")

# ================================================================
# CHART 3: Calibration (Predicted vs Actual Default Rate)
# ================================================================
print("Generating Chart 3: Model Calibration by Regime...")
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(regimes_order))
width = 0.35

bars1 = ax.bar(x - width/2, results["predicted_default_rate"], width,
               label="Predicted Default Rate", color="#3498db", alpha=0.85, edgecolor="white")
bars2 = ax.bar(x + width/2, results["actual_default_rate"], width,
               label="Actual Default Rate", color="#e74c3c", alpha=0.85, edgecolor="white")

for i, (pred, actual) in enumerate(zip(results["predicted_default_rate"], results["actual_default_rate"])):
    ax.text(i - width/2, pred + 0.002, f"{pred:.4f}", ha="center", va="bottom", fontsize=9)
    ax.text(i + width/2, actual + 0.002, f"{actual:.4f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(regimes_order, fontsize=12)
ax.set_ylabel("Default Rate", fontsize=12)
ax.set_title("Model Calibration: Predicted vs Actual Default Rate by Regime",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.set_ylim(0, max(results["actual_default_rate"].max(), results["predicted_default_rate"].max()) * 1.2)
plt.tight_layout()
plt.savefig("figures/calibration_by_regime.png")
plt.close()
print("  Saved: figures/calibration_by_regime.png")

# ================================================================
# CHART 4: Default Rate Over Time with Regime Shading
# ================================================================
print("Generating Chart 4: Default Rate Timeline with Regime Shading...")
fig, ax = plt.subplots(figsize=(16, 6))

# Monthly default rate
loans_sorted = loans.sort_values("issue_d")
loans_sorted["month"] = loans_sorted["issue_d"].dt.to_period("M")
monthly = loans_sorted.groupby("month").agg(
    default_rate=("loan_status", "mean"),
    n_loans=("loan_status", "count"),
).reset_index()
monthly["month_dt"] = monthly["month"].dt.to_timestamp()

# Shade regimes
regime_vals = regimes["regime_name"].values
regime_dates = regimes.index
prev = regime_vals[0]
start = 0
for i in range(1, len(regime_vals)):
    if regime_vals[i] != prev or i == len(regime_vals) - 1:
        color = REGIME_COLORS.get(prev, "#ccc")
        ax.axvspan(regime_dates[start], regime_dates[i], alpha=0.15, color=color)
        prev = regime_vals[i]
        start = i

ax.plot(monthly["month_dt"], monthly["default_rate"], color="#2c3e50", linewidth=1.5)
ax.scatter(monthly["month_dt"], monthly["default_rate"], s=15, color="#2c3e50", zorder=5)

ax.set_ylabel("Monthly Default Rate", fontsize=12)
ax.set_xlabel("Loan Issue Date", fontsize=12)
ax.set_title("Loan Default Rate Over Time with Economic Regime Overlay",
             fontsize=14, fontweight="bold")

legend_patches = [mpatches.Patch(color=c, alpha=0.3, label=n) for n, c in REGIME_COLORS.items()]
ax.legend(handles=legend_patches, loc="upper right", fontsize=10)

plt.tight_layout()
plt.savefig("figures/default_rate_timeline.png")
plt.close()
print("  Saved: figures/default_rate_timeline.png")

# ================================================================
# CHART 5: Loan Volume by Regime Over Time
# ================================================================
print("Generating Chart 5: Loan Volume by Regime...")
fig, ax = plt.subplots(figsize=(14, 5))

for regime in regimes_order:
    mask = loans_sorted["regime_name"] == regime
    regime_monthly = loans_sorted[mask].groupby(
        loans_sorted.loc[mask, "issue_d"].dt.to_period("M")
    ).size().reset_index(name="count")
    regime_monthly["month_dt"] = regime_monthly["issue_d"].dt.to_timestamp()
    ax.bar(regime_monthly["month_dt"], regime_monthly["count"],
           width=25, color=REGIME_COLORS[regime], alpha=0.7, label=regime)

ax.set_ylabel("Number of Loans Issued", fontsize=12)
ax.set_xlabel("Date", fontsize=12)
ax.set_title("Loan Origination Volume by Economic Regime", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("figures/loan_volume_by_regime.png")
plt.close()
print("  Saved: figures/loan_volume_by_regime.png")

# ================================================================
# CHART 6: Combined Dashboard
# ================================================================
print("Generating Chart 6: Combined Dashboard...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top-left: Default rates
ax = axes[0, 0]
bars = ax.bar(regimes_order, default_rates, color=colors, edgecolor="white")
for bar, rate in zip(bars, default_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{rate:.3f}", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Default Rate")
ax.set_title("Default Rate by Regime", fontweight="bold")

# Top-right: AUC
ax = axes[0, 1]
bars = ax.bar(regimes_order, aucs, color=colors, edgecolor="white")
for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{auc:.4f}", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("AUC")
ax.set_title("Model AUC by Regime", fontweight="bold")
ax.set_ylim(0.65, max(aucs) * 1.05)

# Bottom-left: Calibration
ax = axes[1, 0]
x = np.arange(len(regimes_order))
width = 0.35
ax.bar(x - width/2, results["predicted_default_rate"], width,
       label="Predicted", color="#3498db", alpha=0.85)
ax.bar(x + width/2, results["actual_default_rate"], width,
       label="Actual", color="#e74c3c", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(regimes_order)
ax.set_ylabel("Default Rate")
ax.set_title("Calibration: Predicted vs Actual", fontweight="bold")
ax.legend(fontsize=9)

# Bottom-right: Regime distribution pie
ax = axes[1, 1]
sizes = [n for n in n_loans]
ax.pie(sizes, labels=regimes_order, colors=colors, autopct="%1.1f%%",
       startangle=90, textprops={"fontsize": 11})
ax.set_title("Loan Distribution by Regime", fontweight="bold")

fig.suptitle("Macro-Regime Credit Risk Analysis Dashboard",
             fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("figures/combined_dashboard.png")
plt.close()
print("  Saved: figures/combined_dashboard.png")

print("\n" + "=" * 60)
print("ALL CHARTS GENERATED")
print("=" * 60)
print("\nOpen all charts:")
print("  open figures/default_rates_by_regime.png")
print("  open figures/model_auc_by_regime.png")
print("  open figures/calibration_by_regime.png")
print("  open figures/default_rate_timeline.png")
print("  open figures/loan_volume_by_regime.png")
print("  open figures/combined_dashboard.png")
