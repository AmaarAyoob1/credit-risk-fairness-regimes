"""
SHAP-based model explanations.
Provides global feature importance, local explanations, and
group-level analysis to understand if features affect groups differently.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def compute_shap_values(model, X: pd.DataFrame, max_samples: int = 5000) -> shap.Explanation:
    """
    Compute SHAP values using TreeExplainer.
    
    Subsamples data if too large (SHAP can be slow on large datasets).
    """
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    return shap_values, X_sample


def plot_global_importance(
    shap_values: shap.Explanation,
    output_path: str = "results/shap/global_importance.png",
    top_n: int = 15,
):
    """
    Plot global SHAP feature importance (bar chart).
    Shows which features matter most overall.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=top_n, show=False)
    plt.title("Global Feature Importance (SHAP)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved global importance plot to {output_path}")


def plot_beeswarm(
    shap_values: shap.Explanation,
    output_path: str = "results/shap/beeswarm.png",
    top_n: int = 15,
):
    """
    Plot SHAP beeswarm — shows feature impact distribution.
    Each dot is a single prediction, colored by feature value.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=top_n, show=False)
    plt.title("Feature Impact Distribution (SHAP)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved beeswarm plot to {output_path}")


def explain_single_prediction(
    model,
    X_instance: pd.DataFrame,
    output_path: str = None,
) -> dict:
    """
    Generate SHAP explanation for a single loan application.
    
    Returns dict with feature contributions and prediction.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_instance)

    prediction = model.predict_proba(X_instance)[:, 1][0]

    # Get top contributing features
    feature_contributions = pd.DataFrame({
        "feature": X_instance.columns,
        "value": X_instance.values[0],
        "shap_value": shap_values.values[0],
    }).sort_values("shap_value", key=abs, ascending=False)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.figure(figsize=(10, 4))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        plt.title(f"Prediction Explanation (P(default) = {prediction:.3f})")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    return {
        "default_probability": prediction,
        "decision": "DENY" if prediction >= 0.5 else "APPROVE",
        "top_factors": feature_contributions.head(10).to_dict("records"),
    }


def group_shap_analysis(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    protected: np.ndarray,
    output_path: str = "results/shap/group_analysis.png",
    top_n: int = 10,
):
    """
    Compare SHAP feature importance ACROSS demographic groups.
    
    This reveals if the model relies on different features for different groups,
    which could indicate indirect discrimination.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    groups = np.unique(protected)
    group_importance = {}

    for g in groups:
        mask = protected == g
        if mask.sum() == 0:
            continue
        group_shap = np.abs(shap_values.values[mask]).mean(axis=0)
        group_importance[g] = pd.Series(group_shap, index=X.columns)

    importance_df = pd.DataFrame(group_importance)

    # Plot comparison for top features
    top_features = importance_df.mean(axis=1).nlargest(top_n).index

    fig, ax = plt.subplots(figsize=(12, 6))
    importance_df.loc[top_features].plot(kind="barh", ax=ax)
    ax.set_title("Feature Importance by Group (SHAP)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean |SHAP Value|")
    ax.legend(title="Group")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved group analysis plot to {output_path}")

    return importance_df
