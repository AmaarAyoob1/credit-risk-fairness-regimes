"""
Interactive Streamlit dashboard for Fair Lending analysis.

Run with: streamlit run streamlit_app/app.py
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fairness import (
    run_fairness_audit,
    compute_group_thresholds,
    demographic_parity,
    equal_opportunity,
)
from explain import compute_shap_values, explain_single_prediction


# --- Page Config ---
st.set_page_config(
    page_title="Fair Lending Dashboard",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Fair Lending: Loan Default Prediction with Fairness Constraints")
st.markdown("Explore the tradeoff between model accuracy and demographic fairness in credit decisions.")


# --- Load Data & Models ---
@st.cache_resource
def load_artifacts():
    """Load trained models and test data."""
    models = {}
    model_path = "models/"

    for fname in os.listdir(model_path):
        if fname.endswith(".pkl"):
            name = fname.replace(".pkl", "")
            with open(os.path.join(model_path, fname), "rb") as f:
                models[name] = pickle.load(f)

    test = pd.read_parquet("data/processed/test.parquet")
    return models, test


try:
    models, test_data = load_artifacts()
except FileNotFoundError:
    st.error("Models or data not found. Run `python src/train.py` first.")
    st.stop()

# Prepare data
target_col = "loan_status"
protected_col = "home_ownership"
protected_original = f"{protected_col}_original"

drop_cols = [c for c in [target_col, protected_original] if c in test_data.columns]
X_test = test_data.drop(columns=drop_cols)
y_test = test_data[target_col].values
g_test = test_data[protected_col].values

group_names = {}
if protected_original in test_data.columns:
    mapping = test_data[[protected_col, protected_original]].drop_duplicates()
    group_names = dict(zip(mapping[protected_col], mapping[protected_original]))


# --- Sidebar ---
st.sidebar.header("Controls")

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys()),
    format_func=lambda x: x.replace("_", " ").title(),
)

threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.01,
    help="Probability threshold for classifying a loan as default",
)

use_fair_thresholds = st.sidebar.checkbox(
    "Apply Per-Group Fair Thresholds",
    value=False,
    help="Adjust decision thresholds per group to achieve demographic parity",
)

model = models[selected_model_name]
y_prob = model.predict_proba(X_test)[:, 1]


# --- Compute predictions ---
if use_fair_thresholds:
    group_thresholds = compute_group_thresholds(y_test, y_prob, g_test)
    y_pred = np.zeros_like(y_test)
    for group, t in group_thresholds.items():
        mask = g_test == group
        y_pred[mask] = (y_prob[mask] >= t).astype(int)
    st.sidebar.write("**Group Thresholds:**")
    for g, t in group_thresholds.items():
        name = group_names.get(g, g)
        st.sidebar.write(f"  {name}: {t:.3f}")
else:
    y_pred = (y_prob >= threshold).astype(int)


# --- Tab Layout ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Performance",
    "⚖️ Fairness Analysis",
    "🔍 Individual Explanations",
    "📈 Accuracy ↔ Fairness Tradeoff",
])


# === TAB 1: Model Performance ===
with tab1:
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("AUC", f"{roc_auc_score(y_test, y_prob):.4f}")
    col2.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
    col3.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
    col4.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
    col5.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")

    col_left, col_right = st.columns(2)

    # ROC Curve
    with col_left:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Model"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
        fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

    # Confusion Matrix
    with col_right:
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Paid", "Default"],
            y=["Paid", "Default"],
            text_auto=True,
            color_continuous_scale="Blues",
        )
        fig_cm.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)

    # Prediction distribution
    fig_dist = px.histogram(
        x=y_prob,
        color=y_test.astype(str),
        nbins=50,
        labels={"x": "Predicted Default Probability", "color": "Actual"},
        title="Prediction Distribution by Actual Outcome",
        barmode="overlay",
        opacity=0.7,
    )
    fig_dist.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold: {threshold}")
    st.plotly_chart(fig_dist, use_container_width=True)


# === TAB 2: Fairness Analysis ===
with tab2:
    audit = run_fairness_audit(y_test, y_pred, y_prob, g_test)

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)

    for col, (metric_key, label) in zip(
        [col1, col2, col3, col4],
        [
            ("demographic_parity", "Demographic Parity"),
            ("equal_opportunity", "Equal Opportunity"),
            ("predictive_parity", "Predictive Parity"),
            ("calibration", "Calibration"),
        ],
    ):
        result = audit[metric_key]
        status = "✅" if result["passes"] else "❌"
        col.metric(
            f"{status} {label}",
            f"Gap: {result['gap']:.4f}",
            delta=f"Threshold: {result['threshold']}",
            delta_color="off",
        )

    # Group approval rates
    st.subheader("Approval Rates by Group")
    dp = audit["demographic_parity"]
    group_rates = pd.DataFrame([
        {"Group": group_names.get(k, k), "Approval Rate": v}
        for k, v in dp["group_rates"].items()
    ])

    fig_rates = px.bar(
        group_rates,
        x="Group",
        y="Approval Rate",
        color="Approval Rate",
        color_continuous_scale="RdYlGn",
        title="Loan Approval Rates by Demographic Group",
    )
    fig_rates.add_hline(y=group_rates["Approval Rate"].mean(), line_dash="dash", annotation_text="Average")
    st.plotly_chart(fig_rates, use_container_width=True)

    # Equal opportunity rates
    st.subheader("True Positive Rates by Group (Equal Opportunity)")
    eo = audit["equal_opportunity"]
    eo_rates = pd.DataFrame([
        {"Group": group_names.get(k, k), "True Positive Rate": v}
        for k, v in eo["group_rates"].items()
        if not np.isnan(v)
    ])

    fig_eo = px.bar(
        eo_rates,
        x="Group",
        y="True Positive Rate",
        color="True Positive Rate",
        color_continuous_scale="RdYlGn",
        title="Among Creditworthy Borrowers: Approval Rate by Group",
    )
    st.plotly_chart(fig_eo, use_container_width=True)


# === TAB 3: Individual Explanations ===
with tab3:
    st.subheader("Explain a Single Loan Decision")
    st.markdown("Select a loan from the test set to see why the model made its prediction.")

    sample_idx = st.number_input("Sample Index", min_value=0, max_value=len(X_test) - 1, value=0)

    if st.button("Explain This Prediction"):
        instance = X_test.iloc[[sample_idx]]
        prob = model.predict_proba(instance)[:, 1][0]
        decision = "❌ DENY (Predicted Default)" if prob >= threshold else "✅ APPROVE (Predicted Repayment)"

        st.markdown(f"### {decision}")
        st.metric("Default Probability", f"{prob:.4f}")

        # SHAP explanation
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer(instance)

        contributions = pd.DataFrame({
            "Feature": instance.columns,
            "Value": instance.values[0],
            "SHAP Contribution": shap_vals.values[0],
        }).sort_values("SHAP Contribution", key=abs, ascending=False).head(10)

        fig_waterfall = px.bar(
            contributions,
            x="SHAP Contribution",
            y="Feature",
            orientation="h",
            color="SHAP Contribution",
            color_continuous_scale="RdBu_r",
            title="Top Factors in This Decision",
        )
        fig_waterfall.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_waterfall, use_container_width=True)

        st.dataframe(contributions, use_container_width=True)


# === TAB 4: Accuracy ↔ Fairness Tradeoff ===
with tab4:
    st.subheader("How Much Accuracy Do We Sacrifice for Fairness?")
    st.markdown(
        "This chart shows how different decision thresholds affect both "
        "model accuracy (AUC) and fairness (demographic parity gap)."
    )

    # Compute tradeoff curve
    thresholds_range = np.arange(0.2, 0.8, 0.02)
    tradeoff_data = []

    for t in thresholds_range:
        y_pred_t = (y_prob >= t).astype(int)
        acc = accuracy_score(y_test, y_pred_t)
        dp = demographic_parity(y_pred_t, g_test)

        tradeoff_data.append({
            "Threshold": t,
            "Accuracy": acc,
            "DP Gap": dp["gap"],
            "Approval Rate": np.mean(y_pred_t == 0),
        })

    tradeoff_df = pd.DataFrame(tradeoff_data)

    fig_tradeoff = make_subplots(specs=[[{"secondary_y": True}]])
    fig_tradeoff.add_trace(
        go.Scatter(x=tradeoff_df["Threshold"], y=tradeoff_df["Accuracy"], name="Accuracy", line=dict(color="blue")),
        secondary_y=False,
    )
    fig_tradeoff.add_trace(
        go.Scatter(x=tradeoff_df["Threshold"], y=tradeoff_df["DP Gap"], name="Fairness Gap", line=dict(color="red")),
        secondary_y=True,
    )
    fig_tradeoff.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="Fairness Target (0.05)", secondary_y=True)
    fig_tradeoff.update_layout(title="Accuracy vs. Fairness Tradeoff by Threshold")
    fig_tradeoff.update_xaxes(title_text="Decision Threshold")
    fig_tradeoff.update_yaxes(title_text="Accuracy", secondary_y=False)
    fig_tradeoff.update_yaxes(title_text="Demographic Parity Gap", secondary_y=True)

    st.plotly_chart(fig_tradeoff, use_container_width=True)

    # Pareto frontier
    st.subheader("Pareto Frontier: Best Achievable Tradeoffs")
    fig_pareto = px.scatter(
        tradeoff_df,
        x="Accuracy",
        y="DP Gap",
        color="Threshold",
        size="Approval Rate",
        color_continuous_scale="Viridis",
        title="Each Point = A Different Threshold Setting",
        labels={"DP Gap": "Fairness Gap (lower = fairer)"},
    )
    fig_pareto.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="Fairness Target")
    st.plotly_chart(fig_pareto, use_container_width=True)


# --- Footer ---
st.markdown("---")
st.markdown(
    "Built by **Ayoob Amaar** | "
    "[GitHub](https://github.com/YOUR_USERNAME) | "
    "Informed by CFPB Circular 2022-03 & EU AI Act requirements"
)
