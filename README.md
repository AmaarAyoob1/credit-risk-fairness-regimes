# Fair Lending: Loan Default Prediction with Fairness Constraints

<p align="center">
  <img src="Dashboard Preview 1.png" alt="Dashboard Preview" width="800"/>
  <img src="Dashboard Preview 2.png" alt="Dashboard Preview" width="800"/>
</p>

**A credit risk model that doesn't just predict defaults — it audits itself for demographic bias.**

Built with XGBoost, LightGBM, SHAP explainability, and fairness-aware evaluation metrics aligned with CFPB and EU AI Act regulatory requirements.

---

## Why This Project Exists

Most credit risk models optimize for a single metric: accuracy (or AUC). But in lending, accuracy alone isn't enough. A model that's 95% accurate overall could still systematically deny loans to protected demographic groups at higher rates — violating fair lending laws and excluding creditworthy borrowers.

This project builds a loan default predictor that:
- Achieves strong predictive performance (AUC > 0.85)
- Audits predictions across demographic groups using fairness metrics
- Lets users **interactively explore the accuracy ↔ fairness tradeoff**
- Provides SHAP-based explanations for every prediction
- Flags when a model's decisions may be discriminatory

## Results

| Model | AUC | Accuracy | Demographic Parity Gap | Equal Opportunity Gap |
|-------|-----|----------|----------------------|----------------------|
| XGBoost (unconstrained) | 0.88 | 0.84 | 0.12 | 0.09 |
| XGBoost (fairness-tuned) | 0.86 | 0.82 | 0.04 | 0.03 |
| LightGBM (unconstrained) | 0.87 | 0.83 | 0.11 | 0.08 |
| LightGBM (fairness-tuned) | 0.85 | 0.81 | 0.03 | 0.03 |

> **Key finding:** With threshold adjustment and fairness-aware tuning, we reduced demographic parity gap by ~70% while sacrificing only ~2% AUC — a tradeoff most lenders would accept.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/loan-default-fairness.git
cd loan-default-fairness

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the data
python src/data_pipeline.py --download

# Train models
python src/train.py --config configs/config.yaml

# Launch the dashboard
streamlit run streamlit_app/app.py
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Raw Data    │────▶│  Feature     │────▶│  Model      │
│  (Lending    │     │  Engineering │     │  Training   │
│   Club CSV)  │     │  Pipeline    │     │  (XGB/LGBM) │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
                    ┌──────────────┐     ┌───────▼──────┐
                    │  Streamlit   │◀────│  Fairness    │
                    │  Dashboard   │     │  Evaluation  │
                    │              │     │  + SHAP      │
                    └──────────────┘     └──────────────┘
```

## Data

This project uses the [Lending Club Loan Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (~2.2M loans, 150+ features). The data pipeline automatically:
- Downloads and caches the dataset
- Handles missing values and outliers
- Engineers 25+ features from raw loan attributes
- Splits data with stratification on both target and protected attributes

**Note:** Raw data is not committed to this repo. Run `python src/data_pipeline.py --download` to fetch it.

## Fairness Metrics Explained

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Demographic Parity** | Are approval rates equal across groups? | Gap < 0.05 |
| **Equal Opportunity** | Are true positive rates equal across groups? | Gap < 0.05 |
| **Predictive Parity** | Is precision equal across groups? | Gap < 0.05 |
| **Calibration** | Are predicted probabilities accurate across groups? | Gap < 0.03 |

## Project Structure

```
loan-default-fairness/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── configs/
│   └── config.yaml              # Model & pipeline configuration
├── data/
│   └── README.md                # Data sources & download instructions
├── docs/
│   └── architecture.md          # Detailed technical documentation
├── notebooks/
│   └── exploration.ipynb        # EDA and initial analysis
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py         # Data download, cleaning, feature engineering
│   ├── features.py              # Feature engineering functions
│   ├── train.py                 # Model training with fairness constraints
│   ├── evaluate.py              # Evaluation metrics + fairness auditing
│   ├── fairness.py              # Fairness metric calculations
│   └── explain.py               # SHAP explanations
├── streamlit_app/
│   └── app.py                   # Interactive dashboard
└── tests/
    ├── test_features.py
    ├── test_fairness.py
    └── test_pipeline.py
```

## Technical Approach

**Feature Engineering:** 25+ features including debt-to-income ratios, credit utilization, payment history patterns, employment length encoding, and geographic risk indicators.

**Modeling:** XGBoost and LightGBM with Bayesian hyperparameter optimization via Optuna. Models are trained with both unconstrained and fairness-aware objectives.

**Fairness-Aware Tuning:** Three approaches implemented:
1. **Threshold adjustment** — per-group decision thresholds to equalize approval rates
2. **Reweighting** — sample weights inversely proportional to group representation
3. **Constrained optimization** — custom objective penalizing fairness violations during training

**Explainability:** SHAP TreeExplainer for global and local feature importance, with group-level SHAP analysis showing whether features contribute differently to predictions across demographics.

## What I'd Improve

- Add adversarial debiasing as a fourth fairness approach
- Implement reject inference to account for selection bias in historical lending data
- Build a model monitoring pipeline that tracks fairness metrics over time in production
- Add causal inference methods to distinguish correlation from actual discrimination
- Expand to multi-class (default severity) rather than binary default/no-default

## Regulatory Context

This project is informed by:
- **CFPB Circular 2022-03** — adverse action notices for AI-driven lending
- **ECOA / Regulation B** — prohibition of discrimination in credit decisions
- **EU AI Act (2024)** — high-risk AI system requirements for credit scoring
- **SR 11-7 (OCC/Fed)** — model risk management guidance

## License

MIT License — see [LICENSE](LICENSE) for details.
