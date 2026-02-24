# Macro-Regime Credit Risk Analysis

**How do macroeconomic conditions change loan default behavior?**

This project extends a [credit risk model](../) by connecting individual loan defaults to macroeconomic regimes. Using a Hidden Markov Model trained on yield curve dynamics, credit spreads, and market volatility from Yahoo Finance, it identifies three economic regimes (Expansion, Contraction, Crisis) and shows how a credit model's accuracy, calibration, and fairness metrics degrade across regimes.

![Regime Timeline](figures/regime_timeline.png)

## Key Findings

| Metric | Expansion | Contraction | Crisis |
|--------|-----------|-------------|--------|
| Default Rate | 19.7% | 21.3% | 22.9% |
| Model AUC | 0.7361 | 0.7267 | 0.6965 |
| Calibration Error | −0.0002 | −0.0012 | +0.0028 |
| Passes 4/5ths Rule | ✓ | ✓ | ✓ |

> The model loses **4 points of AUC during Crisis** (0.736 → 0.697) and default rates are **1.16× higher** than during Expansion. The model's discriminative power degrades exactly when accurate predictions matter most.

![Combined Dashboard](figures/combined_dashboard.png)

## Why This Matters

Credit risk models are validated and deployed during stable periods, but they're stress-tested against adverse scenarios that fundamentally change borrower behavior. This project demonstrates:

- **Default rates are 1.16× higher during Crisis** — loans originated during economic stress default at 22.9% vs 19.7% during Expansion
- **Model AUC drops 0.040** from Expansion to Crisis — the model trained on all-regime data (88.6% Expansion) loses discriminative power under stress
- **Calibration shifts** — the model is nearly perfectly calibrated during Expansion (−0.0002 error) but slightly overestimates default risk during Crisis (+0.0028)
- **Fairness holds across regimes** — the 4/5ths rule passes in all three regimes, though demographic parity gaps shift (0.053 in Expansion vs 0.048 in Crisis)

This connects directly to what bank model risk management teams do: validate that models perform acceptably not just on average, but under the specific stress scenarios regulators require (CCAR, DFAST, CECL).

## Architecture
```
Yahoo Finance API                    Lending Club (1.3M loans)
    │                                        │
    ▼                                        ▼
┌─────────────────┐               ┌─────────────────────┐
│  Data Pipeline  │               │  Credit Risk Model   │
│  (9 tickers,    │               │  (XGBoost, SHAP,     │
│   15 features)  │               │   fairness auditing) │
└────────┬────────┘               └──────────┬──────────┘
         │                                    │
         ▼                                    │
┌─────────────────┐                           │
│ Regime Detector │                           │
│ (Gaussian HMM,  │                           │
│  3 regimes)     │                           │
└────────┬────────┘                           │
         │              ┌─────────────────┐   │
         └──────────────► Regime-Credit   ◄───┘
                        │ Analysis        │
                        │ (performance,   │
                        │  fairness, and  │
                        │  stress testing │
                        │  by regime)     │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ Visualizations  │
                        │ (8 charts)      │
                        └─────────────────┘
```

## Macro Indicators

| Ticker | Indicator | Economic Rationale |
|--------|-----------|-------------------|
| `^TNX` | 10Y Treasury Yield | Benchmark lending rate; drives mortgage and loan pricing |
| `^IRX` | 13W T-Bill Rate | Short-end rate; yield curve slope = `^TNX − ^IRX` |
| `^FVX` | 5Y Treasury Yield | Mid-curve flattening signal |
| `^VIX` | Volatility Index | Market fear → credit tightening → defaults rise |
| `HYG` | High-Yield Bond ETF | Junk bond proxy; sell-off during stress |
| `LQD` | Investment-Grade Bond ETF | Credit spread = `HYG − LQD` returns |
| `SPY` | S&P 500 | Equity market health; wealth effect on borrowers |
| `GLD` | Gold ETF | Flight to safety signal |
| `DX-Y.NYB` | US Dollar Index | Strong dollar tightens global credit |

## Regime Detection

The Hidden Markov Model identifies three regimes from observable market signals across 3,990 trading days (2008–2025):

- **Expansion** (44.7% of days): Mean VIX 15.5, average duration 7.7 months
- **Contraction** (30.3% of days): Mean VIX 18.8, average duration 9.6 months
- **Crisis** (25.0% of days): Mean VIX 30.8, average duration 4.0 months

Regimes validated against known economic events without supervision:
- ✓ **2008 Global Financial Crisis** — classified as 89.7% Crisis
- ✓ **2020 COVID crash** — classified as 66.1% Contraction + 33.9% Crisis
- ✓ **2022 Rate Shock** — classified as 49.5% Contraction + 47.0% Crisis

![Financial Conditions Index](figures/financial_conditions_index.png)

## Quick Start
```bash
# Clone and install
git clone https://github.com/AmaarAyoob1/credit-risk-fairness-regimes.git
cd credit-risk-fairness-regimes/macro-regime
pip install -r requirements.txt

# Regime detection only (pulls data from Yahoo Finance)
python run_analysis.py --regimes-only

# Full credit-regime analysis (requires trained model in parent repo)
python run_bridge.py

# Generate all visualizations
python generate_charts.py
```

## Project Structure
```
macro-regime/
├── run_analysis.py            # Main entry point
├── run_bridge.py              # Credit model ↔ regime connector
├── generate_charts.py         # Generate all 8 visualizations
├── configs/
│   └── config.yaml            # All parameters (tickers, dates, HMM settings)
├── src/
│   ├── data_pipeline.py       # Yahoo Finance ingestion + feature engineering
│   ├── regime_detector.py     # HMM regime detection + validation
│   ├── regime_credit_analysis.py  # Core analysis: defaults, AUC, fairness by regime
│   └── visualizations.py      # Publication-quality charts
├── data/                      # Generated data (gitignored)
├── figures/                   # Generated charts
└── requirements.txt
```

## Technical Details

**Regime Detection**: Gaussian HMM with full covariance, 3 components, 10 random restarts (best log-likelihood selected). Features: yield curve slope, rolling credit spread, smoothed VIX, 60-day S&P 500 return. All features standardized (zero mean, unit variance). Regimes reordered by mean VIX for economically meaningful labels.

**Loan-Regime Mapping**: Each of 1,331,863 Lending Club loans matched to the macroeconomic regime active on its origination date via nearest-date join to the trading day calendar. 100% match rate achieved.

**Financial Conditions Index**: Equal-weighted z-score composite of yield curve slope (inverted), credit spread, VIX, and S&P 500 momentum. Higher values indicate tighter conditions (worse for borrowers).

**Transition Matrix** (daily probabilities):
```
              Expansion  Contraction  Crisis
Expansion       0.9938       0.0017   0.0045
Contraction     0.0025       0.9950   0.0025
Crisis          0.0080       0.0040   0.9880
```

## Built With

- **Data**: Yahoo Finance (yfinance), Lending Club (1.3M loans)
- **Regime Detection**: hmmlearn (Gaussian HMM)
- **Credit Model**: XGBoost, scikit-learn, SHAP
- **Visualization**: matplotlib, seaborn

## Author

**Ayoob Amaar** — MS Statistics & Machine Learning | MS Financial Engineering, Claremont Graduate University
