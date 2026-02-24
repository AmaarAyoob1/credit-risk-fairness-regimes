# Macro-Regime Credit Risk Analysis

**How do macroeconomic conditions change loan default behavior?**

This project extends a [credit risk model](../loan-default-fairness) by connecting individual loan defaults to macroeconomic regimes. Using a Hidden Markov Model trained on yield curve dynamics, credit spreads, and market volatility from Yahoo Finance, it identifies three economic regimes (Expansion, Contraction, Crisis) and shows how a credit model's accuracy, calibration, and fairness metrics degrade across regimes.

![Regime Timeline](figures/regime_timeline.png)

## Key Findings

| Metric | Expansion | Contraction | Crisis |
|--------|-----------|-------------|--------|
| Default Rate | X.XX% | X.XX% | X.XX% |
| Model AUC | 0.XXX | 0.XXX | 0.XXX |
| Calibration Error | +0.XXX | +0.XXX | −0.XXX |
| Passes 4/5ths Rule | ✓ | ✓ | ✗ |

> The model **underestimates default risk during Crisis** and **fails fair lending compliance** under economic stress — exactly when accuracy matters most.

## Why This Matters

Credit risk models are validated and deployed during stable periods, but they're stress-tested against adverse scenarios that fundamentally change borrower behavior. This project demonstrates:

- **Default rates increase X.Xx during Crisis** — the model trained on all-regime data underestimates this shift
- **Fairness degrades under stress** — certain demographic groups are disproportionately affected by economic downturns, causing the model to fail the 4/5ths rule during Crisis
- **Regime-conditional thresholds** can maintain fairness across economic cycles where static thresholds cannot

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
                        │ & Report        │
                        └─────────────────┘
```

## Macro Indicators

| Ticker | Indicator | Economic Rationale |
|--------|-----------|-------------------|
| `^TNX` | 10Y Treasury Yield | Benchmark lending rate; drives mortgage and loan pricing |
| `^IRX` | 13W T-Bill Rate | Short-end rate; yield curve slope = `^TNX − ^IRX` |
| `^VIX` | Volatility Index | Market fear → credit tightening → defaults rise |
| `HYG` | High-Yield Bond ETF | Junk bond proxy; sell-off during stress |
| `LQD` | Investment-Grade Bond ETF | Credit spread = `HYG − LQD` returns |
| `SPY` | S&P 500 | Equity market health; wealth effect on borrowers |
| `GLD` | Gold ETF | Flight to safety signal |
| `DX-Y.NYB` | US Dollar Index | Strong dollar tightens global credit |

## Regime Detection

The Hidden Markov Model identifies three regimes from observable market signals:

- **Expansion** (green): Low VIX, positive yield curve, tight credit spreads
- **Contraction** (orange): Rising VIX, flattening/inverting yield curve
- **Crisis** (red): Elevated VIX, inverted yield curve, wide credit spreads

Regimes are validated against known economic events:
- ✓ 2001 Dot-Com recession → Contraction/Crisis
- ✓ 2008 Global Financial Crisis → Crisis
- ✓ 2020 COVID crash → Crisis
- ✓ 2022 rate shock → Contraction

## Quick Start

```bash
# Clone and install
git clone https://github.com/AmaarAyoob1/loan-default-fairness.git
cd loan-default-fairness/macro-regime
pip install -r requirements.txt

# Run full pipeline (pulls data from Yahoo Finance)
python run_analysis.py

# Use cached data (skip download)
python run_analysis.py --skip-download

# Regime detection only (no credit model needed)
python run_analysis.py --regimes-only
```

## Project Structure

```
macro-regime/
├── run_analysis.py            # Main entry point
├── configs/
│   └── config.yaml            # All parameters (tickers, dates, HMM settings)
├── src/
│   ├── data_pipeline.py       # Yahoo Finance ingestion + feature engineering
│   ├── regime_detector.py     # HMM regime detection + validation
│   ├── regime_credit_analysis.py  # Core analysis: defaults, AUC, fairness by regime
│   └── visualizations.py      # Publication-quality charts
├── data/                      # Generated data (gitignored)
├── figures/                   # Generated charts
├── requirements.txt
└── README.md
```

## Technical Details

**Regime Detection**: Gaussian HMM with full covariance, 3 components, 10 random restarts (best log-likelihood selected). Features: yield curve slope, rolling credit spread, smoothed VIX, 60-day S&P 500 return. All features standardized (zero mean, unit variance).

**Loan-Regime Mapping**: Each Lending Club loan is matched to the macroeconomic regime active on its origination date via nearest-date join to the trading day calendar.

**Financial Conditions Index**: Equal-weighted z-score composite of yield curve slope (inverted), credit spread, VIX, and S&P 500 momentum. Higher values indicate tighter conditions.

## Built With

- **Data**: Yahoo Finance (yfinance), Lending Club (1.3M loans)
- **Regime Detection**: hmmlearn (Gaussian HMM)
- **Credit Model**: XGBoost, scikit-learn, SHAP
- **Visualization**: matplotlib, seaborn

## Author

Ayoob Amaar — MS Statistics & Machine Learning | MS Financial Engineering, Claremont Graduate University
