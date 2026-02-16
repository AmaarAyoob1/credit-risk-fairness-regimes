# Data

## Source

**Lending Club Loan Dataset**
- Source: [Kaggle - Lending Club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- Size: ~2.2M loans, 150+ features
- Period: 2007-2018
- License: CC0 Public Domain

## Download

### Option 1: Automatic (requires Kaggle API)
```bash
# Set up Kaggle credentials first: https://www.kaggle.com/docs/api
python src/data_pipeline.py --download
```

### Option 2: Manual
1. Download from https://www.kaggle.com/datasets/wordsforthewise/lending-club
2. Place the CSV file in `data/raw/lending_club.csv`
3. Run the pipeline: `python src/data_pipeline.py`

## Important Notes

- **Raw data is NOT committed to this repo** — download it yourself
- The pipeline handles all cleaning, feature engineering, and splitting
- Processed data is saved as parquet files in `data/processed/`
- Data splits are stratified on both target and protected attributes
