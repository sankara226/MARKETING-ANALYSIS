# Marketing Campaign Analysis

An end-to-end data analysis project that turns a messy, multi-language, multi-currency marketing
campaign export into a cleaned dataset, statistical insights, a conversion-rate prediction model,
an interactive dashboard, and a professional PDF report.

## Problem Statement

The source export (`data/messy_marketing_campaign.csv`, 1,260 rows) is full of realistic data-quality
issues: typos and repeated/garbled substrings in country/city/product names ("Fr@nce", "torontotoronto"),
mixed currency symbols (€/$), percent signs, decimal commas, inconsistent date formats, and 1-20% missing
values scattered across every column. The goal is to clean it reliably, quantify campaign performance by
channel/city/country, predict conversion rate from campaign attributes, and produce concrete,
data-grounded budget recommendations.

## Architecture

```
MARKETING CAMPAIGN ANALYSIS/
│
├── data/
│   ├── messy_marketing_campaign.csv   (raw, untouched source data)
│   └── .gitkeep
│
├── outputs/
│   ├── cleaned_marketing_campaign.csv (pipeline-generated clean dataset)
│   ├── dashboard.html                 (interactive Plotly dashboard)
│   ├── report.pdf                     (final PDF report)
│   ├── model.joblib                   (trained sklearn pipeline)
│   └── figures/                       (static PNGs used in the PDF)
│
├── src/
│   ├── data_loader.py           # CSV loading, path configuration
│   ├── cleaning.py              # normalization, fuzzy canonicalization, geo-correction, imputation
│   ├── statistics_analysis.py   # descriptive stats, groupby aggregations, pivots, correlations
│   ├── business_insights.py     # data-driven insights and recommendations
│   ├── ml_model.py              # feature engineering, model selection, prediction
│   ├── plots_interactive.py     # Plotly figures + responsive HTML dashboard
│   ├── plots_static.py          # matplotlib/seaborn static PNGs
│   └── report_pdf.py            # reportlab PDF report assembly
│
├── main.py                  # orchestrates the full pipeline end to end
├── requirements.txt
└── structure.txt
```

## Setup & Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

Running `main.py` executes the whole pipeline (load → clean → export CSV → statistics → ML training →
business insights → interactive dashboard → PDF report) and prints progress for each stage. All
outputs land in `outputs/`.

## Cleaning Pipeline Highlights

- Text normalization + fuzzy canonicalization (`difflib.SequenceMatcher`, similarity threshold **0.72**)
  resolves typos and leetspeak/emoji-mangled variants of country, city, product, category, and channel
  names onto a fixed canonical vocabulary.
- Repeated-substring collapsing (`"usausausa"` → `"usa"`) runs before fuzzy matching.
- Currency (€/$), percent signs, and decimal commas are stripped and coerced to numeric for
  `budget_spent`, `clicks`, `conversion_rate`, `units_sold`.
- Mixed date formats are parsed to ISO `YYYY-MM-DD`.
- Geo-consistency correction: a known city (e.g. `paris`) overrides a mismatched or missing country
  (`france`).
- Remaining gaps are imputed with column mean (numeric) or mode (categorical).
- `impressions` is dropped intentionally — its values were too corrupted and inconsistently scaled
  across rows to trust for any analysis.

## Key Findings

- **1,033 of 1,260 rows (82.0%)** survived cleaning with full data integrity.
- **`google_ads`** is the most budget-efficient channel at 13.8 units sold per $ spent, vs. 7.4 for the
  least efficient channel, `tiktok` — a ~1.9x efficiency gap.
- **`instagram`** has the highest average conversion rate (6.55%) against a 6.08% dataset average.
- **Madrid, Toronto, and Rome** are the top-converting cities (7.39%, 7.14%, 6.71%).
- **France** receives the largest total budget (~$2.19M) while **Japan** generates the most units sold
  (~22.1M), a budget-to-output misalignment worth investigating.
- A Linear Regression / Random Forest / Gradient Boosting comparison (5-fold CV + held-out test set) was
  run to predict `conversion_rate`; **Linear Regression** won by test R² (see `outputs/report.pdf` for the
  full comparison table). The near-zero R² across all three models indicates conversion rate in this
  dataset is largely independent of the available campaign attributes — a finding reported honestly
  rather than masked.

## Screenshots

Static chart PNGs used throughout the PDF report are in `outputs/figures/`:
`budget_by_channel.png`, `conversion_by_city.png`, `units_sold_trend.png`,
`correlation_heatmap.png`, `feature_importance.png`, `actual_vs_predicted.png`.

The fully interactive version of these (plus an animated budget-vs-conversion scatter and a
geographic breakdown) is in `outputs/dashboard.html` — open it directly in a browser.

## Tech Stack

- **Data**: pandas, numpy
- **ML**: scikit-learn (LinearRegression, RandomForestRegressor, GradientBoostingRegressor), joblib
- **Visualization**: Plotly (interactive dashboard), matplotlib + seaborn (static report charts)
- **Reporting**: reportlab (platypus, custom paragraph styles, styled tables)
