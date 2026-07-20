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
├── serve_dashboard.py       # local HTTP server to preview outputs/dashboard.html
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

## Previewing the Dashboard on Localhost

`outputs/dashboard.html` can be opened directly as a file, but some browsers restrict local scripts
when opened via `file://`. To preview it properly over HTTP:

```bash
python3 serve_dashboard.py
```

This serves the `outputs/` folder on `http://localhost:8000` and opens
`http://localhost:8000/dashboard.html` in your default browser automatically. Stop it with `Ctrl+C`.

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

## Visualizations

The dashboard (`outputs/dashboard.html`) has 12 interactive Plotly charts, split into two sections:

**Current Performance** (descriptive views of the cleaned data as it stands):
- Budget spent vs. conversion rate by channel, animated month-by-month
- Geographic breakdown: units sold and conversion rate by country
- Monthly time trend of budget spend and conversion rate
- Budget efficiency: units sold per $ spent, by channel
- Product category breakdown (share of units sold)
- Conversion rate distribution across all campaigns
- Correlation heatmap between budget, clicks, conversion rate and units sold

**Model Predictions** (how the trained conversion-rate model performs):
- Model comparison: test R² and RMSE across candidates
- Top feature importances from the winning model
- Actual vs. predicted conversion rate (test set)
- Prediction residuals distribution
- Actual vs. predicted conversion rate by channel (test set)

Static PNG equivalents used throughout the PDF report are in `outputs/figures/`.

## Tech Stack

- **Data**: pandas, numpy
- **ML**: scikit-learn (LinearRegression, RandomForestRegressor, GradientBoostingRegressor), joblib
- **Visualization**: Plotly (interactive dashboard), matplotlib + seaborn (static report charts)
- **Reporting**: reportlab (platypus, custom paragraph styles, styled tables)
