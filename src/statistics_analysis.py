"""Descriptive statistics, groupby aggregations, pivot tables and correlation analysis."""

from __future__ import annotations

import pandas as pd

FLOAT_COLS = ["budget_spent", "clicks", "conversion_rate", "units_sold"]
CATEGORICAL_COLS = ["campaign_channel", "city", "country", "product_category"]


def compute_full_insights(df: pd.DataFrame) -> dict:
    """Compute meta, totals, averages, numeric profiles, categorical and date insights."""
    insights: dict = {
        "meta": {},
        "totals": {},
        "averages": {},
        "numeric_profiles": {},
        "categorical": {},
        "dates": {},
    }

    insights["meta"] = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "missing_values": df.isna().sum(),
        "dtypes": df.dtypes.astype(str),
    }

    insights["totals"]["units_sold_per_city"] = df.groupby("city")["units_sold"].sum().sort_values(ascending=False)
    insights["totals"]["units_sold_per_country"] = df.groupby("country")["units_sold"].sum().sort_values(ascending=False)
    insights["totals"]["budget_spent_per_channel"] = df.groupby("campaign_channel")["budget_spent"].sum().sort_values(ascending=False)
    insights["totals"]["clicks_per_channel"] = df.groupby("campaign_channel")["clicks"].sum().sort_values(ascending=False)

    insights["averages"]["avg_units_sold_per_city"] = df.groupby("city")["units_sold"].mean().sort_values(ascending=False)
    insights["averages"]["avg_units_sold_per_country"] = df.groupby("country")["units_sold"].mean().sort_values(ascending=False)
    insights["averages"]["avg_budget_spent_per_channel"] = df.groupby("campaign_channel")["budget_spent"].mean().sort_values(ascending=False)
    insights["averages"]["avg_clicks_per_channel"] = df.groupby("campaign_channel")["clicks"].mean().sort_values(ascending=False)
    insights["averages"]["avg_conversion_rate_per_city"] = df.groupby("city")["conversion_rate"].mean().sort_values(ascending=False)
    insights["averages"]["avg_conversion_rate_per_country"] = df.groupby("country")["conversion_rate"].mean().sort_values(ascending=False)
    insights["averages"]["avg_conversion_rate_per_channel"] = df.groupby("campaign_channel")["conversion_rate"].mean().sort_values(ascending=False)

    for col in FLOAT_COLS:
        insights["numeric_profiles"][col] = df[col].describe()

    for col in CATEGORICAL_COLS:
        counts = df[col].value_counts()
        insights["categorical"][col] = {
            "unique_values": df[col].nunique(),
            "top_values": counts.head(10),
            "distribution": df[col].value_counts(normalize=True).round(4),
            "rare_values": counts[counts < 5],
        }

    date_series = pd.to_datetime(df["campaign_date"], errors="coerce")
    if date_series.notna().any():
        insights["dates"] = {
            "min_date": date_series.min(),
            "max_date": date_series.max(),
            "date_range_days": (date_series.max() - date_series.min()).days,
            "records_per_month": df.groupby(date_series.dt.to_period("M")).size(),
            "records_per_year": df.groupby(date_series.dt.year).size(),
            "weekday_distribution": date_series.dt.day_name().value_counts(),
        }

    return insights


def compute_aggregations(df: pd.DataFrame) -> dict:
    """Compute named multi-metric aggregations grouped by city."""
    grouped = df.groupby("city")
    clicks_summary = grouped["clicks"].agg(["mean", "sum", "median", "std"])
    named_agg = grouped.agg(
        avg_budget_spent=("budget_spent", "mean"),
        sum_budget_spent=("budget_spent", "sum"),
        median_budget_spent=("budget_spent", "median"),
        std_budget_spent=("budget_spent", "std"),
    )
    return {"clicks_by_city": clicks_summary, "budget_by_city": named_agg}


def compute_pivot_tables(df: pd.DataFrame) -> dict:
    """Build pivot tables: clicks by country/channel and clicks & units by city/channel."""
    pivot_clicks_by_country = df.pivot_table(
        index="country", columns="campaign_channel", values="clicks", aggfunc="mean", margins=True
    ).sort_values(by="All", ascending=False)

    pivot_units_clicks_by_city = df.pivot_table(
        index="city", columns="campaign_channel", values=["units_sold", "clicks"], aggfunc="sum", margins=True
    ).sort_values(by=("clicks", "All"), ascending=False)

    return {
        "clicks_by_country_channel": pivot_clicks_by_country,
        "units_clicks_by_city_channel": pivot_units_clicks_by_city,
    }


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the Pearson correlation matrix across the numeric campaign metrics."""
    return df[FLOAT_COLS].corr()
