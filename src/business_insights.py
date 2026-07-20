"""Derive concrete, data-driven textual insights and recommendations from cleaned campaign data."""

from __future__ import annotations

import pandas as pd


def _roi_by_group(df: pd.DataFrame, group_col: str) -> pd.Series:
    """Return units_sold generated per unit of budget_spent, grouped by group_col."""
    grouped = df.groupby(group_col).agg(budget=("budget_spent", "sum"), units=("units_sold", "sum"))
    return (grouped["units"] / grouped["budget"]).sort_values(ascending=False)


def generate_insights(df: pd.DataFrame) -> list[dict]:
    """Compute a list of {title, detail} business insights straight from the cleaned data."""
    insights = []

    channel_roi = _roi_by_group(df, "campaign_channel")
    best_channel, worst_channel = channel_roi.index[0], channel_roi.index[-1]
    insights.append(
        {
            "title": "Best and worst channel by budget efficiency",
            "detail": (
                f"'{best_channel}' delivers {channel_roi.iloc[0]:.3f} units sold per $ spent, "
                f"the highest of all channels, while '{worst_channel}' trails at "
                f"{channel_roi.iloc[-1]:.3f} units per $ — a "
                f"{(channel_roi.iloc[0] / max(channel_roi.iloc[-1], 1e-9)):.1f}x efficiency gap."
            ),
        }
    )

    conv_by_channel = df.groupby("campaign_channel")["conversion_rate"].mean().sort_values(ascending=False)
    insights.append(
        {
            "title": "Highest converting channel",
            "detail": (
                f"'{conv_by_channel.index[0]}' has the highest average conversion rate at "
                f"{conv_by_channel.iloc[0]:.2f}%, versus a dataset average of {df['conversion_rate'].mean():.2f}%."
            ),
        }
    )

    conv_by_city = df.groupby("city")["conversion_rate"].mean().sort_values(ascending=False)
    top_cities = conv_by_city.head(3)
    insights.append(
        {
            "title": "Top converting cities",
            "detail": (
                "Top 3 cities by average conversion rate: "
                + ", ".join(f"{city} ({rate:.2f}%)" for city, rate in top_cities.items())
                + f". Lowest is {conv_by_city.index[-1]} at {conv_by_city.iloc[-1]:.2f}%."
            ),
        }
    )

    budget_by_country = df.groupby("country")["budget_spent"].sum().sort_values(ascending=False)
    units_by_country = df.groupby("country")["units_sold"].sum().sort_values(ascending=False)
    insights.append(
        {
            "title": "Budget allocation vs. sales output by country",
            "detail": (
                f"'{budget_by_country.index[0]}' receives the largest budget share "
                f"(${budget_by_country.iloc[0]:,.0f}), while '{units_by_country.index[0]}' actually "
                f"sells the most units ({units_by_country.iloc[0]:,.0f}). "
                + (
                    "Budget allocation is well aligned with sales performance."
                    if budget_by_country.index[0] == units_by_country.index[0]
                    else "This budget-to-output misalignment suggests reallocating spend toward "
                    f"'{units_by_country.index[0]}' could improve overall ROI."
                )
            ),
        }
    )

    category_corr = df.groupby("product_category")["conversion_rate"].mean().sort_values(ascending=False)
    insights.append(
        {
            "title": "Best performing product category",
            "detail": (
                f"'{category_corr.index[0]}' converts best at {category_corr.iloc[0]:.2f}% on average, "
                f"ahead of '{category_corr.index[-1]}' at {category_corr.iloc[-1]:.2f}%."
            ),
        }
    )

    corr_matrix = df[["budget_spent", "clicks", "conversion_rate", "units_sold"]].corr()
    strongest_pair, strongest_val = None, 0.0
    cols = corr_matrix.columns
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            val = corr_matrix.loc[col_a, col_b]
            if abs(val) > abs(strongest_val):
                strongest_val, strongest_pair = val, (col_a, col_b)
    insights.append(
        {
            "title": "Strongest numeric relationship",
            "detail": (
                f"'{strongest_pair[0]}' and '{strongest_pair[1]}' show the strongest correlation "
                f"in the dataset at r={strongest_val:.2f}."
            ),
        }
    )

    return insights


def generate_recommendations(df: pd.DataFrame, insights: list[dict]) -> list[str]:
    """Turn computed insights into a short list of numbered, actionable recommendations."""
    channel_roi = _roi_by_group(df, "campaign_channel")
    conv_by_channel = df.groupby("campaign_channel")["conversion_rate"].mean().sort_values(ascending=False)
    conv_by_city = df.groupby("city")["conversion_rate"].mean().sort_values(ascending=False)
    budget_by_country = df.groupby("country")["budget_spent"].sum().sort_values(ascending=False)
    units_by_country = df.groupby("country")["units_sold"].sum().sort_values(ascending=False)

    recommendations = [
        f"Reallocate incremental budget toward '{channel_roi.index[0]}', which returns "
        f"{channel_roi.iloc[0]:.3f} units sold per $ spent versus {channel_roi.iloc[-1]:.3f} for "
        f"'{channel_roi.index[-1]}'.",
        f"Scale creative testing on '{conv_by_channel.index[0]}' campaigns, which already convert at "
        f"{conv_by_channel.iloc[0]:.2f}% on average, above the {df['conversion_rate'].mean():.2f}% dataset mean.",
        f"Prioritize localized campaigns in '{conv_by_city.index[0]}', the top-converting city at "
        f"{conv_by_city.iloc[0]:.2f}%, to compound existing demand.",
    ]

    if budget_by_country.index[0] != units_by_country.index[0]:
        recommendations.append(
            f"Investigate why '{budget_by_country.index[0]}' absorbs the largest budget "
            f"(${budget_by_country.iloc[0]:,.0f}) yet '{units_by_country.index[0]}' generates more units sold "
            f"({units_by_country.iloc[0]:,.0f}); shifting spend could raise blended ROI."
        )

    recommendations.append(
        "Retire or redesign the lowest-efficiency channel/city combinations identified above before the "
        "next budget cycle, and re-test with a smaller pilot spend."
    )

    return recommendations
