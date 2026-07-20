"""Static high-resolution matplotlib/seaborn PNGs for the PDF report, in a modern clean style."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_loader import FIGURES_DIR

PALETTE = ["#2563eb", "#0891b2", "#059669", "#d97706", "#dc2626", "#7c3aed", "#db2777", "#4b5563"]

sns.set_theme(style="whitegrid", context="talk", font_scale=0.75)
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#d0d3d9",
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "font.family": "sans-serif",
    }
)


def _save(fig: plt.Figure, name: str) -> str:
    """Save a matplotlib figure to outputs/figures/<name>.png at 200 DPI and close it."""
    path = FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_budget_by_channel(df: pd.DataFrame) -> str:
    """Bar chart of total budget spent per campaign channel."""
    grouped = df.groupby("campaign_channel")["budget_spent"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=grouped.values, y=grouped.index, hue=grouped.index, palette=PALETTE[: len(grouped)], legend=False, ax=ax)
    ax.set_title("Total Budget Spent by Campaign Channel")
    ax.set_xlabel("Budget Spent ($)")
    ax.set_ylabel("")
    return _save(fig, "budget_by_channel")


def plot_conversion_by_city(df: pd.DataFrame) -> str:
    """Bar chart of average conversion rate per city."""
    grouped = df.groupby("city")["conversion_rate"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=grouped.values, y=grouped.index, hue=grouped.index, palette=PALETTE[: len(grouped)], legend=False, ax=ax)
    ax.set_title("Average Conversion Rate by City")
    ax.set_xlabel("Conversion Rate (%)")
    ax.set_ylabel("")
    return _save(fig, "conversion_by_city")


def plot_units_sold_trend(df: pd.DataFrame) -> str:
    """Line chart of total units sold per month."""
    dates = pd.to_datetime(df["campaign_date"], errors="coerce")
    trend = df.groupby(dates.dt.to_period("M"))["units_sold"].sum()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(trend.index.astype(str), trend.values, marker="o", color=PALETTE[0], linewidth=2)
    ax.set_title("Monthly Units Sold Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Units Sold")
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, "units_sold_trend")


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    """Heatmap of correlations between the numeric campaign metrics."""
    corr = df[["budget_spent", "clicks", "conversion_rate", "units_sold"]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", vmin=-1, vmax=1, ax=ax, square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Between Campaign Metrics")
    return _save(fig, "correlation_heatmap")


def plot_feature_importance(feature_importance: pd.Series) -> str:
    """Horizontal bar chart of the top feature importances from the best trained model."""
    top = feature_importance.head(12).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top.values, color=PALETTE[1])
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance")
    return _save(fig, "feature_importance")


def plot_actual_vs_predicted(y_test: pd.Series, y_pred: pd.Series) -> str:
    """Scatter of actual vs predicted conversion rate with a diagonal reference line."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.5, color=PALETTE[4], edgecolor="none")
    min_v, max_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="gray")
    ax.set_title("Actual vs Predicted Conversion Rate")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    return _save(fig, "actual_vs_predicted")


def plot_channel_efficiency(df: pd.DataFrame) -> str:
    """Bar chart of units sold per dollar spent, by channel."""
    grouped = df.groupby("campaign_channel").agg(budget_spent=("budget_spent", "sum"), units_sold=("units_sold", "sum"))
    grouped["units_per_dollar"] = grouped["units_sold"] / grouped["budget_spent"]
    grouped = grouped.sort_values("units_per_dollar", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=grouped["units_per_dollar"], y=grouped.index, hue=grouped.index, palette=PALETTE[: len(grouped)], legend=False, ax=ax)
    ax.set_title("Budget Efficiency: Units Sold per $ Spent by Channel")
    ax.set_xlabel("Units Sold per $")
    ax.set_ylabel("")
    return _save(fig, "channel_efficiency")


def plot_category_breakdown(df: pd.DataFrame) -> str:
    """Pie chart of total units sold share by product category."""
    grouped = df.groupby("product_category")["units_sold"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(grouped.values, labels=grouped.index, autopct="%1.1f%%", colors=PALETTE[: len(grouped)], startangle=90, wedgeprops={"edgecolor": "white"})
    ax.set_title("Units Sold Share by Product Category")
    return _save(fig, "category_breakdown")


def plot_model_comparison(results: dict) -> str:
    """Grouped bar comparing test R² across every candidate model."""
    names = list(results.keys())
    r2_vals = [results[n]["test_r2"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, r2_vals, color=PALETTE[: len(names)])
    ax.set_title("Model Comparison: Test R² by Candidate")
    ax.set_ylabel("Test R²")
    ax.tick_params(axis="x", rotation=15)
    return _save(fig, "model_comparison")


def plot_residuals(y_test: pd.Series, y_pred: pd.Series) -> str:
    """Histogram of prediction residuals (actual minus predicted)."""
    residuals = y_test.values - y_pred.values
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=30, color=PALETTE[6], edgecolor="white")
    ax.axvline(0, linestyle="--", color=PALETTE[4])
    ax.set_title("Prediction Residuals (Actual - Predicted)")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    return _save(fig, "residuals")


def generate_all_static_plots(df: pd.DataFrame, ml_results: dict) -> dict:
    """Generate and save every static PNG chart, returning a dict of {name: filepath}."""
    return {
        "budget_by_channel": plot_budget_by_channel(df),
        "conversion_by_city": plot_conversion_by_city(df),
        "units_sold_trend": plot_units_sold_trend(df),
        "channel_efficiency": plot_channel_efficiency(df),
        "category_breakdown": plot_category_breakdown(df),
        "correlation_heatmap": plot_correlation_heatmap(df),
        "model_comparison": plot_model_comparison(ml_results["results"]),
        "feature_importance": plot_feature_importance(ml_results["feature_importance"]),
        "actual_vs_predicted": plot_actual_vs_predicted(ml_results["y_test"], ml_results["y_pred"]),
        "residuals": plot_residuals(ml_results["y_test"], ml_results["y_pred"]),
    }
