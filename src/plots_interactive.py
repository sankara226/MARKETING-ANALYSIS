"""Plotly figures assembled into a single responsive interactive HTML dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader import DASHBOARD_PATH

TEMPLATE = "plotly_white"
COLOR_SEQUENCE = px.colors.qualitative.Set2


def fig_budget_vs_conversion_by_channel(df: pd.DataFrame) -> go.Figure:
    """Scatter of budget_spent vs conversion_rate, colored and animated by channel over months."""
    df = df.copy()
    df["month"] = pd.to_datetime(df["campaign_date"], errors="coerce").dt.to_period("M").astype(str)
    fig = px.scatter(
        df.sort_values("month"),
        x="budget_spent",
        y="conversion_rate",
        color="campaign_channel",
        size="units_sold",
        animation_frame="month",
        hover_data=["city", "country", "product_name"],
        color_discrete_sequence=COLOR_SEQUENCE,
        template=TEMPLATE,
        title="Budget Spent vs Conversion Rate by Channel (animated over time)",
    )
    fig.update_layout(margin=dict(t=60, l=10, r=10, b=10))
    return fig


def fig_geographic_breakdown(df: pd.DataFrame) -> go.Figure:
    """Bar chart of total units sold and average conversion rate by country."""
    grouped = df.groupby("country").agg(units_sold=("units_sold", "sum"), conversion_rate=("conversion_rate", "mean")).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=grouped["country"], y=grouped["units_sold"], name="Total Units Sold", marker_color=COLOR_SEQUENCE[0]),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["country"],
            y=grouped["conversion_rate"],
            name="Avg Conversion Rate (%)",
            mode="lines+markers",
            marker_color=COLOR_SEQUENCE[1],
        ),
        secondary_y=True,
    )
    fig.update_layout(title="Geographic Breakdown: Units Sold vs Conversion Rate by Country", template=TEMPLATE, margin=dict(t=60, l=10, r=10, b=10))
    fig.update_yaxes(title_text="Total Units Sold", secondary_y=False)
    fig.update_yaxes(title_text="Avg Conversion Rate (%)", secondary_y=True)
    return fig


def fig_time_trend(df: pd.DataFrame) -> go.Figure:
    """Line chart of daily/monthly budget spend and conversion rate trend over time."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["campaign_date"], errors="coerce")
    trend = df.groupby(df["date"].dt.to_period("M")).agg(
        budget_spent=("budget_spent", "sum"), conversion_rate=("conversion_rate", "mean")
    )
    trend.index = trend.index.astype(str)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=trend.index, y=trend["budget_spent"], name="Total Budget Spent", mode="lines+markers", marker_color=COLOR_SEQUENCE[2]),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=trend.index, y=trend["conversion_rate"], name="Avg Conversion Rate (%)", mode="lines+markers", marker_color=COLOR_SEQUENCE[3]),
        secondary_y=True,
    )
    fig.update_layout(title="Monthly Trend: Budget Spend vs Conversion Rate", template=TEMPLATE, margin=dict(t=60, l=10, r=10, b=10))
    return fig


def fig_feature_importance(feature_importance: pd.Series) -> go.Figure:
    """Horizontal bar chart of the trained model's top feature importances."""
    top = feature_importance.head(15).sort_values()
    fig = px.bar(
        x=top.values,
        y=top.index,
        orientation="h",
        color=top.values,
        color_continuous_scale="Teal",
        template=TEMPLATE,
        title="Top Feature Importances (Best Model)",
        labels={"x": "Importance", "y": "Feature"},
    )
    fig.update_layout(margin=dict(t=60, l=10, r=10, b=10), coloraxis_showscale=False)
    return fig


def fig_actual_vs_predicted(y_test: pd.Series, y_pred: pd.Series) -> go.Figure:
    """Scatter of actual vs predicted conversion_rate with a perfect-prediction reference line."""
    fig = px.scatter(
        x=y_test,
        y=y_pred,
        template=TEMPLATE,
        title="Actual vs Predicted Conversion Rate",
        labels={"x": "Actual Conversion Rate", "y": "Predicted Conversion Rate"},
        opacity=0.6,
        color_discrete_sequence=[COLOR_SEQUENCE[4]],
    )
    min_v, max_v = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
    fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="Perfect Prediction", line=dict(dash="dash", color="gray")))
    fig.update_layout(margin=dict(t=60, l=10, r=10, b=10))
    return fig


# --- Current-value (descriptive) figures ---


def fig_channel_efficiency(df: pd.DataFrame) -> go.Figure:
    """Bar chart of units sold per dollar spent, by channel — a budget-efficiency ranking."""
    grouped = df.groupby("campaign_channel").agg(budget_spent=("budget_spent", "sum"), units_sold=("units_sold", "sum"))
    grouped["units_per_dollar"] = grouped["units_sold"] / grouped["budget_spent"]
    grouped = grouped.sort_values("units_per_dollar", ascending=True).reset_index()
    fig = px.bar(
        grouped,
        x="units_per_dollar",
        y="campaign_channel",
        orientation="h",
        color="units_per_dollar",
        color_continuous_scale="Tealgrn",
        template=TEMPLATE,
        title="Budget Efficiency: Units Sold per $ Spent, by Channel",
        labels={"units_per_dollar": "Units Sold per $", "campaign_channel": "Channel"},
    )
    fig.update_layout(margin=dict(t=60, l=10, r=10, b=10), coloraxis_showscale=False)
    return fig


def fig_category_breakdown(df: pd.DataFrame) -> go.Figure:
    """Pie chart of total units sold share by product category."""
    grouped = df.groupby("product_category")["units_sold"].sum().sort_values(ascending=False).reset_index()
    fig = px.pie(
        grouped,
        names="product_category",
        values="units_sold",
        color_discrete_sequence=COLOR_SEQUENCE,
        template=TEMPLATE,
        title="Units Sold Share by Product Category",
        hole=0.4,
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(margin=dict(t=60, l=10, r=10, b=10))
    return fig


def fig_conversion_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of conversion_rate distribution across all campaigns, with a mean marker."""
    fig = px.histogram(
        df,
        x="conversion_rate",
        nbins=40,
        color_discrete_sequence=[COLOR_SEQUENCE[5]],
        template=TEMPLATE,
        title="Distribution of Conversion Rate Across All Campaigns",
        labels={"conversion_rate": "Conversion Rate (%)"},
    )
    mean_val = df["conversion_rate"].mean()
    fig.add_vline(x=mean_val, line_dash="dash", line_color="#dc2626", annotation_text=f"Mean {mean_val:.2f}%")
    fig.update_layout(margin=dict(t=60, l=10, r=10, b=10))
    return fig


def fig_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Interactive heatmap of correlations between the numeric campaign metrics."""
    corr = df[["budget_spent", "clicks", "conversion_rate", "units_sold"]].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        template=TEMPLATE,
        title="Correlation Between Campaign Metrics",
    )
    fig.update_layout(margin=dict(t=60, l=10, r=10, b=10))
    return fig


# --- Model-prediction figures ---


def fig_model_comparison(results: dict) -> go.Figure:
    """Grouped bar comparing test R² and RMSE across every candidate model."""
    names = list(results.keys())
    r2_vals = [results[n]["test_r2"] for n in names]
    rmse_vals = [results[n]["test_rmse"] for n in names]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=names, y=r2_vals, name="Test R²", marker_color=COLOR_SEQUENCE[0]), secondary_y=False)
    fig.add_trace(
        go.Scatter(x=names, y=rmse_vals, name="Test RMSE", mode="lines+markers", marker_color=COLOR_SEQUENCE[4]),
        secondary_y=True,
    )
    fig.update_layout(title="Model Comparison: Test R² vs RMSE", template=TEMPLATE, margin=dict(t=60, l=10, r=10, b=10))
    fig.update_yaxes(title_text="Test R²", secondary_y=False)
    fig.update_yaxes(title_text="Test RMSE", secondary_y=True)
    return fig


def fig_residuals(y_test: pd.Series, y_pred: pd.Series) -> go.Figure:
    """Histogram of prediction residuals (actual minus predicted), centered on zero."""
    residuals = pd.Series(y_test.values - y_pred.values, name="residual")
    fig = px.histogram(
        residuals,
        x="residual",
        nbins=40,
        color_discrete_sequence=[COLOR_SEQUENCE[6]],
        template=TEMPLATE,
        title="Prediction Residuals (Actual - Predicted Conversion Rate)",
        labels={"residual": "Residual"},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#374151")
    fig.update_layout(margin=dict(t=60, l=10, r=10, b=10))
    return fig


def fig_predicted_vs_actual_by_channel(y_test: pd.Series, y_pred: pd.Series, channel_test: pd.Series) -> go.Figure:
    """Grouped bar comparing average actual vs average model-predicted conversion rate, by channel, on the held-out test set."""
    comparison = pd.DataFrame({"channel": channel_test.values, "actual": y_test.values, "predicted": y_pred.values})
    grouped = comparison.groupby("channel")[["actual", "predicted"]].mean().sort_values("actual", ascending=False).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=grouped["channel"], y=grouped["actual"], name="Actual", marker_color=COLOR_SEQUENCE[0]))
    fig.add_trace(go.Bar(x=grouped["channel"], y=grouped["predicted"], name="Predicted", marker_color=COLOR_SEQUENCE[3]))
    fig.update_layout(
        title="Actual vs Model-Predicted Conversion Rate by Channel (Test Set)",
        template=TEMPLATE,
        barmode="group",
        margin=dict(t=60, l=10, r=10, b=10),
    )
    return fig


def build_dashboard_html(df: pd.DataFrame, ml_results: dict, path=DASHBOARD_PATH) -> None:
    """Assemble all interactive figures into a single responsive HTML dashboard page, grouped into a current-performance section and a model-predictions section."""
    current_figures = [
        fig_budget_vs_conversion_by_channel(df),
        fig_geographic_breakdown(df),
        fig_time_trend(df),
        fig_channel_efficiency(df),
        fig_category_breakdown(df),
        fig_conversion_distribution(df),
        fig_correlation_heatmap(df),
    ]
    prediction_figures = [
        fig_model_comparison(ml_results["results"]),
        fig_feature_importance(ml_results["feature_importance"]),
        fig_actual_vs_predicted(ml_results["y_test"], ml_results["y_pred"]),
        fig_residuals(ml_results["y_test"], ml_results["y_pred"]),
        fig_predicted_vs_actual_by_channel(ml_results["y_test"], ml_results["y_pred"], ml_results["channel_test"]),
    ]

    all_figures = current_figures + prediction_figures
    divs = []
    for i, fig in enumerate(all_figures):
        include_js = "cdn" if i == 0 else False
        divs.append(fig.to_html(full_html=False, include_plotlyjs=include_js, div_id=f"plot-{i}"))

    current_divs = divs[: len(current_figures)]
    prediction_divs = divs[len(current_figures):]

    def _cards(div_list: list[str], full_first: bool = True) -> str:
        cells = []
        for i, d in enumerate(div_list):
            cls = "card full" if (full_first and i == 0) else "card"
            cells.append(f'<div class="{cls}">{d}</div>')
        return "\n  ".join(cells)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Marketing Campaign Analysis Dashboard</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 0; background: #f7f8fa; color: #1f2430; }}
  header {{ background: linear-gradient(135deg, #1f2937, #374151); color: white; padding: 32px 24px; }}
  header h1 {{ margin: 0 0 8px 0; font-size: 28px; }}
  header p {{ margin: 0; opacity: 0.85; }}
  section.section-header {{ max-width: 1400px; margin: 24px auto 0 auto; padding: 0 24px; }}
  section.section-header h2 {{ font-size: 20px; margin: 0 0 4px 0; color: #1f2937; }}
  section.section-header p {{ margin: 0; color: #6b7280; font-size: 14px; }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 24px; padding: 16px 24px 24px 24px; max-width: 1400px; margin: 0 auto; }}
  .card {{ background: white; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); padding: 12px; overflow-x: auto; }}
  @media (min-width: 1100px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} .card.full {{ grid-column: 1 / -1; }} }}
</style>
</head>
<body>
<header>
  <h1>Marketing Campaign Analysis Dashboard</h1>
  <p>Interactive exploration of current campaign performance and the conversion-rate prediction model.</p>
</header>

<section class="section-header">
  <h2>Current Performance</h2>
  <p>Descriptive views of the cleaned campaign data as it stands today.</p>
</section>
<div class="grid">
  {_cards(current_divs)}
</div>

<section class="section-header">
  <h2>Model Predictions</h2>
  <p>How the trained conversion-rate model performs and what it learned.</p>
</section>
<div class="grid">
  {_cards(prediction_divs)}
</div>
</body>
</html>"""

    with open(path, "w") as f:
        f.write(html)
