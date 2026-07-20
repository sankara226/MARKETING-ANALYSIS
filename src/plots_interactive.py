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


def build_dashboard_html(df: pd.DataFrame, ml_results: dict, path=DASHBOARD_PATH) -> None:
    """Assemble all interactive figures into a single responsive HTML dashboard page."""
    figures = [
        fig_budget_vs_conversion_by_channel(df),
        fig_geographic_breakdown(df),
        fig_time_trend(df),
        fig_feature_importance(ml_results["feature_importance"]),
        fig_actual_vs_predicted(ml_results["y_test"], ml_results["y_pred"]),
    ]

    divs = []
    for i, fig in enumerate(figures):
        include_js = "cdn" if i == 0 else False
        divs.append(fig.to_html(full_html=False, include_plotlyjs=include_js, div_id=f"plot-{i}"))

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
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 24px; padding: 24px; max-width: 1400px; margin: 0 auto; }}
  .card {{ background: white; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); padding: 12px; overflow-x: auto; }}
  @media (min-width: 1100px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} .card.full {{ grid-column: 1 / -1; }} }}
</style>
</head>
<body>
<header>
  <h1>Marketing Campaign Analysis Dashboard</h1>
  <p>Interactive exploration of budget efficiency, geography, trends and the conversion-rate prediction model.</p>
</header>
<div class="grid">
  <div class="card full">{divs[0]}</div>
  <div class="card">{divs[1]}</div>
  <div class="card">{divs[2]}</div>
  <div class="card">{divs[3]}</div>
  <div class="card">{divs[4]}</div>
</div>
</body>
</html>"""

    with open(path, "w") as f:
        f.write(html)
