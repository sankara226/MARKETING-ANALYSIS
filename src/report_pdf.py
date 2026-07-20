"""Multi-page, modern reportlab PDF report: cover, exec summary, data quality, stats, ML, insights."""

from __future__ import annotations

from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.data_loader import REPORT_PATH

NAVY = colors.HexColor("#1f2937")
ACCENT = colors.HexColor("#2563eb")
TEAL = colors.HexColor("#0891b2")
LIGHT_GRAY = colors.HexColor("#f3f4f6")
BORDER_GRAY = colors.HexColor("#d1d5db")
TEXT_GRAY = colors.HexColor("#374151")

STYLES = {
    "cover_title": ParagraphStyle("cover_title", fontName="Helvetica-Bold", fontSize=30, textColor=colors.white, leading=36),
    "cover_subtitle": ParagraphStyle("cover_subtitle", fontName="Helvetica", fontSize=14, textColor=colors.white, leading=20, spaceBefore=12),
    "cover_meta": ParagraphStyle("cover_meta", fontName="Helvetica", fontSize=10, textColor=colors.HexColor("#cbd5e1"), spaceBefore=40),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=20, textColor=NAVY, spaceBefore=0, spaceAfter=14),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=14, textColor=ACCENT, spaceBefore=14, spaceAfter=8),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10.5, textColor=TEXT_GRAY, leading=15, spaceAfter=8),
    "stat_number": ParagraphStyle("stat_number", fontName="Helvetica-Bold", fontSize=20, textColor=ACCENT, alignment=1),
    "stat_label": ParagraphStyle("stat_label", fontName="Helvetica", fontSize=9, textColor=TEXT_GRAY, alignment=1),
    "caption": ParagraphStyle("caption", fontName="Helvetica-Oblique", fontSize=9, textColor=colors.HexColor("#6b7280"), spaceBefore=4, spaceAfter=12),
    "rec": ParagraphStyle("rec", fontName="Helvetica", fontSize=10.5, textColor=TEXT_GRAY, leading=15, spaceAfter=10, leftIndent=10),
}


def _cover_page(story: list, meta: dict) -> None:
    """Build the cover page flowables: title block on a navy band plus generation metadata."""
    story.append(Spacer(1, 1.6 * inch))
    cover_table = Table(
        [
            [Paragraph("Marketing Campaign Analysis", STYLES["cover_title"])],
            [Paragraph("Data Cleaning, Statistics, Machine Learning &amp; Business Insights Report", STYLES["cover_subtitle"])],
            [Paragraph(f"Generated {datetime.now().strftime('%B %d, %Y at %H:%M')}", STYLES["cover_meta"])],
        ],
        colWidths=[6.5 * inch],
    )
    cover_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("LEFTPADDING", (0, 0), (-1, -1), 30),
                ("RIGHTPADDING", (0, 0), (-1, -1), 30),
                ("TOPPADDING", (0, 0), (-1, 0), 40),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 60),
            ]
        )
    )
    story.append(cover_table)
    story.append(PageBreak())


def _stat_row(stats: list[tuple[str, str]]) -> Table:
    """Render a row of callout boxes, each with a big number and a label underneath."""
    numbers = [Paragraph(v, STYLES["stat_number"]) for _, v in stats]
    labels = [Paragraph(k, STYLES["stat_label"]) for k, _ in stats]
    table = Table([numbers, labels], colWidths=[1.5 * inch] * len(stats))
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY),
                ("BOX", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return table


def _df_to_table(df, max_rows: int = 10) -> Table:
    """Convert a small DataFrame into a styled reportlab Table."""
    header = [str(df.index.name or "")] + [str(c) for c in df.columns]
    rows = [header]
    for idx, row in df.head(max_rows).iterrows():
        rows.append([str(idx)] + [f"{v:,.2f}" if isinstance(v, float) else str(v) for v in row])
    table = Table(rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def build_report(
    df_raw_len: int,
    df_clean,
    stats_insights: dict,
    ml_results: dict,
    business_insights: list[dict],
    recommendations: list[str],
    static_figures: dict,
    path=REPORT_PATH,
) -> None:
    """Assemble and write the full multi-page PDF report to outputs/report.pdf."""
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        title="Marketing Campaign Analysis Report",
    )
    story: list = []

    _cover_page(story, {})

    rows_before, rows_after = df_raw_len, len(df_clean)
    pct_retained = 100 * rows_after / rows_before
    avg_conversion = df_clean["conversion_rate"].mean()
    total_budget = df_clean["budget_spent"].sum()

    story.append(Paragraph("Executive Summary", STYLES["h1"]))
    story.append(
        Paragraph(
            "This report analyzes a messy, multi-currency, multilingual marketing campaign dataset spanning "
            f"{df_clean['country'].nunique()} countries and {df_clean['campaign_channel'].nunique()} channels. "
            "After a fuzzy-matching cleaning pipeline resolved typos, currency formats, and geo-inconsistencies, "
            f"{rows_after} of {rows_before} rows ({pct_retained:.1f}%) were retained for analysis. "
            "A machine learning model was trained to predict conversion rate from campaign attributes, and "
            "concrete budget-reallocation recommendations were derived directly from the cleaned data.",
            STYLES["body"],
        )
    )
    story.append(Spacer(1, 10))
    story.append(
        _stat_row(
            [
                ("Rows Analyzed", f"{rows_after:,}"),
                ("Avg Conversion", f"{avg_conversion:.2f}%"),
                ("Total Budget", f"${total_budget:,.0f}"),
                ("Best Model R²", f"{ml_results['best_test_r2']:.3f}"),
            ]
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("Data Quality &amp; Cleaning Summary", STYLES["h1"]))
    story.append(
        Paragraph(
            f"The raw dataset contained {rows_before} rows with typos, mixed currency symbols (€/$), percent "
            "signs, decimal commas, duplicated/garbled category strings, inconsistent date formats, and 1-20% "
            f"missing values per column. After cleaning, {rows_after} rows ({pct_retained:.1f}%) remained. "
            "Corrections applied: normalized text casing and accents; fuzzy-matched typo variants of country, "
            "city, product, category and channel names against a canonical vocabulary (threshold 0.72 similarity); "
            "corrected country values to match known cities; stripped currency/percent symbols and coerced "
            "numeric columns; parsed inconsistent date formats to ISO; and imputed remaining gaps via column "
            "mean (numeric) or mode (categorical). The 'impressions' column was dropped entirely — its values "
            "were too corrupted and inconsistently scaled to trust for analysis.",
            STYLES["body"],
        )
    )
    missing_before = stats_insights["meta"]["missing_values"]
    story.append(Paragraph(f"Rows before cleaning: {rows_before} &rarr; after cleaning: {rows_after}", STYLES["body"]))
    story.append(PageBreak())

    story.append(Paragraph("Descriptive Statistics", STYLES["h1"]))
    story.append(Paragraph("Budget Spent by Campaign Channel", STYLES["h2"]))
    story.append(Image(static_figures["budget_by_channel"], width=5.5 * inch, height=3.4 * inch))
    story.append(Paragraph("Total budget allocated across each marketing channel.", STYLES["caption"]))

    story.append(Paragraph("Conversion Rate by City", STYLES["h2"]))
    story.append(Image(static_figures["conversion_by_city"], width=5.5 * inch, height=3.4 * inch))
    story.append(Paragraph("Average conversion rate per city, cleaned and geo-corrected.", STYLES["caption"]))
    story.append(PageBreak())

    story.append(Paragraph("Monthly Units Sold Trend", STYLES["h2"]))
    story.append(Image(static_figures["units_sold_trend"], width=6 * inch, height=3.4 * inch))
    story.append(Paragraph("Time trend showing total units sold aggregated by month.", STYLES["caption"]))
    story.append(PageBreak())

    story.append(Paragraph("Budget Efficiency by Channel", STYLES["h2"]))
    story.append(Image(static_figures["channel_efficiency"], width=5.5 * inch, height=3.4 * inch))
    story.append(Paragraph("Units sold per dollar spent, ranking channels by return on budget.", STYLES["caption"]))

    story.append(Paragraph("Product Category Breakdown", STYLES["h2"]))
    story.append(Image(static_figures["category_breakdown"], width=4 * inch, height=4 * inch))
    story.append(Paragraph("Share of total units sold by product category.", STYLES["caption"]))
    story.append(PageBreak())

    story.append(Paragraph("Correlation Between Campaign Metrics", STYLES["h2"]))
    story.append(Image(static_figures["correlation_heatmap"], width=4.2 * inch, height=3.6 * inch))
    story.append(Paragraph("Pearson correlation across budget, clicks, conversion rate and units sold.", STYLES["caption"]))
    story.append(PageBreak())

    story.append(Paragraph("Machine Learning: Conversion Rate Prediction", STYLES["h1"]))
    story.append(
        Paragraph(
            "A regression model was trained to predict conversion_rate from campaign attributes (country, city, "
            "product, category, channel, budget, clicks, units sold, and date-derived month/day-of-week). Three "
            "candidates were compared via 5-fold cross-validation and a held-out test set: Linear Regression, "
            "Random Forest, and Gradient Boosting.",
            STYLES["body"],
        )
    )
    model_rows = [["Model", "CV R² (mean)", "Test R²", "Test RMSE"]]
    for name, res in ml_results["results"].items():
        marker = " (best)" if name == ml_results["best_model_name"] else ""
        model_rows.append([name + marker, f"{res['cv_r2_mean']:.3f}", f"{res['test_r2']:.3f}", f"{res['test_rmse']:.3f}"])
    model_table = Table(model_rows, repeatRows=1)
    model_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(Spacer(1, 6))
    story.append(model_table)
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            f"Winning model: <b>{ml_results['best_model_name']}</b> with test R² = {ml_results['best_test_r2']:.3f} "
            f"and RMSE = {ml_results['best_test_rmse']:.3f}.",
            STYLES["body"],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("Model Comparison", STYLES["h2"]))
    story.append(Image(static_figures["model_comparison"], width=5.5 * inch, height=3.4 * inch))
    story.append(Paragraph("Test R² for each candidate model; higher is better.", STYLES["caption"]))
    story.append(PageBreak())

    story.append(Paragraph("Top Feature Importances", STYLES["h2"]))
    story.append(Image(static_figures["feature_importance"], width=5.5 * inch, height=4.1 * inch))
    story.append(Paragraph("Actual vs Predicted Conversion Rate", STYLES["h2"]))
    story.append(Image(static_figures["actual_vs_predicted"], width=4.2 * inch, height=4.2 * inch))
    story.append(PageBreak())

    story.append(Paragraph("Prediction Residuals", STYLES["h2"]))
    story.append(Image(static_figures["residuals"], width=5 * inch, height=3.6 * inch))
    story.append(Paragraph("Distribution of actual minus predicted conversion rate; centered on zero indicates unbiased predictions.", STYLES["caption"]))
    story.append(PageBreak())

    story.append(Paragraph("Business Insights &amp; Recommendations", STYLES["h1"]))
    for insight in business_insights:
        story.append(Paragraph(insight["title"], STYLES["h2"]))
        story.append(Paragraph(insight["detail"], STYLES["body"]))

    story.append(Paragraph("Recommendations", STYLES["h2"]))
    for i, rec in enumerate(recommendations, start=1):
        story.append(Paragraph(f"{i}. {rec}", STYLES["rec"]))
    story.append(PageBreak())

    story.append(Paragraph("Appendix: Methodology", STYLES["h1"]))
    story.append(
        Paragraph(
            "Cleaning: string normalization, fuzzy canonicalization (difflib SequenceMatcher, threshold 0.72), "
            "currency/percent stripping, numeric coercion, date parsing, city-implies-country geo correction, "
            "and mean/mode imputation. Statistics: pandas groupby aggregations, pivot tables, and Pearson "
            "correlation. Machine learning: scikit-learn ColumnTransformer with one-hot encoding, 80/20 "
            "train/test split, 5-fold cross-validation, model selection by test R²/RMSE. Visualization: Plotly "
            "for the interactive dashboard, matplotlib/seaborn for static report charts. Report generation: "
            "reportlab platypus.",
            STYLES["body"],
        )
    )

    doc.build(story)
