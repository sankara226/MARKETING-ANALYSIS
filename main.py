"""Orchestrates the full pipeline: load -> clean -> export -> stats -> ML -> insights -> dashboard -> PDF."""

from src.business_insights import generate_insights, generate_recommendations
from src.cleaning import export_cleaned_csv, run_cleaning_pipeline
from src.data_loader import (
    CLEANED_CSV_PATH,
    DASHBOARD_PATH,
    REPORT_PATH,
    ensure_output_dirs,
    load_raw_data,
)
from src.ml_model import save_model, train_and_select_model
from src.plots_interactive import build_dashboard_html
from src.plots_static import generate_all_static_plots
from src.report_pdf import build_report
from src.statistics_analysis import compute_aggregations, compute_full_insights, compute_pivot_tables


def main() -> None:
    print("[1/8] Setting up output directories...")
    ensure_output_dirs()

    print("[2/8] Loading raw data...")
    df_raw = load_raw_data()
    print(f"       Raw shape: {df_raw.shape}")

    print("[3/8] Running cleaning pipeline...")
    df_clean = run_cleaning_pipeline(df_raw)
    print(f"       Cleaned shape: {df_clean.shape}")

    print("[4/8] Exporting cleaned CSV...")
    export_cleaned_csv(df_clean)
    print(f"       Saved to {CLEANED_CSV_PATH}")

    print("[5/8] Computing statistics and aggregations...")
    stats_insights = compute_full_insights(df_clean)
    compute_aggregations(df_clean)
    compute_pivot_tables(df_clean)

    print("[6/8] Training ML models to predict conversion_rate...")
    ml_results = train_and_select_model(df_clean)
    save_model(ml_results["best_pipeline"])
    print(f"       Best model: {ml_results['best_model_name']} (R²={ml_results['best_test_r2']:.3f}, RMSE={ml_results['best_test_rmse']:.3f})")

    print("[7/8] Generating business insights, static and interactive plots...")
    insights = generate_insights(df_clean)
    recommendations = generate_recommendations(df_clean, insights)
    static_figures = generate_all_static_plots(df_clean, ml_results)
    build_dashboard_html(df_clean, ml_results)
    print(f"       Dashboard saved to {DASHBOARD_PATH}")

    print("[8/8] Building PDF report...")
    build_report(
        df_raw_len=len(df_raw),
        df_clean=df_clean,
        stats_insights=stats_insights,
        ml_results=ml_results,
        business_insights=insights,
        recommendations=recommendations,
        static_figures=static_figures,
    )
    print(f"       Report saved to {REPORT_PATH}")

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
