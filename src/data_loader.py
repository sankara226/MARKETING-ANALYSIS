"""Path configuration and raw CSV loading for the marketing campaign dataset."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

RAW_CSV_PATH = DATA_DIR / "messy_marketing_campaign.csv"
CLEANED_CSV_PATH = OUTPUTS_DIR / "cleaned_marketing_campaign.csv"
MODEL_PATH = OUTPUTS_DIR / "model.joblib"
DASHBOARD_PATH = OUTPUTS_DIR / "dashboard.html"
REPORT_PATH = OUTPUTS_DIR / "report.pdf"


def ensure_output_dirs() -> None:
    """Create outputs/ and outputs/figures/ if they don't already exist."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)


def load_raw_data(path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Load the raw, messy marketing campaign CSV into a DataFrame."""
    return pd.read_csv(path, sep=",", engine="python")
