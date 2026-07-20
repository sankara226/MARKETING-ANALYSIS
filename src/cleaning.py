"""Cleaning pipeline: string canonicalization, numeric coercion, dates, geo-consistency, imputation."""

from __future__ import annotations

import difflib

import numpy as np
import pandas as pd

from src.data_loader import CLEANED_CSV_PATH

NA_TOKEN = "NaN"
NA_LIKE = {"nan", "na", "none", "null", "unknown", "unknowncity", "unknowncountry"}

FLOAT_COLS = ["budget_spent", "clicks", "conversion_rate", "units_sold"]
CATEGORICAL_COLS = ["country", "city", "product_name", "product_category", "campaign_channel"]

CITY_RULES = {
    "sao_paulo": ["saopaulo", "sopaulo"],
    "new_york": ["newyork"],
    "tokyo": ["tokyo"],
    "paris": ["paris", "prs"],
    "berlin": ["berlin"],
    "madrid": ["madrid"],
    "toronto": ["toronto"],
    "rome": ["rome", "roma"],
}
COUNTRY_RULES = {
    "usa": ["usa", "unitedstates"],
    "canada": ["canada", "kanada"],
    "france": ["france", "franse", "frnce", "franc"],
    "spain": ["espana", "espaa", "spain"],
    "japan": ["japan", "jappan", "japon"],
    "germany": ["germany", "germny", "deutschland"],
    "italy": ["italy", "italie", "italia"],
    "brazil": ["brazil", "brasil"],
}
PRODUCT_RULES = {
    "hydra_cream": ["hydracream"],
    "pure_clean_toner": ["purecleantoner", "pctoner"],
    "glow_serum": ["glowserum"],
}
CATEGORY_RULES = {
    "skin_care": ["skincare"],
}
CHANNEL_RULES = {
    "facebook": ["facebook", "fb"],
    "email": ["email"],
    "google_ads": ["googleads"],
    "tiktok": ["tiktok"],
    "instagram": ["instagram", "insta"],
    "billboard": ["billboard"],
}

CITY_TO_COUNTRY = {
    "paris": "france",
    "new_york": "usa",
    "sao_paulo": "brazil",
    "tokyo": "japan",
    "berlin": "germany",
    "madrid": "spain",
    "rome": "italy",
    "toronto": "canada",
}

# Fuzzy-match acceptance threshold: below 0.72 too many unrelated tokens (e.g. "toner"
# vs "toronto") collide; above it, genuine typos ("franse", "germny") stop matching.
FUZZY_MATCH_THRESHOLD = 0.72


def _normalize(series: pd.Series) -> pd.Series:
    """Lowercase, strip accents/symbols and collapse whitespace in a text column."""
    return (
        series.astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9\s_%-]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace("", "nan")
    )


def _strip_variants(series: pd.Series) -> pd.Series:
    """Remove separators/leetspeak digits so typo variants collapse onto the same root."""
    series = series.str.replace(r"[ _\-\s%]", "", regex=True)
    series = series.str.replace("0", "o", regex=False)
    series = series.str.replace("3", "e", regex=False)
    series = series.str.replace("1", "i", regex=False)
    series = series.str.replace("%", "", regex=True)
    return series


def _dedupe_repeat(x: str) -> str:
    """Collapse clean repetition, e.g. 'usausausa' -> 'usa', 'torontotoronto' -> 'toronto'."""
    n = len(x)
    for length in range(1, n // 2 + 1):
        if n % length == 0 and x[:length] * (n // length) == x:
            return x[:length]
    return x


def _best_match(x: str, rules: dict) -> str | None:
    """Return the canonical label for x under rules, or None if below the match threshold."""
    x = _dedupe_repeat(x)

    matched_canons = set()
    best_canon, best_score = None, 0.0
    for canon, roots in rules.items():
        for root in roots:
            score = difflib.SequenceMatcher(None, x, root).ratio()
            if root in x:
                score = max(score, 0.95)
                matched_canons.add(canon)
            if score > best_score:
                best_canon, best_score = canon, score

    # value contains 2+ different known roots concatenated (e.g. "torontotokyo")
    # -> ambiguous/corrupted, can't safely resolve to either -> treat as missing
    if len(matched_canons) > 1:
        return NA_TOKEN

    return best_canon if best_score >= FUZZY_MATCH_THRESHOLD else None


def _canonicalize(series: pd.Series, rules: dict) -> pd.Series:
    """Map a normalized text column onto its canonical vocabulary via fuzzy matching."""
    stripped = _strip_variants(series)
    result = series.copy()
    na_mask = stripped.isin(NA_LIKE)
    result[na_mask] = NA_TOKEN
    matches = stripped[~na_mask].apply(lambda x: _best_match(x, rules))
    result.loc[matches.index] = matches.where(matches.notna(), result.loc[matches.index])
    return result


def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and fuzzy-canonicalize country/city/product/category/channel text columns."""
    df = df.dropna(thresh=df.shape[1] - 2)
    df = df.fillna(NA_TOKEN)
    # impressions is dropped: the column is too corrupted/inconsistent across rows
    # (mixed scales, garbled values) to be trusted for any downstream analysis
    df = df.drop(columns=["impressions"], errors="ignore")

    for col in CATEGORICAL_COLS:
        df[col] = _normalize(df[col])

    df["city"] = _canonicalize(df["city"], CITY_RULES)
    df["country"] = _canonicalize(df["country"], COUNTRY_RULES)
    df["product_name"] = _canonicalize(df["product_name"], PRODUCT_RULES)
    df["product_category"] = _canonicalize(df["product_category"], CATEGORY_RULES)
    df["campaign_channel"] = _canonicalize(df["campaign_channel"], CHANNEL_RULES)

    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].replace(
        to_replace=r"(?i)^(nan|na|n/a|none|null|unknown|unknown city|unknown_city)$",
        value=NA_TOKEN,
        regex=True,
    )
    for col in CATEGORICAL_COLS:
        no_alnum_mask = ~df[col].astype(str).str.contains(r"[a-zA-Z0-9]", regex=True)
        df.loc[no_alnum_mask, col] = NA_TOKEN
    df = df.replace(["NaN", "nan", "None", "null", ""], np.nan)

    return df


def clean_float(df: pd.DataFrame) -> pd.DataFrame:
    """Strip currency/percent symbols and coerce budget/clicks/conversion/units to numeric."""
    for col in FLOAT_COLS:
        series = df[col].astype(str).str.lower()
        series = series.str.replace("beaucoup", "", regex=False)
        series = series.str.replace("budget non communique", "", regex=False)
        series = series.str.replace("budgetnoncommunique", "", regex=False)
        for symbol in ["€", "$", "%", "?", "-"]:
            series = series.str.replace(symbol, "", regex=False)
        series = series.str.replace(r"\s+", "", regex=True)
        series = series.str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(series, errors="coerce")
    return df


def clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse mixed-format campaign_date strings into ISO YYYY-MM-DD."""
    df["campaign_date"] = pd.to_datetime(df["campaign_date"], errors="coerce")
    df["campaign_date"] = df["campaign_date"].dt.strftime("%Y-%m-%d")
    return df


def correct_geo_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    """Fix country to match a known canonical city, and infer country from city if missing."""
    for index, row in df.iterrows():
        city, country = row["city"], row["country"]

        if pd.notna(city) and pd.notna(country):
            if city in CITY_TO_COUNTRY and country != CITY_TO_COUNTRY[city]:
                df.loc[index, "country"] = CITY_TO_COUNTRY[city]
        elif pd.notna(city) and pd.isna(country):
            if city in CITY_TO_COUNTRY:
                df.loc[index, "country"] = CITY_TO_COUNTRY[city]
        # city missing & country present: left as NaN, handled by fill_blanks via mode

    return df


def fill_blanks(df: pd.DataFrame) -> pd.DataFrame:
    """Impute remaining NaNs: mean for numeric columns, mode/ffill for categorical/date."""
    for col in ["budget_spent", "conversion_rate", "clicks", "units_sold"]:
        df[col] = df[col].fillna(df[col].mean())
    for col in ["campaign_channel", "city", "country", "product_category"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    df["campaign_date"] = df["campaign_date"].ffill()
    return df


def run_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline end to end on the raw DataFrame."""
    df = clean_strings(df)
    df = clean_float(df)
    df = clean_dates(df)
    df = correct_geo_inconsistencies(df)
    df = fill_blanks(df)
    return df.reset_index(drop=True)


def export_cleaned_csv(df: pd.DataFrame, path=CLEANED_CSV_PATH) -> None:
    """Write the cleaned DataFrame to outputs/cleaned_marketing_campaign.csv."""
    df.to_csv(path, index=False)
