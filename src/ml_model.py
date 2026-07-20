"""Regression pipeline predicting conversion_rate: feature engineering, model selection, prediction."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data_loader import MODEL_PATH

TARGET = "conversion_rate"
CATEGORICAL_FEATURES = ["country", "city", "product_name", "product_category", "campaign_channel"]
NUMERIC_FEATURES = ["budget_spent", "clicks", "units_sold", "month", "day_of_week"]

CANDIDATE_MODELS = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add month/day-of-week date features on top of the cleaned campaign dataframe."""
    df = df.copy()
    dates = pd.to_datetime(df["campaign_date"], errors="coerce")
    df["month"] = dates.dt.month.fillna(dates.dt.month.median())
    df["day_of_week"] = dates.dt.dayofweek.fillna(dates.dt.dayofweek.median())
    return df


def _build_pipeline(model) -> Pipeline:
    """Wrap a regressor in a preprocessing pipeline with one-hot encoded categoricals."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def train_and_select_model(df: pd.DataFrame) -> dict:
    """Train candidate models, cross-validate, and pick the best by test R²/RMSE."""
    df = engineer_features(df)
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    best_name, best_pipeline, best_r2 = None, None, -np.inf

    for name, model in CANDIDATE_MODELS.items():
        pipeline = _build_pipeline(model)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        test_r2 = r2_score(y_test, preds)
        test_rmse = np.sqrt(mean_squared_error(y_test, preds))

        results[name] = {
            "cv_r2_mean": cv_scores.mean(),
            "cv_r2_std": cv_scores.std(),
            "test_r2": test_r2,
            "test_rmse": test_rmse,
        }

        if test_r2 > best_r2:
            best_name, best_pipeline, best_r2 = name, pipeline, test_r2

    best_preds = best_pipeline.predict(X_test)

    feature_importance = _extract_feature_importance(best_pipeline)

    return {
        "results": results,
        "best_model_name": best_name,
        "best_pipeline": best_pipeline,
        "best_test_r2": best_r2,
        "best_test_rmse": results[best_name]["test_rmse"],
        "y_test": y_test.reset_index(drop=True),
        "y_pred": pd.Series(best_preds),
        "feature_importance": feature_importance,
    }


def _extract_feature_importance(pipeline: Pipeline) -> pd.Series | None:
    """Pull out feature importances from tree-based models, or coefficients from linear ones."""
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names = list(cat_names) + NUMERIC_FEATURES

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        return None

    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


def save_model(pipeline: Pipeline, path=MODEL_PATH) -> None:
    """Persist the trained sklearn pipeline to disk with joblib."""
    joblib.dump(pipeline, path)


def load_model(path=MODEL_PATH) -> Pipeline:
    """Load a previously trained sklearn pipeline from disk."""
    return joblib.load(path)


def predict_new_data(pipeline: Pipeline, new_df: pd.DataFrame) -> np.ndarray:
    """Predict conversion_rate on new, already-cleaned campaign rows."""
    new_df = engineer_features(new_df)
    return pipeline.predict(new_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES])
