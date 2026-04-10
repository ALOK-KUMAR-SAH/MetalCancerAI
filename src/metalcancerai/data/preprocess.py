from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------
# Schema (edit here if you rename columns)
# -----------------------------

ELECTROCHEMICAL_NUMERIC = [
    "ip",          # peak current
    "ep",          # peak potential
    "peak_width",
    "area",
    "baseline",
]

CONDITIONS_NUMERIC = [
    "ph",
    "temperature",
]

CONDITIONS_CATEGORICAL = [
    "sample_type",  # urine/saliva
]

DEMOGRAPHICS_NUMERIC = [
    "age",
]

DEMOGRAPHICS_CATEGORICAL = [
    "sex",
]

DISEASE_CONTEXT_BINARY = [
    "diabetes",
    "ckd",
    "cvd",
    "cancer",
    "inflammation",
]

TARGET_COLUMNS = [
    "cancer_risk_score",  # percentage (0-100)
    "months_to_onset",    # time (months)
]


ALL_FEATURE_COLUMNS = (
    ELECTROCHEMICAL_NUMERIC
    + CONDITIONS_NUMERIC
    + CONDITIONS_CATEGORICAL
    + DEMOGRAPHICS_NUMERIC
    + DEMOGRAPHICS_CATEGORICAL
    + DISEASE_CONTEXT_BINARY
)

NUMERIC_COLUMNS = (
    ELECTROCHEMICAL_NUMERIC
    + CONDITIONS_NUMERIC
    + DEMOGRAPHICS_NUMERIC
    + DISEASE_CONTEXT_BINARY
)

CATEGORICAL_COLUMNS = CONDITIONS_CATEGORICAL + DEMOGRAPHICS_CATEGORICAL


# -----------------------------
# Helpers
# -----------------------------

def _normalize_column_name(col: str) -> str:
    """
    Normalize column names to a stable snake_case-ish form:
    - strip whitespace
    - lower
    - replace spaces, hyphens with underscores
    """
    return (
        col.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of df with normalized column names.
    """
    out = df.copy()
    out.columns = [_normalize_column_name(c) for c in out.columns]
    return out


def validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + ". Available columns: "
            + ", ".join(sorted(df.columns))
        )


def coerce_disease_binaries(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """
    Coerce disease context columns to 0/1 where possible.

    Accepts common representations:
      - booleans True/False
      - strings: yes/no, y/n, true/false, 1/0
      - numeric already 0/1
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue

        series = out[c]

        if pd.api.types.is_bool_dtype(series):
            out[c] = series.astype(int)
            continue

        if pd.api.types.is_numeric_dtype(series):
            # leave as-is (but still attempt to clean NaNs)
            out[c] = pd.to_numeric(series, errors="coerce")
            continue

        # treat as string category
        s = series.astype(str).str.strip().str.lower()
        mapping = {
            "true": 1, "false": 0,
            "yes": 1, "no": 0,
            "y": 1, "n": 0,
            "1": 1, "0": 0,
            "t": 1, "f": 0,
        }
        out[c] = s.map(mapping).astype("float")  # keep NaN if unmapped
    return out


def clean_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light cleaning for the two targets:
    - cancer_risk_score coerced to numeric (expects 0-100; validation is optional)
    - months_to_onset coerced to numeric
    """
    out = df.copy()
    for t in TARGET_COLUMNS:
        if t in out.columns:
            out[t] = pd.to_numeric(out[t], errors="coerce")
    return out


# -----------------------------
# Preprocessor builder
# -----------------------------

@dataclass(frozen=True)
class PreprocessConfig:
    """
    Configuration for preprocessing behavior.

    You can later load these values from YAML in scripts if desired.
    """
    # If True, raise if any target value is missing after coercion.
    require_targets: bool = True

    # If True, drop rows that have missing targets (common for supervised training).
    drop_missing_targets: bool = True

    # OneHotEncoder behavior
    handle_unknown: str = "ignore"

    # If True, allow unseen columns (we only select required feature columns).
    allow_extra_columns: bool = True


def build_preprocessor(
    numeric_cols: Optional[Iterable[str]] = None,
    categorical_cols: Optional[Iterable[str]] = None,
    handle_unknown: str = "ignore",
) -> ColumnTransformer:
    numeric_cols = list(numeric_cols or NUMERIC_COLUMNS)
    categorical_cols = list(categorical_cols or CATEGORICAL_COLUMNS)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def preprocess_table(
    df: pd.DataFrame,
    config: Optional[PreprocessConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """
    Prepare feature matrix X and target matrix y from a single-table dataset.

    Returns:
      X: DataFrame containing selected raw feature columns (NOT transformed yet)
      y: DataFrame with the two targets
      preprocessor: sklearn ColumnTransformer to fit/transform X into model-ready arrays

    Typical usage:
      X_raw, y, pre = preprocess_table(df)
      X = pre.fit_transform(X_raw)
    """
    config = config or PreprocessConfig()

    df2 = normalize_columns(df)
    df2 = coerce_disease_binaries(df2, DISEASE_CONTEXT_BINARY)
    df2 = clean_targets(df2)

    # Validate schema
    validate_columns(df2, ALL_FEATURE_COLUMNS + TARGET_COLUMNS)

    # Target handling
    y = df2[TARGET_COLUMNS].copy()
    if config.drop_missing_targets:
        keep = y.notna().all(axis=1)
        df2 = df2.loc[keep].copy()
        y = y.loc[keep].copy()

    if config.require_targets and y.isna().any().any():
        # If drop_missing_targets=False, this can trigger.
        raise ValueError("Targets contain missing values after coercion.")

    # Select features
    X = df2[ALL_FEATURE_COLUMNS].copy()

    # Create preprocessor (fit later in training code)
    preprocessor = build_preprocessor(handle_unknown=config.handle_unknown)

    return X, y, preprocessor