from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Schema
ALL_FEATURE_COLUMNS = ["ip", "ep", "peak_width", "area", "baseline", "ph", "temperature", 
                       "sample_type", "age", "sex", "diabetes", "ckd", "cvd", "cancer", "inflammation"]
NUMERIC_COLUMNS = ["ip", "ep", "peak_width", "area", "baseline", "ph", "temperature", "age", 
                   "diabetes", "ckd", "cvd", "cancer", "inflammation"]
CATEGORICAL_COLUMNS = ["sample_type", "sex"]
TARGET_COLUMNS = ["cancer_risk_score", "months_to_onset"]

def build_preprocessor(handle_unknown: str = "ignore") -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False))
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_pipe, NUMERIC_COLUMNS),
        ("cat", categorical_pipe, CATEGORICAL_COLUMNS)
    ])

def preprocess_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    df_clean = df.copy()
    # Basic normalization: ensure columns are lowercase for matching
    df_clean.columns = [c.lower() for c in df_clean.columns]
    
    X = df_clean[ALL_FEATURE_COLUMNS].copy()
    y = df_clean[TARGET_COLUMNS].copy()
    
    preprocessor = build_preprocessor()
    return X, y, preprocessor