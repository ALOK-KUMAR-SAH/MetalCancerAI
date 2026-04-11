import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple

def get_preprocessor(handle_unknown: str = "ignore") -> ColumnTransformer:
    # Define features based on your dataset schema
    numeric_features = ['ip', 'ep', 'peak_width', 'area', 'baseline', 'ph', 'temperature', 'age']
    categorical_features = ['sample_type', 'sex', 'diabetes', 'ckd', 'cvd', 'cancer', 'inflammation']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown=handle_unknown), categorical_features)
        ]
    )
    return preprocessor

def preprocess_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    # 1. Standardize column names
    df.columns = [col.lower().strip() for col in df.columns]
    
    # 2. NEW: Normalize Categorical Data (Fixes "Male" vs "male" issue)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.lower().str.strip()
    
    # 3. Define Targets
    y = df[['cancer_risk_score', 'months_to_onset']]
    X = df.drop(columns=['cancer_risk_score', 'months_to_onset'])
    
    preprocessor = get_preprocessor()
    return X, y, preprocessor