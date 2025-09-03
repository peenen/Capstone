import pandas as pd
import numpy as np
from scipy import stats

def handle_missing_values(df: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if method == "mean":
            fill_value = df[col].mean()
        elif method == "median":
            fill_value = df[col].median()
        elif method == "zero":
            fill_value = 0
        else:
            raise ValueError("Invalid fill method.")
        df[col] = df[col].fillna(fill_value)
    return df

def handle_duplicate_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)

def detect_outliers(df: pd.DataFrame, method: str = "zscore", threshold: float = 3.0) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df.reset_index(drop=True)
    mask = pd.Series(False, index=df.index)
    if method == "zscore":
        for col in numeric_cols:
            col_vals = df[col].astype(float).fillna(df[col].mean())
            z = np.abs(stats.zscore(col_vals))
            mask |= (z > threshold)
    elif method == "iqr":
        for col in numeric_cols:
            q1 = df[col].quantile(0.25); q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr; upper = q3 + threshold * iqr
            mask |= (df[col] < lower) | (df[col] > upper)
    else:
        raise ValueError("Invalid outlier method.")
    return df.loc[~mask].reset_index(drop=True)