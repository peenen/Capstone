import pandas as pd
import numpy as np
from scipy import stats

def handle_missing_values(df, method='mean'):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if method == 'mean':
            fill_value = df[col].mean()
        elif method == 'median':
            fill_value = df[col].median()
        elif method == 'zero':
            fill_value = 0
        else:
            raise ValueError("Invalid method. Use 'mean', 'median', or 'zero'.")
        df[col] = df[col].fillna(fill_value)
    return df

def handle_duplicate_values(df):
    return df.drop_duplicates()

def detect_outliers(df, method='zscore', threshold=3.0):
    outlier_flags = pd.DataFrame(index=df.index)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if method == 'zscore':
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(df[col].fillna(0)))
            outlier_flags[col + '_outlier'] = z_scores > threshold
    elif method == 'iqr':
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outlier_flags[col + '_outlier'] = (df[col] < lower_bound) | (df[col] > upper_bound)
    else:
        raise ValueError("Invalid method. Use 'zscore' or 'iqr'.")
    return df