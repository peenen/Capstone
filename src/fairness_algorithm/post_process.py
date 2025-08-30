# src/fairness_algorithm/post_process.py

import pandas as pd
import numpy as np


def apply_reranking(predictions: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Simple fairness-aware reranking: add tiny noise per user to reduce popularity dominance,
    then re-rank within each user.
    """
    if not config.get("post_process", {}).get("enable", False):
        return predictions

    df = predictions.copy()
    if 'pred_rating' not in df.columns:
        return df

    # Add a small user-wise random perturbation
    df['pred_rating'] = df['pred_rating'] + np.random.uniform(-0.01, 0.01, size=len(df))

    # Compute rank per user
    df['rank'] = df.groupby('user_id')['pred_rating'].rank(method='first', ascending=False)
    df = df.sort_values(['user_id', 'rank']).reset_index(drop=True)
    return df