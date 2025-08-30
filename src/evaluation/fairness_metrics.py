# src/evaluation/fairness_metrics.py

import numpy as np
import pandas as pd
from scipy.stats import entropy


def evaluate_gini_index(predictions: pd.DataFrame) -> float:
    """
    Gini Index over item exposure counts in predictions (lower is better).
    """
    if predictions.empty:
        return 0.0
    counts = predictions['item_id'].value_counts().values.astype(float)
    if counts.size == 0:
        return 0.0
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    cum = np.cumsum(sorted_counts)
    gini = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    return float(gini)


def evaluate_kl_divergence(predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    """
    KL divergence between predicted item distribution and true item distribution.
    """
    if predictions.empty or ground_truth.empty:
        return 0.0
    p_counts = predictions['item_id'].value_counts(normalize=True)
    q_counts = ground_truth['item_id'].value_counts(normalize=True)
    items = list(set(p_counts.index) | set(q_counts.index))
    p = np.array([p_counts.get(i, 1e-12) for i in items])
    q = np.array([q_counts.get(i, 1e-12) for i in items])
    return float(entropy(p, q))


def evaluate_recall_dispersion(predictions: pd.DataFrame) -> float:
    """
    Variance of per-group recall proxy.
    Since we lack per-user ground-truth here, we approximate recall proxy by the
    average ranked share within top-K per group if 'user_activity_group' exists.
    """
    if 'user_activity_group' not in predictions.columns:
        return 0.0

    # Compute average score per user then aggregate by group as a proxy
    recalls = []
    for g, gdf in predictions.groupby('user_activity_group'):
        # proxy: mean of normalized rank position (lower is better) inverted
        gdf = gdf.copy()
        gdf['rank'] = gdf.groupby('user_id')['pred_rating'].rank(ascending=False, method='first')
        # invert rank to [0,1] proxy (assuming top 10 typical)
        gdf['proxy'] = 1.0 / (1.0 + gdf['rank'])
        recalls.append(gdf['proxy'].mean())
    return float(np.var(recalls)) if recalls else 0.0