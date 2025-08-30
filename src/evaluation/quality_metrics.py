# src/evaluation/quality_metrics.py

import numpy as np
import pandas as pd


def evaluate_f1_at_k(predictions: pd.DataFrame, ground_truth: pd.DataFrame, k: int = 10) -> float:
    """
    Average F1@K across users.
    predictions: DataFrame with columns ['user_id','item_id','pred_rating']
    ground_truth: DataFrame with columns ['user_id','item_id'] (and optionally 'rating')
    """
    if predictions.empty or ground_truth.empty:
        return 0.0

    f1s = []
    for uid, pred_grp in predictions.groupby('user_id'):
        topk_items = pred_grp.sort_values('pred_rating', ascending=False).head(k)['item_id'].tolist()
        true_items = ground_truth.loc[ground_truth['user_id'] == uid, 'item_id'].tolist()
        if not true_items:
            continue
        tp = len(set(topk_items) & set(true_items))
        precision = tp / max(k, 1)
        recall = tp / max(len(true_items), 1)
        denom = precision + recall
        f1 = (2 * precision * recall / denom) if denom > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def evaluate_ndcg_at_k(predictions: pd.DataFrame, ground_truth: pd.DataFrame, k: int = 10) -> float:
    """
    Average NDCG@K across users.
    """
    if predictions.empty or ground_truth.empty:
        return 0.0

    ndcgs = []
    for uid, pred_grp in predictions.groupby('user_id'):
        topk_items = pred_grp.sort_values('pred_rating', ascending=False).head(k)['item_id'].tolist()
        true_items = set(ground_truth.loc[ground_truth['user_id'] == uid, 'item_id'].tolist())
        if not true_items:
            continue
        dcg = 0.0
        for idx, item in enumerate(topk_items):
            if item in true_items:
                dcg += 1.0 / np.log2(idx + 2)
        ideal_len = min(len(true_items), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_len))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else 0.0