import numpy as np
import pandas as pd

def evaluate_precision_at_k(preds_df: pd.DataFrame, ground_truth_df: pd.DataFrame, k: int = 20) -> float:
    if preds_df.empty or ground_truth_df.empty:
        return 0.0
    topk = preds_df.sort_values(['user_id','prediction'], ascending=[True, False]).groupby('user_id').head(k)
    gt = ground_truth_df[['user_id','item_id']].drop_duplicates()
    per_user = []
    for user, group in topk.groupby('user_id'):
        recs = set(group['item_id'].tolist())
        true = set(gt[gt['user_id']==user]['item_id'].tolist())
        if len(true) == 0:
            continue
        per_user.append(len(recs & true) / k)
    return float(np.mean(per_user)) if per_user else 0.0

def evaluate_recall_at_k(preds_df: pd.DataFrame, ground_truth_df: pd.DataFrame, k: int = 20) -> float:
    if preds_df.empty or ground_truth_df.empty:
        return 0.0
    topk = preds_df.sort_values(['user_id','prediction'], ascending=[True, False]).groupby('user_id').head(k)
    gt = ground_truth_df[['user_id','item_id']].drop_duplicates()
    per_user = []
    for user, group in topk.groupby('user_id'):
        recs = set(group['item_id'].tolist())
        true = set(gt[gt['user_id']==user]['item_id'].tolist())
        if len(true) == 0:
            continue
        per_user.append(len(recs & true) / len(true))
    return float(np.mean(per_user)) if per_user else 0.0

def evaluate_ndcg_at_k(preds_df: pd.DataFrame, ground_truth_df: pd.DataFrame, k: int = 20) -> float:
    if preds_df.empty or ground_truth_df.empty:
        return 0.0
    df = preds_df.sort_values(['user_id','prediction'], ascending=[True, False])
    ndcgs = []
    gt = ground_truth_df[['user_id','item_id']].drop_duplicates()
    for user, group in df.groupby('user_id'):
        topk = group.head(k)['item_id'].tolist()
        true = set(gt[gt['user_id']==user]['item_id'].tolist())
        if len(true) == 0:
            continue
        dcg = 0.0
        for i, it in enumerate(topk):
            if it in true:
                dcg += 1.0 / np.log2(i + 2)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true), k)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else 0.0