import numpy as np
from scipy.stats import entropy

def evaluate_gini_index(preds_df):
    if preds_df.empty:
        return 0.0
    counts = preds_df['item_id'].value_counts().values.astype(float)
    if counts.size == 0:
        return 0.0
    counts = np.sort(counts)
    n = len(counts)
    cum = np.cumsum(counts)
    gini = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    return float(gini)

def evaluate_kl_divergence(preds_df, ground_truth_df):
    if preds_df.empty or ground_truth_df.empty:
        return 0.0
    p = preds_df['item_id'].value_counts(normalize=True)
    q = ground_truth_df['item_id'].value_counts(normalize=True)
    all_items = set(p.index) | set(q.index)
    p_arr = np.array([p.get(i, 1e-12) for i in all_items])
    q_arr = np.array([q.get(i, 1e-12) for i in all_items])
    return float(entropy(p_arr, q_arr))

def evaluate_recall_dispersion(preds_df, ground_truth_df):
    if preds_df.empty:
        return 0.0
    per_user_counts = preds_df.groupby('user_id').size().values
    return float(np.std(per_user_counts))