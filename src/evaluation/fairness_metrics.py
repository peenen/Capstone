import numpy as np
import pandas as pd

def evaluate_gini_index(predictions):
    counts = predictions['item_id'].value_counts().values
    counts = np.sort(counts)
    n = len(counts)
    cumcounts = np.cumsum(counts)
    gini = (n+1 - 2*np.sum(cumcounts)/cumcounts[-1])/n
    return gini

def evaluate_kl_divergence(predictions, ground_truth):
    p = predictions['item_id'].value_counts(normalize=True)
    q = ground_truth['item_id'].value_counts(normalize=True)
    all_items = set(p.index).union(set(q.index))
    p = p.reindex(all_items, fill_value=0.0001)
    q = q.reindex(all_items, fill_value=0.0001)
    kl = np.sum(p*np.log(p/q))
    return kl

def evaluate_recall_dispersion(predictions):
    groups = predictions['user_id'].unique()
    recalls = []
    for u in groups:
        rec_count = predictions[predictions['user_id']==u].shape[0]
        recalls.append(rec_count)
    return np.std(recalls)