import pandas as pd
import numpy as np

def evaluate_f1_at_k(predictions, ground_truth, k=20):
    df = predictions.copy()
    df = df.sort_values('prediction', ascending=False).groupby('user_id').head(k)
    merged = pd.merge(df[['user_id','item_id']], ground_truth[['user_id','item_id']], on=['user_id','item_id'])
    precision = len(merged)/len(df)
    recall = len(merged)/len(ground_truth)
    if precision+recall==0:
        return 0.0
    return 2*precision*recall/(precision+recall)

def evaluate_ndcg_at_k(predictions, ground_truth, k=20):
    df = predictions.copy()
    df = df.sort_values('prediction', ascending=False).groupby('user_id').head(k)
    user_groups = df['user_id'].unique()
    ndcg_list = []
    for u in user_groups:
        rel = df[df['user_id']==u]['item_id'].tolist()
        gt = ground_truth[ground_truth['user_id']==u]['item_id'].tolist()
        dcg = sum([1/np.log2(i+2) for i,it in enumerate(rel) if it in gt])
        idcg = sum([1/np.log2(i+2) for i in range(min(len(gt),k))])
        ndcg_list.append(dcg/idcg if idcg>0 else 0.0)
    return np.mean(ndcg_list)