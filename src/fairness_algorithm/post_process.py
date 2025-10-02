import pandas as pd
import numpy as np

def apply_reranking(predictions: pd.DataFrame, config: dict) -> pd.DataFrame:
    if not config.get("post_process", {}).get("enable", False):
        return predictions
    k = config.get("evaluation", {}).get("topk", 20)
    df = predictions.copy()
    np.random.seed(42)
    df['perturb'] = np.random.uniform(-1e-3, 1e-3, size=len(df))
    df['score2'] = df['prediction'] + df['perturb']
    df = df.sort_values(['user_id','score2'], ascending=[True, False]).groupby('user_id').head(k)
    df = df.drop(columns=['perturb','score2']).reset_index(drop=True)
    return df