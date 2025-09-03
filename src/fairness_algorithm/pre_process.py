import pandas as pd

def group_by_user_activity(df: pd.DataFrame, percentile: float = 50.0) -> pd.DataFrame:
    df = df.copy()
    user_counts = df.groupby("user_id").size()
    threshold = user_counts.quantile(percentile / 100.0)
    df["user_group"] = df["user_id"].apply(lambda u: "high" if user_counts[u] >= threshold else "low")
    return df

def group_by_item_popularity(df: pd.DataFrame, top_percent: float = 20.0) -> pd.DataFrame:
    df = df.copy()
    item_counts = df.groupby("item_id").size()
    threshold = item_counts.quantile(1 - top_percent / 100.0)
    df["item_group"] = df["item_id"].apply(lambda i: "popular" if item_counts[i] >= threshold else "unpopular")
    return df