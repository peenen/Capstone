import pandas as pd

def group_by_user_activity(df):
    user_counts = df.groupby("user_id").size()
    threshold = user_counts.quantile(0.5)
    df["user_group"] = df["user_id"].apply(lambda x: "high" if user_counts[x] >= threshold else "low")
    return df

def group_by_item_popularity(df):
    item_counts = df.groupby("item_id").size()
    threshold = item_counts.quantile(0.8)
    df["item_group"] = df["item_id"].apply(lambda x: "popular" if item_counts[x] >= threshold else "unpopular")
    return df