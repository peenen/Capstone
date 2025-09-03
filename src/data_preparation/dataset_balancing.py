import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

def random_sampling(df: pd.DataFrame) -> pd.DataFrame:
    if "user_id" not in df.columns:
        return df.reset_index(drop=True)
    counts = df["user_id"].value_counts()
    min_count = counts.min()
    balanced = df.groupby("user_id", group_keys=False).apply(lambda g: g.sample(n=min_count, random_state=42))
    return balanced.reset_index(drop=True)

def cleaning_plus_sampling(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) == 0:
        return random_sampling(df)
    df_clean = df.copy()
    for col in numeric_cols:
        col_vals = df_clean[col].astype(float).fillna(df_clean[col].mean())
        z = np.abs((col_vals - col_vals.mean()) / (col_vals.std() + 1e-8))
        df_clean = df_clean[z <= 3]
    return random_sampling(df_clean)

def cluster_based_sampling(df: pd.DataFrame) -> pd.DataFrame:
    if "user_id" not in df.columns:
        return df.reset_index(drop=True)
    user_counts = df.groupby("user_id").size().reset_index(name="count")
    if user_counts.shape[0] < 3:
        return random_sampling(df)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    user_counts["cluster"] = kmeans.fit_predict(user_counts[["count"]])
    min_per_cluster = user_counts["cluster"].value_counts().min()
    sampled_users = user_counts.groupby("cluster", group_keys=False).apply(lambda g: g.sample(n=min_per_cluster, random_state=42))
    users = sampled_users["user_id"].unique()
    balanced = df[df["user_id"].isin(users)].reset_index(drop=True)
    return balanced