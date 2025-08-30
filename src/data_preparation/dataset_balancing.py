# src/data_preparation/dataset_balancing.py

import pandas as pd
from sklearn.cluster import KMeans
from src.data_preparation import data_cleaning


def random_sampling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Randomly down-sample each user's interactions to balance user distribution.
    """
    if 'user_id' not in df.columns:
        return df.reset_index(drop=True)
    counts = df['user_id'].value_counts()
    min_count = counts.min()
    balanced_df = df.groupby('user_id', group_keys=False).apply(
        lambda g: g.sample(n=min_count, random_state=42)
    )
    return balanced_df.reset_index(drop=True)


def cleaning_plus_sampling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers first (IQR), then apply random sampling.
    """
    df_cleaned = data_cleaning.detect_outliers(df, method="iqr")
    return random_sampling(df_cleaned)


def cluster_based_sampling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cluster users by interaction count and sample evenly from clusters.
    """
    if 'user_id' not in df.columns:
        return df.reset_index(drop=True)

    user_interactions = df.groupby("user_id").size().reset_index(name="interaction_count")
    if user_interactions["interaction_count"].nunique() < 3:
        # Not enough diversity to cluster; fall back to random
        return random_sampling(df)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    user_interactions["cluster"] = kmeans.fit_predict(user_interactions[["interaction_count"]])

    min_per_cluster = user_interactions["cluster"].value_counts().min()
    sampled_users = (
        user_interactions.groupby("cluster", group_keys=False)
        .apply(lambda g: g.sample(n=min_per_cluster, random_state=42))
        .reset_index(drop=True)
    )

    balanced_df = df[df["user_id"].isin(sampled_users["user_id"])].reset_index(drop=True)
    return balanced_df