import pandas as pd
from sklearn.cluster import KMeans

def random_sampling(df):
    target_col = "user_id"
    counts = df[target_col].value_counts()
    min_count = counts.min()
    balanced_df = df.groupby(target_col).sample(n=min_count, random_state=42).reset_index(drop=True)
    return balanced_df

def cleaning_plus_sampling(df):
    # Simplified: remove outliers first
    numeric_cols = df.select_dtypes(include='number').columns
    df = df[(np.abs(df[numeric_cols] - df[numeric_cols].mean()) <= 3*df[numeric_cols].std()).all(axis=1)]
    return random_sampling(df)

def cluster_based_sampling(df):
    user_interactions = df.groupby("user_id").size().reset_index(name="interaction_count")
    kmeans = KMeans(n_clusters=3, random_state=42)
    user_interactions["cluster"] = kmeans.fit_predict(user_interactions[["interaction_count"]])
    min_samples_per_cluster = user_interactions["cluster"].value_counts().min()
    sampled_users = user_interactions.groupby("cluster").sample(n=min_samples_per_cluster, random_state=42)
    balanced_df = df[df["user_id"].isin(sampled_users["user_id"])].reset_index(drop=True)
    return balanced_df