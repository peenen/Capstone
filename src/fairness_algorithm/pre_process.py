# src/fairness_algorithm/pre_process.py

import pandas as pd


def group_by_user_activity(df: pd.DataFrame, percentile_threshold: int = 50) -> pd.DataFrame:
    """
    Add 'user_activity_group' (1=high, 0=low) based on interaction counts percentile.
    """
    if 'user_id' not in df.columns:
        return df
    user_counts = df.groupby('user_id').size().reset_index(name='interaction_count')
    threshold = user_counts['interaction_count'].quantile(percentile_threshold / 100)
    user_counts['user_activity_group'] = (user_counts['interaction_count'] >= threshold).astype(int)
    return df.merge(user_counts[['user_id', 'user_activity_group']], on='user_id', how='left')


def group_by_item_popularity(df: pd.DataFrame, top_percent: int = 20) -> pd.DataFrame:
    """
    Add 'item_popularity_group' (1=popular, 0=non-popular) based on top percentile by interactions.
    """
    if 'item_id' not in df.columns:
        return df
    item_counts = df.groupby('item_id').size().reset_index(name='interaction_count')
    threshold = item_counts['interaction_count'].quantile(1 - top_percent / 100)
    item_counts['item_popularity_group'] = (item_counts['interaction_count'] >= threshold).astype(int)
    return df.merge(item_counts[['item_id', 'item_popularity_group']], on='item_id', how='left')