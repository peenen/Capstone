# src/data_preparation/dataset_splitting.py

from sklearn.model_selection import train_test_split
import pandas as pd


def split_data(df: pd.DataFrame, config: dict):
    """
    Split dataset into train, validation, and test sets.
    """
    train_ratio = config.get("train_ratio", 0.8)
    val_ratio = config.get("val_ratio", 0.1)
    test_ratio = config.get("test_ratio", 0.1)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    train_df, temp_df = train_test_split(df, test_size=(1 - train_ratio), random_state=42)
    if (val_ratio + test_ratio) == 0:
        return train_df.reset_index(drop=True), pd.DataFrame(), pd.DataFrame()

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        random_state=42
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)