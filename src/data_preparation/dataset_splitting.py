from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(df: pd.DataFrame, split_cfg: dict):
    train_ratio = split_cfg.get("train_ratio", 0.8)
    val_ratio = split_cfg.get("val_ratio", 0.1)
    test_ratio = split_cfg.get("test_ratio", 0.1)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "split ratios must sum to 1"
    train_df, temp_df = train_test_split(df, test_size=(1 - train_ratio), random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)