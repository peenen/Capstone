from sklearn.model_selection import train_test_split

def split_data(df, config):
    train_ratio = config.get("train_ratio", 0.8)
    val_ratio = config.get("val_ratio", 0.1)
    test_ratio = config.get("test_ratio", 0.1)
    train_df, temp_df = train_test_split(df, test_size=(1 - train_ratio), random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)
    return train_df, val_df, test_df