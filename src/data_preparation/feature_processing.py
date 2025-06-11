from sklearn.preprocessing import StandardScaler
import pandas as pd


def standardize_numeric(df, columns):
    # Standardize numeric features like age
    # 记得写在config中补充，columns (list): 需要标准化的数值型列名列表，如 ["age", "rating", "timestamp"]
    if not columns:
    return df 

    # 创建一个副本避免修改原始数据
    df = df.copy()

    X = df[columns]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df[columns] = X_scaled
    return df
