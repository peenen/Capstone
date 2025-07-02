import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def random_sampling(df):
    """
    对数据集进行随机下采样以平衡用户或物品分布。
    假设目标字段为 'user_id' 或可配置。
    """
    # 可以通过 groupby 来控制哪个维度进行平衡，例如 user_id 或 item_id
    target_col = "user_id"  # 示例，按 user_id 分组后做等量采样（采样成最小用户组的量）

    # 获取每个用户的评分数量
    counts = df[target_col].value_counts()
    min_count = counts.min()

    # 随机下采样到最小样本数
    balanced_df = df.groupby(target_col).sample(n=min_count, random_state=42).reset_index(drop=True)

    return balanced_df

def cleaning_plus_sampling(df):
    """
    先执行基础的数据清洗（如去除极端值），然后进行随机采样。
    """
    from src.data_preparation.data_cleaning import detect_outliers

    # Step 1: 清洗异常值（示例方法，具体看 data_cleaning 模块如何定义）
    df_cleaned = detect_outliers(df, method="iqr")  # 假设 detect_outliers 支持 iqr 方法

    # Step 2: 再做一次随机采样（调用上面的 random_sampling 函数）
    df_balanced = random_sampling(df_cleaned)

    return df_balanced


def cluster_based_sampling(df):
    """
    对用户进行聚类分组，然后在每组中进行采样，保证各类群均衡。
    """

    # Step 1: 构建用户-交互统计表
    user_interactions = df.groupby("user_id").size().reset_index(name="interaction_count")

    # Step 2: 聚类（假设分为3类：高频、中频、低频用户）
    kmeans = KMeans(n_clusters=3, random_state=42)
    user_interactions["cluster"] = kmeans.fit_predict(user_interactions[["interaction_count"]])

    # Step 3: 在每个簇中随机采样相同数量的用户
    min_samples_per_cluster = user_interactions["cluster"].value_counts().min()
    sampled_users = (
        user_interactions.groupby("cluster")
        .sample(n=min_samples_per_cluster, random_state=42)
        .reset_index(drop=True)
    )

    # Step 4: 筛选原始数据中对应的用户
    balanced_df = df[df["user_id"].isin(sampled_users["user_id"])].reset_index(drop=True)
    return balanced_df

# NOTE: SMOTE sampling was removed because SMOTE is used only for classification tasks.
# 注意：SMOTE采样已移除，因为SMOTE仅适用于分类任务。
# def smote_sampling(df):
#     return df