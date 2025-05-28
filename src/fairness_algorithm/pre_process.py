def group_by_interaction_frequency(df):
    """
    分组：根据用户或物品的交互频率划分热门组与长尾组
    Grouping: Split users/items into popular vs. long-tail groups based on interaction frequency

    TODO: 根据交互频率打标签，增加 group_id 字段（如 top 20% vs. bottom 80%）
    TODO: Add a group_id column based on interaction frequency (e.g., top 20% vs. bottom 80%)
    """
    return df

def group_by_user_activity(df):
    """
    分组：按用户活跃度（交互次数）划分为高活跃与低活跃用户
    Grouping: Divide users into high-activity and low-activity groups based on total interactions

    TODO: 统计每个用户的交互次数，并按百分比阈值分组
    TODO: Count interactions per user and group users by percentile thresholds
    """
    return df

def group_by_item_popularity(df):
    """
    分组：根据物品被交互次数划分热门与非热门物品
    Grouping: Group items by popularity (e.g., top 20% most clicked/rated items)

    TODO: 计算每个 item 的曝光/评分次数，按比例分组
    TODO: Calculate exposure counts and assign items into popularity-based groups
    """
    return df

def resample_by_sensitive_attributes(df):
    """
    重采样：根据敏感属性（如性别、年龄）调整组别分布
    Resample: Adjust group balance using sensitive attributes (e.g., gender-based upsampling)

    TODO: 使用分组重采样策略（如 SMOTE、随机欠采样）实现数据均衡
    TODO: Apply resampling techniques (e.g., SMOTE or undersampling) to balance group sizes
    """
    return df
