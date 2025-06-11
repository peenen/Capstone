def apply_pca(df, config):
    # Apply PCA with n_components or auto selection
    from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def apply_pca(df: pd.DataFrame, config: dict):
    """    
    参数：
        config (dict): 配置字典，待配置：
            {
                "enable": True,
                "method": "variance",  # 或 "fixed"
                "n_components": 10,     # 当 method="fixed" 时使用
                "variance_threshold": 0.95,  # 当 method="variance" 时使用
                "feature_columns": ["feat_1", "feat_2", ...]  # 要降维的特征列
            }
    
    返回：
        pd.DataFrame: 包含降维后特征的新数据框。
    """

    feature_cols = config.get("feature_columns")
    if not feature_cols:
        raise ValueError("please  provide feature_columns in config")

    X = df[feature_cols]

    # Step 1: 标准化数据（PCA 对量纲敏感）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: 确定 n_components
    method = config.get("method", "fixed")
    if method == "fixed":
        n_components = config.get("n_components")
        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError("当 method='fixed' 时，请提供一个正整数 n_components")
    elif method == "variance":
        variance_threshold = config.get("variance_threshold", 0.95)
        pca = PCA()
        pca.fit(X_scaled)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        print(f"[PCA] 自动选择 {n_components} 个主成分以保留至少 {variance_threshold * 100:.0f}% 的方差")
    else:
        raise ValueError(f"未知的 PCA 方法：{method}")

    # Step 3: 应用 PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Step 4: 构建新的 DataFrame
    new_feature_names = [f"pca_{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca, columns=new_feature_names, index=df.index)

    # Step 5: 合并回原始数据帧，去掉原来的特征列
    df_reduced = df.drop(columns=feature_cols).join(pca_df)

    return df_reduced

