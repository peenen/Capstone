class MFModel:
    """
    矩阵分解模型：用于评分预测或隐因子推荐
    Matrix Factorization model: for rating prediction or embedding-based recommendation
    """
    def __init__(self, config):
        """
        初始化模型参数（如嵌入维度、学习率等）
        Initialize model with configuration (e.g., latent dimensions, learning rate)
        """
        self.config = config
        # TODO: 初始化 user/item 嵌入矩阵
        # TODO: Initialize user/item embedding matrices
        pass

    def fit(self, train_df, val_df=None):
        """
        模型训练：基于交互数据拟合用户和物品的潜在向量
        Train the model using user-item interaction data
        """
        pass

    def predict(self, test_df):
        """
        模型预测：为测试集用户生成推荐列表或评分
        Generate predictions for users in the test set
        """
        # TODO: 根据 user-item 向量计算得分，输出推荐列表或评分
        # TODO: Compute user-item scores and output recommendation list or ratings
        return test_df  # 或替换为预测结果列表/字典

class LightGCNModel:
    """
    LightGCN 模型：图卷积推荐模型，建模高阶邻居关系
    LightGCN Model: Graph-based recommender leveraging user-item neighborhood
    """
    def __init__(self, config):
        """
        初始化图结构与参数
        Initialize graph structure and hyperparameters
        """
        self.config = config
        # TODO: 初始化邻接矩阵、嵌入矩阵等
        # TODO: Initialize adjacency matrix, user/item embeddings, etc.
        pass

    def fit(self, train_df, val_df=None):
        """
        训练图卷积推荐模型
        Train LightGCN using sampled user-item graph edges
        """
        # TODO: 构建 user-item 二分图，执行多层传播与优化
        # TODO: Construct bipartite graph and apply multiple propagation layers
        pass

    def predict(self, test_df):
        """
        预测用户的推荐列表
        Generate recommendation scores or top-K items for users
        """
        # TODO: 聚合邻接信息后预测 user-item 评分或 Top-K 推荐
        # TODO: Aggregate graph information and compute recommendation scores
        return test_df  # 或替换为预测结果 DataFrame
