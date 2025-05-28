import pandas as pd
from src.data_preparation import (
    data_cleaning,
    dataset_balancing,
    dataset_splitting,
    dimension_reduction,
    feature_processing,
)
from src.fairness_algorithm import pre_process, in_process, post_process
from src.models.baseline_models import MFModel, LightGCNModel
from src.evaluation import quality_metrics, fairness_metrics


def run_pipeline(config):
    """
    执行推荐系统主流程，串联数据处理、公平性策略、模型训练与评估
    Run the full recommendation pipeline with data preparation, fairness, training, and evaluation
    """

    # ========= Step 1: Load raw data 加载原始数据 =========
    df = pd.read_csv(config["data_path"])
    print("[Step 1] Dataset loaded.")

    # ========= Step 2: Data Cleaning 数据清洗 =========
    df = data_cleaning.handle_duplicate_values(df)
    df = data_cleaning.handle_missing_values(df, method=config["data_cleaning"]["fill_missing"])
    df = data_cleaning.detect_outliers(df, method=config["data_cleaning"]["outlier_method"])
    print("[Step 2] Data cleaned.")

    # ========= Step 3: Data Balancing 数据平衡处理（可选） =========
    method = config["balancing"]["method"]
    if method == "random":
        df = dataset_balancing.random_sampling(df)
    elif method == "smote":
        df = dataset_balancing.smote_sampling(df)
    elif method == "cleaning+sampling":
        df = dataset_balancing.cleaning_plus_sampling(df)
    elif method == "cluster":
        df = dataset_balancing.cluster_based_sampling(df)
    print(f"[Step 3] Balancing method applied: {method}")

    # ========= Step 4: Dimension Reduction 降维（可选） =========
    if config["dimension_reduction"]["enable"]:
        df = dimension_reduction.apply_pca(df, config["dimension_reduction"])
        print("[Step 4] Dimension reduction applied.")

    # ========= Step 5: Feature Processing 特征处理 =========
    df = feature_processing.standardize_numeric(df, config["feature_processing"]["numeric_columns"])
    print("[Step 5] Feature processing completed.")

    # ========= Step 6: Pre-processing Fairness 公平性预处理 =========
    df = pre_process.group_by_interaction_frequency(df)
    df = pre_process.group_by_user_activity(df)
    df = pre_process.group_by_item_popularity(df)
    df = pre_process.resample_by_sensitive_attributes(df)
    print("[Step 6] Pre-processing fairness strategies applied.")

    # ========= Step 7: Dataset Splitting 数据集拆分 =========
    train_df, val_df, test_df = dataset_splitting.split_data(df, config["split"])
    print("[Step 7] Data split into train/val/test sets.")

    # ========= Step 8: In-processing Fairness 训练过程中的公平性处理 =========
    train_df = in_process.apply_negative_sampling(train_df, config)
    train_df = in_process.apply_fairneg(train_df, config)
    print("[Step 8] In-processing fairness applied.")

    # ========= Step 9: Model Initialization 模型初始化 =========
    model_name = config["model"]["name"]
    if model_name == "MF":
        model = MFModel(config["model"])
    elif model_name == "LightGCN":
        model = LightGCNModel(config["model"])
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model = in_process.apply_regularization(model, config)
    print(f"[Step 9] Model initialized: {model_name}")

    # ========= Step 10: Model Training 模型训练 =========
    model.fit(train_df, val_df)
    print("[Step 10] Model training complete.")

    # ========= Step 11: Inference 模型预测 =========
    predictions = model.predict(test_df)
    print("[Step 11] Model prediction complete.")

    # ========= Step 12: Post-processing Fairness 重排序处理 =========
    predictions = post_process.apply_reranking(predictions, config)
    print("[Step 12] Post-processing reranking applied.")

    # ========= Step 13: Evaluation 模型评估 =========
    quality = {
        "F1@10": quality_metrics.evaluate_f1_at_k(predictions, test_df),
        "NDCG@10": quality_metrics.evaluate_ndcg_at_k(predictions, test_df)
    }
    fairness = {
        "Gini": fairness_metrics.evaluate_gini_index(predictions),
        "KL": fairness_metrics.evaluate_kl_divergence(predictions, test_df),
        "Recall-Disp": fairness_metrics.evaluate_recall_dispersion(predictions)
    }
    print("[Step 13] Evaluation complete.")

    return {"quality": quality, "fairness": fairness}
