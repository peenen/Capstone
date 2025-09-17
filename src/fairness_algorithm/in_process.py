import pandas as pd
from src.models.baseline_models import MFRecommender, LightGCNRecommender

def train_and_predict(model_name, config, train_df, val_df, test_df, group_info=None):
    """
    训练基线模型（MF / LightGCN），并对 test_df 的每个用户生成 Top-K 推荐。
    返回：preds_df[user_id, item_id, prediction]
    """
    model_name = (model_name or "").lower()

    # 读取/组装 UGF 超参
    ugf_cfg = {
        "enable_ugf": config.get("in_process", {}).get("enable_ugf", False),
        "lambda_ugf": config.get("in_process", {}).get("lambda_ugf", 0.2),
        "tau": config.get("in_process", {}).get("tau", 0.7),
        "use_js_divergence": config.get("in_process", {}).get("use_js_divergence", True),
        "beta_user_personalization": config.get("in_process", {}).get("beta_user_personalization", 0.0),
        "target_mix": config.get("in_process", {}).get("target_mix", {"popular": 0.6, "unpopular": 0.4}),
        "min_quota": config.get("in_process", {}).get("min_quota", None),  # 预留硬约束位
    }
    if group_info is None:
        group_info = {}
    group_info = {**group_info, "ugf_config": ugf_cfg}

    # === 选择并实例化模型 ===
    if model_name == "mf":
        model = MFRecommender(config)
    elif model_name == "lightgcn":
        model = LightGCNRecommender(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # === 训练 ===
    model.fit(train_df, val_df, group_info)

    # === 预测 Top-K ===
    k = int(config.get("evaluation", {}).get("topk", 20))
    preds = []
    # 用 test_df 中实际出现的用户
    for uid in sorted(pd.unique(test_df["user_id"].astype(int))):
        topk = model.recommend_topk(int(uid), k)
        for iid, score in topk:
            preds.append({"user_id": int(uid), "item_id": int(iid), "prediction": float(score)})

    return pd.DataFrame(preds)
