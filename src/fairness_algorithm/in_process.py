import pandas as pd
from src.models.baseline_models import MFModel, LightGCNModel

def train_and_predict(model_name: str, config: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    model_cfg = {
        "latent_dim": config["model"].get("latent_dim", config["model"].get("embedding_dim", 64)),
        "learning_rate": config["model"]["learning_rate"],
        "epochs": config["model"]["epochs"],
        "batch_size": config["model"]["batch_size"],
        "device": "cuda" if (config["model"].get("device","auto") == "auto" and __import__('torch').cuda.is_available()) else config["model"].get("device","cpu"),
        "reg_weight": config["model"].get("reg_weight", 1e-3),
        "neg_alpha": config["model"].get("neg_alpha", 0.8),
        "n_layers": config["model"].get("n_layers", 2),
        "in_method": config.get("in_process", {}).get("method", None) if config.get("in_process", {}).get("enable", False) else None
    }

    # Prepare group_info mapping for fairness regularization
    group_info = {}
    if "user_group" in train_df.columns:
        group_info["user_group_map"] = train_df[['user_id','user_group']].drop_duplicates().set_index('user_id')['user_group'].to_dict()
    if "item_group" in train_df.columns:
        group_info["item_group_map"] = train_df[['item_id','item_group']].drop_duplicates().set_index('item_id')['item_group'].to_dict()

    if model_name == "MF":
        model = MFModel(model_cfg)
    elif model_name == "LightGCN":
        model = LightGCNModel(model_cfg)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.fit(train_df, val_df=val_df, group_info=group_info)

    # For all test users, get top-k per user and return a DataFrame
    users = sorted(test_df["user_id"].unique().tolist())
    k = config.get("evaluation", {}).get("topk", 20)
    rows = []
    for u in users:
        recs = model.recommend_topk(u, k=k)
        for item_id, score in recs:
            rows.append({"user_id": u, "item_id": item_id, "prediction": score})
    preds_df = pd.DataFrame(rows)
    return preds_df