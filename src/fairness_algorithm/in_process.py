import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.baseline_models import MF, LightGCN

def train_and_predict(config, train_data, adj=None):
    """
    Train the specified model and return predictions.
    
    Args:
        config (dict): Config with model params.
        train_data (dict): Contains user_ids, item_ids, labels, neg_items(optional).
        adj (torch.sparse.FloatTensor, optional): adjacency matrix (for LightGCN).
    Returns:
        predictions (torch.Tensor): model predictions on (user_ids, item_ids).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_users = config["data"]["num_users"]
    num_items = config["data"]["num_items"]
    model_type = config["model"]["type"]       # "MF" or "LightGCN"
    in_method = config["model"]["in_method"]   # None / "regularization" / "negative_sampling"
    embed_dim = config["model"].get("embed_dim", 64)
    lr = config["train"].get("lr", 1e-3)
    epochs = config["train"].get("epochs", 10)
    batch_size = config["train"].get("batch_size", 512)

    # Initialize model
    if model_type == "MF":
        model = MF(num_users, num_items, embed_dim, in_method).to(device)
    elif model_type == "LightGCN":
        model = LightGCN(num_users, num_items, embed_dim, in_method=in_method).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    user_ids = torch.tensor(train_data["user_ids"], dtype=torch.long)
    item_ids = torch.tensor(train_data["item_ids"], dtype=torch.long)
    labels = torch.tensor(train_data["labels"], dtype=torch.float)
    dataset = TensorDataset(user_ids, item_ids, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for u, i, y in loader:
            u, i, y = u.to(device), i.to(device), y.to(device)
            if in_method == "negative_sampling":
                neg_items = torch.randint(0, num_items, i.shape, device=device)
                loss = model.loss(u, i, adj if model_type=="LightGCN" else None, neg_items=neg_items) \
                       if model_type == "LightGCN" \
                       else model.loss(u, i, neg_items=neg_items)
            else:
                loss = model.loss(u, i, adj) if model_type == "LightGCN" else model.loss(u, i)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Get predictions on training pairs
    model.eval()
    with torch.no_grad():
        if model_type == "LightGCN":
            preds = model(user_ids.to(device), item_ids.to(device), adj.to(device))
        else:
            preds = model(user_ids.to(device), item_ids.to(device))
    return preds.cpu()