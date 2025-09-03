# PyTorch-based MF and LightGCN, with device handling, recommend_topk(user_id,k),
# fairness-aware negative sampling and regularization.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import defaultdict
import scipy.sparse as sp

def sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

class FairnessAwareNegativeSampler:
    def __init__(self, item_counts_map, alpha=0.8):
        # item_counts_map: dict {item_id: count}
        items = sorted(item_counts_map.keys())
        counts = np.array([item_counts_map[i] for i in items], dtype=float)
        counts += 1e-6
        inv = 1.0 / counts
        inv = inv / inv.sum()
        uniform = np.ones_like(inv) / inv.size
        self.probs = (1 - alpha) * uniform + alpha * inv
        self.index_to_item = items

    def sample(self, n):
        idxs = np.random.choice(len(self.index_to_item), size=n, p=self.probs)
        return [self.index_to_item[i] for i in idxs]

class FairnessAwareRegularizer:
    def __init__(self, lambda_fair=1e-3):
        self.lambda_fair = lambda_fair

    def compute_group_variance(self, group_labels, preds_tensor):
        unique = list(set(group_labels))
        if len(unique) <= 1:
            return torch.tensor(0.0, device=preds_tensor.device)
        means = []
        for g in unique:
            mask = torch.tensor([1 if x == g else 0 for x in group_labels], dtype=torch.bool, device=preds_tensor.device)
            if mask.sum() == 0:
                means.append(torch.tensor(0.0, device=preds_tensor.device))
            else:
                means.append(preds_tensor[mask].mean())
        means = torch.stack(means)
        return self.lambda_fair * torch.var(means)

class MFModel:
    class Net(nn.Module):
        def __init__(self, n_users, n_items, dim):
            super().__init__()
            self.user_emb = nn.Embedding(n_users, dim)
            self.item_emb = nn.Embedding(n_items, dim)
            nn.init.normal_(self.user_emb.weight, std=0.01)
            nn.init.normal_(self.item_emb.weight, std=0.01)
        def forward(self, u, i):
            return (self.user_emb(u) * self.item_emb(i)).sum(dim=1)

    def __init__(self, config: dict):
        self.latent_dim = int(config.get("latent_dim", 64))
        self.lr = float(config.get("learning_rate", 0.01))
        self.epochs = int(config.get("epochs", 10))
        self.batch_size = int(config.get("batch_size", 256))
        device_cfg = config.get("device", "auto")
        self.device = torch.device("cuda" if (device_cfg == "auto" and torch.cuda.is_available()) or device_cfg == "cuda" else "cpu")
        self.reg_weight = float(config.get("reg_weight", 1e-3))
        self.in_method = config.get("in_method", None)
        self.neg_alpha = float(config.get("neg_alpha", 0.8))
        # will be set in fit
        self.user2idx = {}
        self.item2idx = {}
        self.idx2user = {}
        self.idx2item = {}
        self.net = None
        self.neg_sampler = None
        self.fair_reg = None

    def _build_maps(self, df):
        users = sorted(df["user_id"].unique().tolist())
        items = sorted(df["item_id"].unique().tolist())
        self.user2idx = {u: idx for idx, u in enumerate(users)}
        self.item2idx = {i: idx for idx, i in enumerate(items)}
        self.idx2user = {idx: u for u, idx in self.user2idx.items()}
        self.idx2item = {idx: i for i, idx in self.item2idx.items()}
        self.n_users = len(users)
        self.n_items = len(items)

    def fit(self, train_df, val_df=None, group_info=None):
        self._build_maps(train_df)
        self.net = MFModel.Net(self.n_users, self.n_items, self.latent_dim).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        users = train_df["user_id"].map(self.user2idx).values
        items = train_df["item_id"].map(self.item2idx).values
        ratings = train_df["rating"].values.astype(np.float32)
        n = len(users)
        # prepare user_pos (global ids)
        user_pos = defaultdict(set)
        for _, r in train_df.iterrows():
            user_pos[int(r["user_id"])].add(int(r["item_id"]))
        # prepare negative sampler if needed
        if self.in_method == "negative_sampling":
            item_counts = train_df["item_id"].value_counts().to_dict()
            self.neg_sampler = FairnessAwareNegativeSampler(item_counts, alpha=self.neg_alpha)
        if self.in_method == "regularization":
            self.fair_reg = FairnessAwareRegularizer(lambda_fair=self.reg_weight)
        for epoch in range(self.epochs):
            perm = np.random.permutation(n)
            total_loss = 0.0
            self.net.train()
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                u_idx = torch.LongTensor(users[idx]).to(self.device)
                i_idx = torch.LongTensor(items[idx]).to(self.device)
                y = torch.FloatTensor(ratings[idx]).to(self.device)
                preds = self.net(u_idx, i_idx)
                if self.in_method == "negative_sampling":
                    neg_items_global = self.neg_sampler.sample(len(idx))
                    neg_local = [ self.item2idx.get(it, random.randint(0, self.n_items-1)) for it in neg_items_global ]
                    neg_i_idx = torch.LongTensor(neg_local).to(self.device)
                    neg_preds = self.net(u_idx, neg_i_idx)
                    loss = -torch.mean(torch.log(torch.sigmoid(preds - neg_preds)))
                else:
                    loss = F.mse_loss(preds, y)
                if self.in_method == "regularization" and group_info is not None:
                    if "user_group_map" in group_info:
                        batch_global_users = [ self.idx2user[int(x)] for x in u_idx.cpu().numpy() ]
                        group_labels = [ group_info["user_group_map"].get(u, "none") for u in batch_global_users ]
                        loss = loss + self.fair_reg.compute_group_variance(group_labels, preds)
                # L2
                l2 = (self.net.user_emb(u_idx).norm(2).pow(2).mean() + self.net.item_emb(i_idx).norm(2).pow(2).mean())
                loss = loss + self.reg_weight * l2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[MF] Epoch {epoch+1}/{self.epochs} loss={total_loss:.4f}")

    def recommend_topk(self, user_id: int, k: int = 20):
        if self.net is None:
            raise RuntimeError("Model not trained.")
        if user_id not in self.user2idx:
            return []
        self.net.eval()
        with torch.no_grad():
            u_local = self.user2idx[user_id]
            u_t = torch.LongTensor([u_local]).to(self.device)
            item_embs = self.net.item_emb.weight  # local indices
            scores = (self.net.user_emb(u_t) * item_embs).sum(dim=1).cpu().numpy()
            topk = np.argsort(-scores)[:k]
            results = [(self.idx2item[idx], float(scores[idx])) for idx in topk]
            return results

class LightGCNModel:
    def __init__(self, config: dict):
        self.latent_dim = int(config.get("latent_dim", 64))
        self.lr = float(config.get("learning_rate", 0.01))
        self.epochs = int(config.get("epochs", 10))
        self.batch_size = int(config.get("batch_size", 256))
        self.n_layers = int(config.get("n_layers", 2))
        device_cfg = config.get("device", "auto")
        self.device = torch.device("cuda" if (device_cfg == "auto" and torch.cuda.is_available()) or device_cfg == "cuda" else "cpu")
        self.reg_weight = float(config.get("reg_weight", 1e-3))
        self.in_method = config.get("in_method", None)
        self.neg_alpha = float(config.get("neg_alpha", 0.8))
        # will be set in fit
        self.user2idx = {}
        self.item2idx = {}
        self.idx2user = {}
        self.idx2item = {}
        self.n_users = 0
        self.n_items = 0
        self.user_emb = None
        self.item_emb = None
        self.norm_adj = None
        self.neg_sampler = None
        self.fair_reg = None

    def _build_maps(self, df):
        users = sorted(df["user_id"].unique().tolist())
        items = sorted(df["item_id"].unique().tolist())
        self.user2idx = {u: idx for idx, u in enumerate(users)}
        self.item2idx = {i: idx for idx, i in enumerate(items)}
        self.idx2user = {idx: u for u, idx in self.user2idx.items()}
        self.idx2item = {idx: i for i, idx in self.item2idx.items()}
        self.n_users = len(users)
        self.n_items = len(items)

    def _build_norm_adj(self, df):
        rows = []
        cols = []
        data = []
        for _, r in df.iterrows():
            u = self.user2idx[int(r["user_id"])]
            i = self.item2idx[int(r["item_id"])]
            rows.append(u); cols.append(i); data.append(1.0)
        A = sp.coo_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items))
        top = sp.hstack([sp.csr_matrix((self.n_users, self.n_users)), A])
        bottom = sp.hstack([A.T, sp.csr_matrix((self.n_items, self.n_items))])
        adj = sp.vstack([top, bottom]).tocsr()
        rowsum = np.array(adj.sum(1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj = D_inv_sqrt.dot(adj).dot(D_inv_sqrt)
        return sparse_mat_to_torch_sparse_tensor(norm_adj).coalesce().to(self.device)

    def fit(self, train_df, val_df=None, group_info=None):
        self._build_maps(train_df)
        self.user_emb = nn.Embedding(self.n_users, self.latent_dim).to(self.device)
        self.item_emb = nn.Embedding(self.n_items, self.latent_dim).to(self.device)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        self.norm_adj = self._build_norm_adj(train_df)
        optimizer = torch.optim.Adam([self.user_emb.weight, self.item_emb.weight], lr=self.lr)
        if self.in_method == "negative_sampling":
            item_counts = train_df["item_id"].value_counts().to_dict()
            self.neg_sampler = FairnessAwareNegativeSampler(item_counts, alpha=self.neg_alpha)
        if self.in_method == "regularization":
            self.fair_reg = FairnessAwareRegularizer(lambda_fair=self.reg_weight)
        users_local = train_df["user_id"].map(self.user2idx).values
        items_local = train_df["item_id"].map(self.item2idx).values
        n = len(users_local)
        for epoch in range(self.epochs):
            perm = np.random.permutation(n)
            total_loss = 0.0
            # compute propagated embeddings once per epoch
            all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
            embs = [all_emb]
            for _ in range(self.n_layers):
                all_emb = torch.sparse.mm(self.norm_adj, all_emb)
                embs.append(all_emb)
            final_emb = torch.stack(embs, dim=1).mean(dim=1)
            user_embs = final_emb[:self.n_users]
            item_embs = final_emb[self.n_users:]
            # update embedding parameters with propagated values
            self.user_emb.weight.data = user_embs
            self.item_emb.weight.data = item_embs
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                u_idx = torch.LongTensor(users_local[idx]).to(self.device)
                i_idx = torch.LongTensor(items_local[idx]).to(self.device)
                pos_preds = (self.user_emb(u_idx) * self.item_emb(i_idx)).sum(dim=1)
                if self.in_method == "negative_sampling":
                    neg_global = self.neg_sampler.sample(len(idx))
                    neg_local = [ self.item2idx.get(it, random.randint(0, self.n_items-1)) for it in neg_global ]
                    neg_i = torch.LongTensor(neg_local).to(self.device)
                    neg_preds = (self.user_emb(u_idx) * self.item_emb(neg_i)).sum(dim=1)
                    loss = -torch.mean(torch.log(torch.sigmoid(pos_preds - neg_preds)))
                else:
                    labels = torch.ones_like(pos_preds).to(self.device)
                    loss = F.mse_loss(pos_preds, labels)
                if self.in_method == "regularization" and group_info is not None:
                    batch_global_users = [ self.idx2user[int(x)] for x in u_idx.cpu().numpy() ]
                    group_labels = [ group_info.get("user_group_map", {}).get(u, "none") for u in batch_global_users ]
                    loss = loss + self.fair_reg.compute_group_variance(group_labels, pos_preds)
                l2 = (self.user_emb(u_idx).norm(2).pow(2).mean() + self.item_emb(i_idx).norm(2).pow(2).mean())
                loss = loss + self.reg_weight * l2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[LightGCN] Epoch {epoch+1}/{self.epochs} loss={total_loss:.4f}")

    def recommend_topk(self, user_id: int, k: int = 20):
        if self.user_emb is None:
            raise RuntimeError("Model not trained.")
        if user_id not in self.user2idx:
            return []
        # compute final embeddings
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            embs.append(all_emb)
        final_emb = torch.stack(embs, dim=1).mean(dim=1)
        user_embs = final_emb[:self.n_users]
        item_embs = final_emb[self.n_users:]
        u_local = self.user2idx[user_id]
        u_vec = user_embs[u_local].unsqueeze(0)
        scores = torch.matmul(u_vec, item_embs.T).squeeze(0).cpu().numpy()
        topk = np.argsort(-scores)[:k]
        results = [(self.idx2item[idx], float(scores[idx])) for idx in topk]
        return results