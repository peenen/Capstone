# test.py
# Minimal MF (BPR) with fast training and fairness metrics — NO masked negative sampling.
# - Uniform negatives per example (only avoid neg==pos for stability)
# - No data-prep; per-user LOO split
# - AMP on CUDA; big batches; concise diagnostics
#
# Usage (MovieLens-like):
#   python pure_mf_debug_fair_fast_nomask.py --data_path data/movielens.csv \
#       --epochs 120 --latent_dim 128 --batch_size 4096 --lr 0.002 --topk 20

import argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from src.data_preparation.dataset_balancing import *
import math, random
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple

# ============ Quality metrics ============
def _topk(preds_df: pd.DataFrame, k: int):
    if preds_df.empty or k <= 0: return pd.DataFrame(columns=["user_id","item_id"])
    return (preds_df.sort_values(["user_id","prediction"], ascending=[True, False])
                   .groupby("user_id", as_index=False)
                   .head(k)[["user_id","item_id"]])

def _item2genre_map(gt: pd.DataFrame) -> pd.Series:
    return (
        gt[["item_id","genre"]]
          .dropna()
          .drop_duplicates(subset=["item_id"])
          .set_index("item_id")["genre"]
    )

def _safe_norm_counts(s: pd.Series) -> pd.Series:
    s = s.astype(float); tot = s.sum()
    return s*0.0 if tot <= 0 else s/tot

def precision_at_k(preds_df, gt_df, k):
    if preds_df.empty or gt_df.empty or k <= 0: return 0.0
    topk = _topk(preds_df, k); gt = gt_df[["user_id","item_id"]].drop_duplicates()
    users = gt["user_id"].unique().tolist()
    hits = topk.merge(gt, on=["user_id","item_id"])
    hpu = hits.groupby("user_id").size()
    vals = [(hpu.get(u, 0) / float(k)) for u in users if (gt[gt.user_id==u].shape[0] > 0)]
    return float(np.mean(vals)) if vals else 0.0
    
    # if preds_df.empty or gt_df.empty or k <= 0: return 0.0
    # topk = _topk(preds_df, k); gt = gt_df[["user_id","item_id"]].drop_duplicates()
    # users = gt["user_id"].unique().tolist()
    # hits = topk.merge(gt, on=["user_id","item_id"])
    # hits_per_user = hits.groupby("user_id").size()
    # gt_per_user   = gt.groupby("user_id").size()
    # vals = []
    # for u in users:
    #     denom = min(k, int(gt_per_user.get(u, 0)))
    #     if denom > 0:
    #         vals.append(hits_per_user.get(u, 0) / float(denom))
    # return float(np.mean(vals)) if vals else 0.0

def recall_at_k(preds_df, gt_df, k):
    if preds_df.empty or gt_df.empty or k <= 0: return 0.0
    topk = _topk(preds_df, k); gt = gt_df[["user_id","item_id"]].drop_duplicates()
    users = gt["user_id"].unique().tolist()
    hits = topk.merge(gt, on=["user_id","item_id"])
    hpu = hits.groupby("user_id").size()
    gtu = gt.groupby("user_id").size()
    vals = []
    for u in users:
        denom = int(gtu.get(u, 0))
        if denom > 0: vals.append(hpu.get(u, 0) / float(denom))
    return float(np.mean(vals)) if vals else 0.0
    
    # if preds_df.empty or gt_df.empty or k <= 0: return 0.0
    # topk = _topk(preds_df, k); gt = gt_df[["user_id","item_id"]].drop_duplicates()
    # users = gt["user_id"].unique().tolist()
    # topk_groups = {u: set(g.item_id.tolist()) for u,g in topk.groupby("user_id")}
    # gt_groups   = {u: set(g.item_id.tolist()) for u,g in gt.groupby("user_id")}
    # vals = []
    # for u in users:
    #     if not gt_groups.get(u): continue
    #     hits = topk_groups.get(u, set()) & gt_groups[u]
    #     vals.append(1.0 if len(hits) > 0 else 0.0)
    # return float(np.mean(vals)) if vals else 0.0

def ndcg_at_k(preds_df, gt_df, k):
    if preds_df.empty or gt_df.empty or k <= 0: return 0.0
    topk = _topk(preds_df, k); gt = gt_df[["user_id","item_id"]].drop_duplicates()
    discounts = np.log2(np.arange(2, k + 2)); ideal_gains = 1.0 / discounts
    gt_groups = {u: set(g.item_id.tolist()) for u,g in gt.groupby("user_id")}
    ndcgs = []
    for u, g in topk.groupby("user_id"):
        rel = gt_groups.get(u, set())
        if not rel: continue
        items = g["item_id"].tolist()[:k]
        gains = np.array([1.0 if it in rel else 0.0 for it in items], dtype=np.float64)
        dcg = float((gains / discounts[:len(gains)]).sum())
        idcg = float(ideal_gains[:min(len(rel), k)].sum())
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else 0.0

# ============ Fairness metrics ============
def gini_index(preds_df, k):
    if preds_df.empty: return 0.0
    topk = _topk(preds_df, k)
    counts = topk['item_id'].value_counts().values.astype(float)
    if counts.size == 0: return 0.0
    counts = np.sort(counts); n = len(counts); cum = np.cumsum(counts)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)

def kl_divergence_items(preds_df, gt_df, k):
    if preds_df.empty or gt_df.empty: return 0.0
    topk = _topk(preds_df, k)
    p = topk['item_id'].value_counts(normalize=True)
    q = gt_df['item_id'].value_counts(normalize=True)
    all_items = set(p.index) | set(q.index)
    p_arr = np.array([p.get(i, 1e-12) for i in all_items], dtype=np.float64)
    q_arr = np.array([q.get(i, 1e-12) for i in all_items], dtype=np.float64)
    return float(np.sum(p_arr * (np.log(p_arr) - np.log(q_arr))))

def recall_dispersion(preds_df, gt_df, k):
    if preds_df.empty or gt_df.empty: return 0.0
    topk = _topk(preds_df, k); gt = gt_df[["user_id","item_id"]].drop_duplicates()
    users = gt["user_id"].unique().tolist()
    hits = topk.merge(gt, on=["user_id","item_id"])
    hpu = hits.groupby("user_id").size()
    gtu = gt.groupby("user_id").size()
    recalls = []
    for u in users:
        denom = int(gtu.get(u, 0))
        if denom > 0: recalls.append(hpu.get(u, 0) / float(denom))
    return float(np.std(recalls)) if recalls else 0.0

# ---------- genre-based disparity (single scalars; KEEP ONLY KL + RD) ----------
def kl_genre_divergence(preds_df: pd.DataFrame, ground_truth_df: pd.DataFrame, k: int) -> float:
    """
    KL divergence between the GENRE distributions of exposure (Top-K) and ground truth:
    KL( P_genre(topK) || Q_genre(GT) ). Lower = closer to GT genre demand.
    """
    if preds_df.empty or ground_truth_df.empty:
        return 0.0
    topk = _topk(preds_df, k)
    item2genre = _item2genre_map(ground_truth_df)
    # genre exposure from Top-K
    topk_g = (
        topk.merge(item2genre.rename("genre"), left_on="item_id", right_index=True, how="left")
            .dropna(subset=["genre"])
    )
    p_genre = _safe_norm_counts(topk_g["genre"].value_counts())
    # genre distribution in ground truth
    q_genre = _safe_norm_counts(ground_truth_df.dropna(subset=["genre"])["genre"].value_counts())
    # align supports
    genres = set(p_genre.index) | set(q_genre.index)
    p_arr = np.array([p_genre.get(g, 1e-12) for g in genres], dtype=np.float64)
    q_arr = np.array([q_genre.get(g, 1e-12) for g in genres], dtype=np.float64)
    return float(np.sum(p_arr * (np.log(p_arr) - np.log(q_arr))))

def recall_dispersion_genre(preds_df: pd.DataFrame, ground_truth_df: pd.DataFrame, k: int) -> float:
    """
    Std dev across genres of per-genre average Recall@K.
    For each genre g: compute per-user recall restricted to GT items in g; average over users; then std over genres.
    Lower = more even recall across genres.
    """
    if preds_df.empty or ground_truth_df.empty:
        return 0.0
    topk = _topk(preds_df, k)
    item2genre = _item2genre_map(ground_truth_df)
    if item2genre.empty:
        return 0.0

    # annotate predictions and ground truth with genre
    topk_g = (
        topk.merge(item2genre.rename("genre"), left_on="item_id", right_index=True, how="left")
            .dropna(subset=["genre"])
    )[["user_id","item_id","genre"]]
    gt = ground_truth_df[["user_id","item_id","genre"]].dropna().drop_duplicates()

    recalls_by_genre = []
    for g in sorted(gt["genre"].unique().tolist()):
        gt_g = gt[gt["genre"] == g][["user_id","item_id"]]
        if gt_g.empty:
            continue
        users = gt_g["user_id"].unique().tolist()
        topk_g_sub = topk_g[topk_g["genre"] == g][["user_id","item_id"]]
        hits = topk_g_sub.merge(gt_g, on=["user_id","item_id"])
        hpu = hits.groupby("user_id").size()
        gtu = gt_g.groupby("user_id").size()
        per_user_recalls = []
        for u in users:
            denom = int(gtu.get(u, 0))
            if denom > 0:
                per_user_recalls.append(hpu.get(u, 0) / float(denom))
        if per_user_recalls:
            recalls_by_genre.append(float(np.mean(per_user_recalls)))
    return float(np.std(recalls_by_genre)) if recalls_by_genre else 0.0

# ============ Pre-Processing ==================

def _inverse_freq_weights(series: pd.Series) -> pd.Series:
    counts = series.value_counts()
    inv = series.map(lambda x: 1.0 / (counts[x] + 1e-8))
    return inv / inv.mean()

def preprocess_by_genre(df: pd.DataFrame, enable_weights: bool = True) -> pd.DataFrame:
    """
    Auto-detect advantaged/disadvantaged genres by unique-user coverage:
      advantaged = genre with the MOST unique users
      disadvantaged = genre with the LEAST unique users
      others => 'other'
    Adds:
      genre_group in {'adv','disadv','other'}
      sample_weight = inverse-frequency over genre_group (if enable_weights)
    """
    df = df.copy()
    if "genre" not in df.columns:
        if enable_weights: df["sample_weight"] = 1.0
        return df

    # unique-user coverage per genre
    user_cov = df.groupby("genre")["user_id"].nunique().sort_values(ascending=False)
    if user_cov.empty:
        if enable_weights: df["sample_weight"] = 1.0
        return df

    adv_genre = user_cov.index[0]
    disadv_genre = user_cov.index[-1] if len(user_cov) > 1 else user_cov.index[0]

    def _tag(g):
        if g == adv_genre: return "adv"
        if g == disadv_genre: return "disadv"
        return "other"

    df["genre_group"] = df["genre"].apply(_tag)

    if enable_weights:
        counts = df["genre_group"].value_counts()
        inv = df["genre_group"].map(lambda x: 1.0 / (counts[x] + 1e-8))
        df["sample_weight"] = (inv / inv.mean()).astype(float)

    # (Optional) store detected labels for logging
    df.attrs["adv_genre"] = str(adv_genre)
    df.attrs["disadv_genre"] = str(disadv_genre)
    return df

def preprocess_by_user_activity(df: pd.DataFrame,
                                percentile: float = 50.0,
                                enable_weights: bool = True) -> pd.DataFrame:
    """
    Group users by activity (high/low via percentile threshold) and assign weights.
    """
    df = df.copy()
    if "user_id" not in df.columns:
        if enable_weights: df["sample_weight"] = 1.0
        return df
    user_counts = df.groupby("user_id").size()
    thr = user_counts.quantile(percentile / 100.0)
    df["user_group"] = df["user_id"].apply(lambda u: "high" if user_counts[u] >= thr else "low")
    if enable_weights:
        df["user_group_weight"] = _inverse_freq_weights(df["user_group"]).astype(float)
        df["sample_weight"] = df["user_group_weight"].astype(float)
    return df

# ============ Split (per-user LOO) ============
# def leave_one_out(df: pd.DataFrame, per_user_val=1, per_user_test=1, seed=42):
#     rng = np.random.RandomState(seed)
#     train, val, test = [], [], []
#     for _, g in df.groupby("user_id"):
#         g = g.sample(frac=1.0, random_state=rng)
#         n = len(g); need = per_user_val + per_user_test + 1
#         if n < need: train.append(g); continue
#         test.append(g.iloc[-per_user_test:])
#         val.append(g.iloc[-(per_user_test+per_user_val):-per_user_test] if per_user_val>0 else g.iloc[0:0])
#         train.append(g.iloc[:-(per_user_test+per_user_val)])
#     train_df = pd.concat(train, ignore_index=True) if train else pd.DataFrame(columns=df.columns)
#     val_df   = pd.concat(val,   ignore_index=True) if val   else pd.DataFrame(columns=df.columns)
#     test_df  = pd.concat(test,  ignore_index=True) if test  else pd.DataFrame(columns=df.columns)
#     # Ensure eval items seen in train (optional but common)
#     seen_items = set(train_df.item_id.unique())
#     test_df = test_df[test_df.item_id.isin(seen_items)].reset_index(drop=True)
#     val_df  = val_df[val_df.item_id.isin(seen_items)].reset_index(drop=True)
#     return train_df.reset_index(drop=True), val_df, test_df

# ---------------- Traditional random split ----------------
def random_split(df: pd.DataFrame, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    train_df, temp_df = train_test_split(df, test_size=(1 - train_ratio), random_state=seed, shuffle=True)
    val_df, test_df   = train_test_split(temp_df, test_size=(test_ratio / (val_ratio + test_ratio)),
                                         random_state=seed, shuffle=True)
    # Optional but typical: filter eval to seen users/items to avoid cold-start evaluation
    seen_users = set(train_df.user_id.unique()); seen_items = set(train_df.item_id.unique())
    val_df  = val_df[val_df.user_id.isin(seen_users) & val_df.item_id.isin(seen_items)].reset_index(drop=True)
    test_df = test_df[test_df.user_id.isin(seen_users) & test_df.item_id.isin(seen_items)].reset_index(drop=True)
    return train_df.reset_index(drop=True), val_df, test_df

# ============ FairNeg ==========

class FairNegController:
    """
    Implements FairNeg per WWW'23:
    - G-BCE per group (Eq. 3) to perceive fairness
    - Gradient for group probs (Eq. 4): grad[z] = L_plus[z] - mean(L_plus)
    - Momentum update (Eq. 5): v = gamma*v + alpha*grad; p = p - v; renorm
    - Mixup (Eq. 8): p = beta * p_fair + (1-beta) * p_imp where
      p_fair(j) = p[z]/|{i in pool: group(i)=z}| (Eq. 6),
      p_imp ∝ exp(score/τ) (Eq. 7)
    """
    def __init__(self, train_df, item_group_col, n_items, device,
                 beta=0.5, tau=0.2, alpha=0.05, gamma=0.9, eps=1e-6, candidate_pool=64,
                 user2idx=None, item2idx=None):
        from collections import defaultdict
        self.device = device
        self.n_items = int(n_items)
        self.beta = float(beta); self.tau = float(tau)
        self.alpha = float(alpha); self.gamma = float(gamma)
        self.eps = float(eps); self.candidate_pool = int(candidate_pool)
        self.user2idx = user2idx or {}
        self.item2idx = item2idx or {}
    
        # Ensure a grouping column exists
        if item_group_col not in train_df.columns:
            train_df = train_df.assign(_grp="all")
            item_group_col = "_grp"
    
        # Group dictionaries
        groups = sorted(train_df[item_group_col].astype(str).unique())
        self.group_to_idx = {g: i for i, g in enumerate(groups)}
        self.idx_to_group = {i: g for g, i in self.group_to_idx.items()}
        self.n_groups = len(self.group_to_idx)
    
        # Local item -> group index
        self.item_group_idx = np.zeros(self.n_items, dtype=np.int64)
        for it_raw, g in train_df[['item_id', item_group_col]].drop_duplicates().itertuples(index=False, name=None):
            li = self.item2idx.get(int(it_raw), None)
            if li is not None:
                self.item_group_idx[li] = self.group_to_idx[str(g)]
    
        # Initialize group probabilities p (count local items per group)
        cnt = np.zeros(self.n_groups, dtype=np.float64)
        for it_raw, g in train_df[['item_id', item_group_col]].drop_duplicates().itertuples(index=False, name=None):
            li = self.item2idx.get(int(it_raw), None)
            if li is not None:
                cnt[self.group_to_idx[str(g)]] += 1.0
        cnt = np.clip(cnt, self.eps, None)
        self.p = cnt / cnt.sum()
        self.v = np.zeros_like(self.p)
    
        # Local user -> set(local positive items)
        self.user_pos = defaultdict(set)
        for u_raw, it_raw in train_df[['user_id', 'item_id']].itertuples(index=False, name=None):
            lu = self.user2idx.get(int(u_raw), None)
            li = self.item2idx.get(int(it_raw), None)
            if lu is not None and li is not None:
                self.user_pos[lu].add(li)

    def _gbce_by_group(self, model, train_df, batch_size=4096):
        # Map raw ids -> local indices and filter invalids
        u_local = train_df['user_id'].map(lambda x: self.user2idx.get(int(x), -1)).to_numpy(np.int64)
        i_local = train_df['item_id'].map(lambda x: self.item2idx.get(int(x), -1)).to_numpy(np.int64)
        keep = (u_local >= 0) & (i_local >= 0) & (i_local < self.n_items)
        u_local, i_local = u_local[keep], i_local[keep]
    
        L = np.zeros(self.n_groups, dtype=np.float64)
        for start in range(0, len(u_local), batch_size):
            sl = slice(start, start + batch_size)
            u = torch.as_tensor(u_local[sl], device=self.device, dtype=torch.long)
            v = torch.as_tensor(i_local[sl], device=self.device, dtype=torch.long)
            if u.numel() == 0:
                continue
            with torch.no_grad():
                s  = model.score(u, v)                              # local indices
                lp = -torch.log(torch.sigmoid(s).clamp_min(1e-12))  # -log σ(score)
            lp_np = lp.detach().cpu().numpy()
            g = self.item_group_idx[i_local[sl]]                    # local item groups
            for z in range(self.n_groups):
                m = (g == z)
                if m.any():
                    L[z] += float(lp_np[m].mean())
        return L

    def update_group_probs(self, model, train_df):
        # Compute L_plus per group and gradient per Eq. (4)
        L = self._gbce_by_group(model, train_df)
        L_mean = L.mean() if self.n_groups > 0 else 0.0
        grad = L - L_mean
        # Momentum update per Eq. (5)
        self.v = self.gamma * self.v + self.alpha * grad
        self.p = self.p - self.v
        # Project onto simplex: clip & renorm
        self.p = np.clip(self.p, self.eps, None)
        self.p = self.p / self.p.sum()

    # def _sample_pool(self, user_id, all_items):
    #     # Uniform candidate pool from unobserved
    #     pos = self.user_pos.get(int(user_id), set())
    #     pool = []
    #     tries = 0
    #     while len(pool) < self.candidate_pool and tries < self.candidate_pool*10:
    #         j = random.randrange(self.n_items)
    #         if j not in pos:
    #             pool.append(j)
    #         tries += 1
    #     if not pool:  # fallback
    #         pool = [random.randrange(self.n_items)]
    #     return np.array(pool, dtype=np.int64)

    # def sample_negatives(self, model, users, pos_items):
    #     """
    #     For each (u, pos) return one negative j sampled from
    #     mixup distribution (Eq. 8) computed on a candidate pool.
    #     """
    #     users = users.detach().cpu().numpy().astype(np.int64)
    #     negs = np.empty_like(users)
    #     for idx, u in enumerate(users):
    #         pool = self._sample_pool(u, self.n_items)
    #         # fairness-aware per Eq. (6)
    #         g_idx = self.item_group_idx[pool]
    #         # mass per group: p[z] equally divided in pool items of that group
    #         counts = np.maximum(np.bincount(g_idx, minlength=self.n_groups), 1)
    #         p_fair = self.p[g_idx] / counts[g_idx]

    #         # importance-aware per Eq. (7)
    #         with torch.no_grad():
    #             u_t = torch.tensor([u], device=self.device, dtype=torch.long).repeat(len(pool))
    #             j_t = torch.tensor(pool, device=self.device, dtype=torch.long)
    #             scores = model.score(u_t, j_t).detach()  # raw
    #             s = (scores / self.tau).clamp(-50, 50)   # numeric stability
    #             p_imp = torch.softmax(s, dim=0).cpu().numpy()

    #         # mixup (Eq. 8)
    #         p_mix = self.beta * p_fair + (1.0 - self.beta) * p_imp
    #         p_mix = np.maximum(p_mix, self.eps)
    #         p_mix = p_mix / p_mix.sum()
    #         negs[idx] = int(np.random.choice(pool, p=p_mix))
    #     return torch.as_tensor(negs, device=self.device, dtype=torch.long)

    def _sample_pool(self, user_local: int):
        # Uniform candidate pool from unobserved local items
        pos = self.user_pos.get(int(user_local), set())
        pool = []
        tries = 0
        while len(pool) < self.candidate_pool and tries < self.candidate_pool * 10:
            j = random.randrange(self.n_items)   # local item index
            if j not in pos:
                pool.append(j)
            tries += 1
        if not pool:  # fallback
            pool = [random.randrange(self.n_items)]
        return np.array(pool, dtype=np.int64)
    
    def sample_negatives(self, model, users: torch.Tensor, pos_items: torch.Tensor):
        """
        For each (u, pos) return one negative j (local item index) sampled from
        mixup distribution (Eq. 8) computed on a candidate pool.
        """
        users = users.detach().cpu().numpy().astype(np.int64)
        negs = np.empty_like(users)
        for idx, u in enumerate(users):
            pool = self._sample_pool(u)
            g_idx = self.item_group_idx[pool]
    
            # fairness-aware (Eq. 6)
            counts = np.maximum(np.bincount(g_idx, minlength=self.n_groups), 1)
            p_fair = self.p[g_idx] / counts[g_idx]
    
            # importance-aware (Eq. 7)
            with torch.no_grad():
                u_t = torch.tensor([u], device=self.device, dtype=torch.long).repeat(len(pool))
                j_t = torch.tensor(pool, device=self.device, dtype=torch.long)
                scores = model.score(u_t, j_t)  # local indices
                s = (scores / self.tau).clamp(-50, 50)
                p_imp = torch.softmax(s, dim=0).cpu().numpy()
    
            # mixup (Eq. 8)
            p_mix = self.beta * p_fair + (1.0 - self.beta) * p_imp
            p_mix = np.maximum(p_mix, self.eps)
            p_mix = p_mix / p_mix.sum()
            negs[idx] = int(np.random.choice(pool, p=p_mix))
        return torch.as_tensor(negs, device=self.device, dtype=torch.long)

    # --- add this new method in FairNegController ---
    def sample_negatives_batch(self, model, users_local: torch.Tensor):
        """
        users_local: (B,) long cuda (LOCAL user ids)
        return: (B,) long cuda negatives (LOCAL item ids)
        """
        B = users_local.size(0)
        P = self.candidate_pool
        device = self.device
    
        # (B, P) candidate pool, uniform local ids
        pool = torch.randint(0, self.n_items, (B, P), device=device, dtype=torch.long)
    
        # mask user positives (optional cheap pass; may leave few positives)
        # build a boolean mask where pool equals any positive of user u (vectorized-ish fallback)
        # For speed, we skip strict masking; rely on soft probabilities. Uncomment if needed:
        # for b in range(B):
        #     if int(users_local[b].item()) in self.user_pos:
        #         pos = torch.tensor(list(self.user_pos[int(users_local[b].item())]),
        #                           device=device, dtype=torch.long)
        #         if pos.numel() > 0:
        #             bad = (pool[b:b+1, :] == pos.view(-1,1)).any(0)
        #             while bad.any():
        #                 pool[b, bad] = torch.randint(0, self.n_items, (bad.sum(),), device=device)
    
        # fairness-aware probs (Eq.6)
        g_idx = torch.as_tensor(self.item_group_idx, device=device, dtype=torch.long)  # (I,)
        g_pool = g_idx[pool]                                    # (B, P)
        # counts per group within each row
        max_g = int(self.n_groups)
        counts = torch.stack([torch.bincount(g_pool[b], minlength=max_g) for b in range(B)], dim=0)  # (B,G)
        counts = counts.clamp_min_(1)
        p_g = torch.as_tensor(self.p, device=device, dtype=torch.float)            # (G,)
        p_fair = (p_g[g_pool] / counts.gather(1, g_pool))                          # (B, P)
    
        # importance-aware (Eq.7)
        with torch.no_grad():
            # user (B,D), items (B,P,D) via gather
            u_emb = model.cached_user[users_local]                                  # (B,D)
            i_emb = model.cached_item[pool]                                         # (B,P,D)
            scores = (i_emb * u_emb.unsqueeze(1)).sum(dim=2)                        # (B,P)
            s = (scores / self.tau).clamp(-50, 50).float()
            p_imp = torch.softmax(s, dim=1)                                         # (B,P)
    
        # mixup (Eq.8)
        beta = float(self.beta)
        p_mix = beta * p_fair + (1.0 - beta) * p_imp
        p_mix = p_mix.clamp_min_(1e-12)
        p_mix = p_mix / p_mix.sum(dim=1, keepdim=True)
        # sample one per row
        idx = torch.multinomial(p_mix, num_samples=1).squeeze(1)                    # (B,)
        negs = pool.gather(1, idx.view(-1,1)).squeeze(1)                            # (B,)
        return negs

    # === Add to FairNegController (MF version) ===
    def sample_negatives_batch_mf(self, model, users_local: torch.Tensor):
        """
        users_local: (B,) long cuda (LOCAL user ids)
        return: (B,) long cuda negatives (LOCAL item ids)
        """
        B, P, dev = users_local.size(0), self.candidate_pool, self.device
    
        # (B, P) candidate pool
        pool = torch.randint(0, self.n_items, (B, P), device=dev, dtype=torch.long)
    
        # group-aware probs (Eq.6)
        g_idx_all = torch.as_tensor(self.item_group_idx, device=dev, dtype=torch.long)  # (I,)
        g_pool = g_idx_all[pool]                                 # (B,P)
        G = int(self.n_groups)
        counts = torch.stack([torch.bincount(g_pool[b], minlength=G) for b in range(B)], 0).clamp_min_(1)  # (B,G)
        p_g = torch.as_tensor(self.p, device=dev, dtype=torch.float)                 # (G,)
        p_fair = p_g[g_pool] / counts.gather(1, g_pool)                              # (B,P)
    
        # importance-aware (Eq.7) using MF embeddings
        # get embedding tables (support either .user_emb/.item_emb or .user/.item)
        U = getattr(model, "user_emb", getattr(model, "user", None)).weight          # (U,D)
        V = getattr(model, "item_emb", getattr(model, "item", None)).weight          # (I,D)
        u_emb = U[users_local]                                                       # (B,D)
        i_emb = V[pool]                                                              # (B,P,D)
        with torch.no_grad():
            scores = (i_emb * u_emb.unsqueeze(1)).sum(dim=2)                         # (B,P)
            s = (scores / self.tau).clamp(-50, 50).float()
            p_imp = torch.softmax(s, dim=1)                                          # (B,P)
    
        # mixup (Eq.8)
        p_mix = float(self.beta) * p_fair + (1.0 - float(self.beta)) * p_imp
        p_mix = p_mix.clamp_min_(1e-12)
        p_mix = p_mix / p_mix.sum(dim=1, keepdim=True)
    
        # sample one per user
        idx = torch.multinomial(p_mix, num_samples=1).squeeze(1)                     # (B,)
        negs = pool.gather(1, idx.view(-1, 1)).squeeze(1)                            # (B,)
        return negs
    

# ============ Model ============
class MFNet(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user.weight, std=0.01)
        nn.init.normal_(self.item.weight, std=0.01)
    def forward(self, u, i): return (self.user(u) * self.item(i)).sum(dim=1)
    def score(self, u_idx: torch.Tensor, i_idx: torch.Tensor) -> torch.Tensor:
        # raw dot-product scores (no sigmoid)
        return (self.user(u_idx) * self.item(i_idx)).sum(dim=1)

class MFTrainer:
    def __init__(self, dim=64, lr=2e-3, epochs=100, batch_size=1024, device="auto",
                in_method="unineg", fairneg_beta=0.5, fairneg_tau=0.2, fairneg_lr=0.05,
                 fairneg_momentum=0.9, fairneg_pool=64,
                es_patience=8,       # stop if no improvement for N epochs
                 es_min_delta=1e-4,     # required improvement margin
                 es_warmup=3,          # don't early-stop in the first N epochs
                 es_cooldown=1):       # after an improvement, wait N epochs before counting patience):
        dev = device
        self.device = torch.device("cuda" if (dev=="auto" and torch.cuda.is_available()) or dev=="cuda" else "cpu")
        self.dim, self.lr, self.epochs, self.bs = dim, lr, epochs, batch_size
        self.user2idx={}; self.item2idx={}; self.idx2user={}; self.idx2item={}
        self.net=None
        self.use_amp = (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.in_method = in_method
        self.fairneg_beta = float(fairneg_beta)
        self.fairneg_tau = float(fairneg_tau)
        self.fairneg_lr = float(fairneg_lr)
        self.fairneg_momentum = float(fairneg_momentum)
        self.fairneg_pool = int(fairneg_pool)
        torch.backends.cudnn.benchmark = True

        # ---- Early stopping state (new) ----
        self.es_patience   = int(es_patience)
        self.es_min_delta  = float(es_min_delta)
        self.es_warmup     = int(es_warmup)
        self.es_cooldown   = int(es_cooldown)

    def _build_maps(self, df):
        users = sorted(df.user_id.unique().tolist())
        items = sorted(df.item_id.unique().tolist())
        self.user2idx = {u:i for i,u in enumerate(users)}
        self.item2idx = {it:i for i,it in enumerate(items)}
        self.idx2user = {i:u for u,i in self.user2idx.items()}
        self.idx2item = {i:it for it,i in self.item2idx.items()}
        self.n_users, self.n_items = len(users), len(items)

    @torch.no_grad()
    def _sample_neg_uniform(self, batch_size, pos_items):
        # Uniform negatives per example; only avoid neg == current positive
        neg = torch.randint(low=0, high=self.n_items, size=(batch_size,), device=self.device)
        same = neg.eq(pos_items)
        if same.any():
            neg[same] = (neg[same] + 1) % self.n_items
        return neg

    def fit(self, train_df):
        self._build_maps(train_df)
        self.net = MFNet(self.n_users, self.n_items, self.dim).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=0.0)  # reg off
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(10, self.epochs))

        u_arr = train_df.user_id.map(self.user2idx).values.astype(np.int64)
        i_arr = train_df.item_id.map(self.item2idx).values.astype(np.int64)
        n = len(u_arr)

        # pre-process related (INSERT after u_arr/i_arr creation)
        if "sample_weight" in train_df.columns:
            self._w_arr = train_df["sample_weight"].astype(np.float32).values
        else:
            self._w_arr = None

        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))

        fairneg = None
        if self.in_method == "fairneg":
            fairneg = FairNegController(
                train_df=train_df,
                item_group_col=("genre" if "genre" in train_df.columns else "item_group"),
                n_items=self.n_items,
                device=self.device,
                beta=self.fairneg_beta,          # from __init__
                tau=self.fairneg_tau,
                alpha=self.fairneg_lr,
                gamma=self.fairneg_momentum,
                candidate_pool=self.fairneg_pool,
                user2idx=self.user2idx,          # <<< add
                item2idx=self.item2idx           # <<< add
            )

        # ===== Early stopping bookkeeping (new) =====
        best_loss = float("inf")
        best_state = None
        epochs_no_improve = 0
        cooldown_left = 0
        
        for ep in range(self.epochs):
            epoch_loss_sum = 0.0
            epoch_count    = 0
            perm = np.random.permutation(n)
            # total = 0.0
            for s in range(0, n, self.bs):
                idx = perm[s:s+self.bs]
                u = torch.from_numpy(u_arr[idx]).to(self.device)
                i_pos = torch.from_numpy(i_arr[idx]).to(self.device)

                # ==== NEGATIVE SAMPLING (conditional) ====
                # if self.in_method == "fairneg" and fairneg is not None:
                #     neg_i = fairneg.sample_negatives(self.net, u, i_pos)
                # else:
                #     neg_i = self._sample_neg_uniform(u.size(0), i_pos)
                
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    # pos = self.net(u, i_pos)
                    # neg_i = self._sample_neg_uniform(u.size(0), i_pos)
                    # neg = self.net(u, neg_i)
                    pos = self.net.score(u, i_pos)
                    if self.in_method == "fairneg" and fairneg is not None:
                        neg_i = fairneg.sample_negatives_batch_mf(self.net, u)
                    else:
                        neg_i = self._sample_neg_uniform(u.size(0), i_pos)
                    neg = self.net.score(u, neg_i)
                    per  = F.softplus(neg - pos)       # (B,)
                    # pre-process related (Optional per-sample weighting from train_df)
                    if hasattr(self, "_w_arr") and self._w_arr is not None:
                        w = torch.from_numpy(self._w_arr[idx]).to(self.device).float()
                        loss = (per * w).sum() / (w.sum() + 1e-8)
                    else:
                        loss = per.mean()
                opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                self.scaler.step(opt); self.scaler.update()
                # total += float(loss.detach().cpu())
                # >>> replace old accumulation with this:
                epoch_loss_sum += per.sum().item()
                epoch_count    += per.numel()
            sched.step()
            epoch_loss = epoch_loss_sum / max(1, epoch_count)
            print(f"[MF] Epoch {ep+1}/{self.epochs} loss={epoch_loss:.4f}")

            if fairneg is not None and (ep + 1) % 5 == 0:
                fairneg.update_group_probs(self.net, train_df)

            # ===== Early stopping check (new) =====
            improved = (best_loss - epoch_loss) > self.es_min_delta

            if improved:
                best_loss = epoch_loss
                # keep a lightweight CPU copy of the best weights
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                epochs_no_improve = 0
                cooldown_left = self.es_cooldown
                print(f"[MF][ES] ✓ New best loss: {best_loss:.6f}")
            else:
                if ep + 1 > self.es_warmup:
                    if cooldown_left > 0:
                        cooldown_left -= 1
                        print(f"[MF][ES] cooldown {cooldown_left} left (no patience count this epoch)")
                    else:
                        epochs_no_improve += 1
                        print(f"[MF][ES] no improve: {epochs_no_improve}/{self.es_patience}")

                    if epochs_no_improve >= self.es_patience:
                        print(f"[MF][ES] Early stopping at epoch {ep+1}. Best loss: {best_loss:.6f}")
                        break

        # ===== Restore best weights if we have them (new) =====
        if best_state is not None:
            self.net.load_state_dict(best_state)
            self.net.to(self.device)
            print("[MF][ES] Restored best model weights.")

    @torch.no_grad()
    def recommend_topk(self, user_id: int, k: int = 20):
        # NOTE: We do NOT mask seen items here to keep things aligned with "no masking" request.
        # If you prefer classic evaluation (exclude seen), uncomment the masking block below.
        if self.net is None or user_id not in self.user2idx: return []
        u_local = self.user2idx[user_id]
        u = torch.tensor([u_local], dtype=torch.long, device=self.device)
        scores = (self.net.user(u) * self.net.item.weight).sum(dim=1).squeeze(0)
        k = min(k, scores.numel())
        topk_idx = torch.topk(scores, k).indices.tolist()
        return [(self.idx2item[i], float(scores[i].cpu())) for i in topk_idx]

        # --- Classic masking of seen items (optional) ---
        # seen = ...  # set of item indices seen by this user in TRAIN
        # if seen:
        #     idx = torch.tensor(list(seen), dtype=torch.long, device=self.device)
        #     scores.index_fill_(0, idx, float("-inf"))

# ============LightGCN =============

import scipy.sparse as sp

def _sparse_coo_torch(mx):
    mx = mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((mx.row, mx.col)).astype(np.int64))
    values  = torch.from_numpy(mx.data)
    shape   = torch.Size(mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

class LightGCNNet(nn.Module):
    def __init__(self, n_users, n_items, dim, n_layers):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.dim     = dim
        self.n_layers= n_layers
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user.weight, std=0.01)
        nn.init.normal_(self.item.weight, std=0.01)
        self.cached_user = None
        self.cached_item = None

    # def propagate(self, norm_adj: torch.Tensor):
    #     """Return final (user_embs, item_embs) after layer-wise propagation + mean pooling."""
    #     all_emb = torch.cat([self.user.weight, self.item.weight], dim=0)               # (U+I, D)
    #     embs = [all_emb]
    #     for _ in range(self.n_layers):
    #         all_emb = torch.sparse.mm(norm_adj, all_emb)                               # (U+I, D)
    #         embs.append(all_emb)
    #     out = torch.stack(embs, dim=1).mean(dim=1)                                     # (U+I, D)
    #     return out[:self.n_users], out[self.n_users:]

    def _propagate_fp32(self, norm_adj: torch.Tensor):
        with torch.amp.autocast('cuda', enabled=False):
            all_emb = torch.cat([self.user.weight, self.item.weight], dim=0).float()
            embs = [all_emb]
            for _ in range(self.n_layers):
                all_emb = torch.sparse.mm(norm_adj.float(), all_emb)
                embs.append(all_emb)
            out = torch.stack(embs, dim=1).mean(dim=1)
        return out[:self.n_users], out[self.n_users:]

    def propagate(self, norm_adj: torch.Tensor):
        return self._propagate_fp32(norm_adj)
    
    def update_cache(self, norm_adj: torch.Tensor):
        u, i = self._propagate_fp32(norm_adj)
        self.cached_user = u
        self.cached_item = i

    def score(self, u_idx: torch.Tensor, i_idx: torch.Tensor) -> torch.Tensor:
        if self.cached_user is None or self.cached_item is None:
            raise RuntimeError("LightGCN embeddings cache is empty; call update_cache(norm_adj) first.")
        return (self.cached_user[u_idx] * self.cached_item[i_idx]).sum(dim=1)

class LightGCNTrainer:
    """
    Pure LightGCN with BPR loss and uniform negatives (no masked sampler).
    API mirrors MFTrainer: fit(train_df), recommend_topk(user_id, k, seen_ui=None)
    """
    def __init__(self, dim=64, lr=2e-3, epochs=100, batch_size=4096, n_layers=3, device="auto",
                 in_method="unineg", fairneg_beta=0.5, fairneg_tau=0.2, fairneg_lr=0.05,
                 fairneg_momentum=0.9, fairneg_pool=64,
                es_patience=8,       # stop if no improvement for N epochs
                 es_min_delta=1e-4,     # required improvement margin
                 es_warmup=3,          # don't early-stop in the first N epochs
                 es_cooldown=1):       # after improvement, wait N epochs before counting patience):
        dev = device
        self.device = torch.device("cuda" if (dev=="auto" and torch.cuda.is_available()) or dev=="cuda" else "cpu")
        self.dim, self.lr, self.epochs, self.bs, self.n_layers = dim, lr, epochs, batch_size, n_layers
        self.user2idx={}; self.item2idx={}; self.idx2user={}; self.idx2item={}
        self.n_users=0; self.n_items=0
        self.net=None; self.norm_adj=None
        self.use_amp = (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.in_method = in_method
        self.fairneg_beta = float(fairneg_beta)
        self.fairneg_tau = float(fairneg_tau)
        self.fairneg_lr = float(fairneg_lr)
        self.fairneg_momentum = float(fairneg_momentum)
        self.fairneg_pool = int(fairneg_pool)
        torch.backends.cudnn.benchmark = True

        # ---- Early stopping state (new) ----
        self.es_patience   = int(es_patience)
        self.es_min_delta  = float(es_min_delta)
        self.es_warmup     = int(es_warmup)
        self.es_cooldown   = int(es_cooldown)

    # --------- mappings ---------
    def _build_maps(self, df):
        users = sorted(df.user_id.unique().tolist())
        items = sorted(df.item_id.unique().tolist())
        self.user2idx = {u:i for i,u in enumerate(users)}
        self.item2idx = {it:i for i,it in enumerate(items)}
        self.idx2user = {i:u for u,i in self.user2idx.items()}
        self.idx2item = {i:it for it,i in self.item2idx.items()}
        self.n_users, self.n_items = len(users), len(items)

    # --------- graph ----------
    def _build_norm_adj(self, df):
        # build bipartite adjacency A (U x I)
        rows, cols, data = [], [], []
        for _, r in df.iterrows():
            u = self.user2idx.get(int(r.user_id)); i = self.item2idx.get(int(r.item_id))
            if u is None or i is None: continue
            rows.append(u); cols.append(i); data.append(1.0)
        A = sp.coo_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items), dtype=np.float32)
        # block matrix [[0, A], [A^T, 0]]
        top    = sp.hstack([sp.csr_matrix((self.n_users, self.n_users)), A.tocsr()], format="csr")
        bottom = sp.hstack([A.T.tocsr(), sp.csr_matrix((self.n_items, self.n_items))], format="csr")
        adj = sp.vstack([top, bottom], format="csr")
        # D^{-1/2} A D^{-1/2}
        deg = np.array(adj.sum(1)).flatten()
        d_inv_sqrt = np.power(deg, -0.5, where=deg>0)
        d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
        D_inv = sp.diags(d_inv_sqrt)
        norm_adj = D_inv @ adj @ D_inv
        return _sparse_coo_torch(norm_adj).coalesce().to(self.device).float()

    # --------- negatives ----------
    @torch.no_grad()
    def _sample_neg_uniform(self, batch_size):
        return torch.randint(low=0, high=self.n_items, size=(batch_size,), device=self.device)

    # --------- training ----------
    def fit(self, train_df: pd.DataFrame):
        self._build_maps(train_df)
        self.norm_adj = self._build_norm_adj(train_df)
        self.net = LightGCNNet(self.n_users, self.n_items, self.dim, self.n_layers).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(10, self.epochs))

        u_arr = train_df.user_id.map(self.user2idx).values.astype(np.int64)
        i_arr = train_df.item_id.map(self.item2idx).values.astype(np.int64)
        n = len(u_arr)

        # pre-process related (INSERT after u_arr/i_arr creation)
        if "sample_weight" in train_df.columns:
            self._w_arr = train_df["sample_weight"].astype(np.float32).values
        else:
            self._w_arr = None

        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))

        fairneg = None
        if self.in_method == "fairneg":
            fairneg = FairNegController(
                train_df=train_df,
                item_group_col=("genre" if "genre" in train_df.columns else "item_group"),
                n_items=self.n_items,
                device=self.device,
                beta=self.fairneg_beta,
                tau=self.fairneg_tau,
                alpha=self.fairneg_lr,
                gamma=self.fairneg_momentum,
                candidate_pool=self.fairneg_pool,
                user2idx=self.user2idx,          # <<< add
                item2idx=self.item2idx           # <<< add
            )

        # ===== Early stopping bookkeeping (new) =====
        best_loss = float("inf")
        best_state = None
        epochs_no_improve = 0
        cooldown_left = 0

        for ep in range(self.epochs):
            # self.net.update_cache(self.norm_adj)
            epoch_sum, epoch_cnt = 0.0, 0
            perm = np.random.permutation(n)
            for s in range(0, n, self.bs):
                idx = perm[s:s+self.bs]
                u = torch.from_numpy(u_arr[idx]).to(self.device)
                i_pos = torch.from_numpy(i_arr[idx]).to(self.device)

                self.net.update_cache(self.norm_adj)

                # ==== NEGATIVE SAMPLING (conditional) ====
                # if self.in_method == "fairneg" and fairneg is not None:
                #     i_neg = fairneg.sample_negatives(self.net, u, i_pos)
                # else:
                #     i_neg = self._sample_neg_uniform(u.size(0))
                
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    # propagate once per batch (you can cache per-epoch if memory allows)
                    # user_embs, item_embs = self.net.propagate(self.norm_adj)
                    # pos = (user_embs[u] * item_embs[i_pos]).sum(dim=1)
                    # i_neg = self._sample_neg_uniform(u.size(0))
                    # neg = (user_embs[u] * item_embs[i_neg]).sum(dim=1)
                    pos = self.net.score(u, i_pos)      # uses cached embeddings
                    if self.in_method == "fairneg" and fairneg is not None:
                        i_neg = fairneg.sample_negatives_batch(self.net, u)
                    else:
                        i_neg = self._sample_neg_uniform(u.size(0))
                    neg = self.net.score(u, i_neg)
                    per  = F.softplus(neg - pos)                    # BPR loss
                    # lightweight L2 on current embeddings
                    l2 = (self.net.cached_user[u].pow(2).sum(dim=1).mean() + self.net.cached_item[i_pos].pow(2).sum(dim=1).mean())

                    # pre-process related
                    if hasattr(self, "_w_arr") and self._w_arr is not None:
                        w = torch.from_numpy(self._w_arr[idx]).to(self.device).float()
                        bpr = (per * w).sum() / (w.sum() + 1e-8)
                    else:
                        bpr = per.mean()
                    
                    loss = bpr + 1e-4 * l2
                opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                self.scaler.step(opt); self.scaler.update()
                epoch_sum += per.sum().item(); epoch_cnt += per.numel()
            sched.step()
            epoch_loss = epoch_sum / max(1, epoch_cnt)
            print(f"[LightGCN] Epoch {ep+1}/{self.epochs} loss={epoch_loss:.6f}")

            if fairneg is not None and (ep + 1) % 5 == 0:
                fairneg.update_group_probs(self.net, train_df)

            # ===== Early stopping check (new) =====
            improved = (best_loss - epoch_loss) > self.es_min_delta
            if improved:
                best_loss = epoch_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                epochs_no_improve = 0
                cooldown_left = self.es_cooldown
                print(f"[LightGCN][ES] ✓ New best loss: {best_loss:.6f}")
            else:
                if ep + 1 > self.es_warmup:
                    if cooldown_left > 0:
                        cooldown_left -= 1
                        print(f"[LightGCN][ES] cooldown {cooldown_left} left (no patience count this epoch)")
                    else:
                        epochs_no_improve += 1
                        print(f"[LightGCN][ES] no improve: {epochs_no_improve}/{self.es_patience}")
                        if epochs_no_improve >= self.es_patience:
                            print(f"[LightGCN][ES] Early stopping at epoch {ep+1}. Best loss: {best_loss:.6f}")
                            break

        # ===== Restore best weights if available (new) =====
        if best_state is not None:
            self.net.load_state_dict(best_state)
            self.net.to(self.device)
            # refresh cache with best params
            self.net.update_cache(self.norm_adj)
            print("[LightGCN][ES] Restored best model weights.")

    # --------- inference ----------
    # @torch.no_grad()
    # def recommend_topk(self, user_id: int, k: int = 20, seen_ui=None):
    #     if self.net is None or user_id not in self.user2idx: return []
    #     self.net.update_cache(self.norm_adj)
    #     user_embs, item_embs = self.net.propagate(self.norm_adj)
    #     u_local = self.user2idx[user_id]
    #     u_vec = user_embs[u_local].unsqueeze(0)                                 # (1, D)
    #     scores = torch.matmul(u_vec, item_embs.T).squeeze(0)                     # (I,)
    #     if seen_ui is not None and u_local in seen_ui:
    #         idx = torch.tensor(list(seen_ui[u_local]), dtype=torch.long, device=self.device)
    #         if idx.numel() > 0:
    #             scores.index_fill_(0, idx, float("-inf"))
    #     k = min(k, scores.numel())
    #     topk = torch.topk(scores, k).indices.tolist()
    #     return [(self.idx2item[i], float(scores[i].detach().cpu())) for i in topk]
    @torch.no_grad()
    def recommend_topk(self, user_id: int, k: int = 20, seen_ui=None):
        if self.net is None or user_id not in self.user2idx:
            return []
        # ensure cache exists (no duplicate propagate)
        if self.net.cached_user is None or self.net.cached_item is None:
            self.net.update_cache(self.norm_adj)
    
        u_local = self.user2idx[user_id]
        u_vec   = self.net.cached_user[u_local].unsqueeze(0)          # (1,D)
        scores  = torch.matmul(u_vec, self.net.cached_item.T).squeeze(0)  # (I,)
    
        # mask train positives if provided (local item ids)
        if seen_ui is not None and u_local in seen_ui:
            idx = torch.tensor(list(seen_ui[u_local]), dtype=torch.long, device=scores.device)
            if idx.numel() > 0:
                scores.index_fill_(0, idx, float("-inf"))
    
        k = min(k, scores.numel())
        topk_idx = torch.topk(scores, k).indices.tolist()
        return [(self.idx2item[i], float(scores[i].detach().cpu())) for i in topk_idx]


# ========== post-process ==============

def _item2group_map(df: pd.DataFrame) -> pd.Series:
    """
    Map item_id -> group label.
    Uses 'genre' if present, otherwise falls back to 'unknown'.
    """
    if "genre" in df.columns:
        return (df[["item_id","genre"]]
                  .drop_duplicates("item_id")
                  .set_index("item_id")["genre"])
    return pd.Series(dtype=object)

def _target_props(df: pd.DataFrame) -> dict:
    """
    Global group proportions from ground truth interactions.
    """
    gmap = _item2group_map(df)
    if gmap.empty:
        return {"unknown": 1.0}
    merged = df.merge(gmap.rename("group"), on="item_id", how="left").fillna({"group":"unknown"})
    counts = merged["group"].value_counts(normalize=True)
    return counts.to_dict()

def _quota(k: int, props: dict) -> dict:
    """
    Largest remainder method to assign quotas summing to k.
    """
    raw = {g: max(0.0, p) for g,p in props.items()}
    s = sum(raw.values()) or 1.0
    raw = {g: p/s for g,p in raw.items()}
    floors = {g: int(np.floor(k*p)) for g,p in raw.items()}
    used = sum(floors.values())
    # distribute remaining
    frac = sorted(((k*raw[g] - floors[g], g) for g in raw.keys()), reverse=True)
    for _,g in frac:
        if used < k:
            floors[g]+=1; used+=1
    return floors

def _stochastic_rerank_user(topk_df_u: pd.DataFrame,
                            item2group: pd.Series,
                            target_props: dict,
                            k: int,
                            tau: float = None) -> pd.DataFrame:
    """
    Fairness-oriented, *deterministic* reranker with per-user feasible quotas.

    - Localize global target proportions to the user's candidate groups only.
    - Take top-by-score within each quota (no randomness by default).
    - For leftovers, respect a soft per-group cap derived from target proportions to
      avoid collapsing back to advantaged groups; then fill by score.

    Input/Output: columns ['user_id','item_id','prediction']
    """
    if topk_df_u.empty:
        return topk_df_u
    user_id = topk_df_u["user_id"].iloc[0]

    # Candidate annotations
    items  = topk_df_u["item_id"].to_numpy()
    scores = topk_df_u["prediction"].astype(float).to_numpy()
    groups = item2group.reindex(items).fillna("unknown").astype(str).to_numpy()

    # Localize target proportions to only groups present in this user's candidate list
    cand_groups = sorted(set(groups.tolist()))
    props_local_raw = {g: float(target_props.get(g, 0.0)) for g in cand_groups}
    s = sum(props_local_raw.values())
    # If globals have zero mass on all these groups (edge case), fallback to uniform
    if s <= 0:
        props_local = {g: 1.0 / len(cand_groups) for g in cand_groups}
    else:
        props_local = {g: v / s for g, v in props_local_raw.items()}

    # Quotas by largest remainder method (sum == k)
    quotas = _quota(k, props_local)

    # Build per-group candidate lists, sorted by score desc
    from collections import defaultdict
    by_g = defaultdict(list)
    for it, sc, g in zip(items, scores, groups):
        by_g[g].append((it, float(sc)))
    for g in by_g.keys():
        by_g[g].sort(key=lambda x: x[1], reverse=True)

    chosen = []
    taken_per_g = {g: 0 for g in cand_groups}

    # First pass: take up to quota per group (top-by-score)
    for g in cand_groups:
        need = int(quotas.get(g, 0))
        if need <= 0: 
            continue
        got = by_g[g][:min(need, len(by_g[g]))]
        chosen.extend(got)
        taken_per_g[g] += len(got)
        by_g[g] = by_g[g][len(got):]

    # Leftover slots
    remaining = k - len(chosen)
    if remaining > 0:
        # Soft cap per group: ceil(k * p_g). Guarantees no group can dominate leftovers.
        soft_cap = {g: int(np.ceil(k * props_local.get(g, 0.0))) for g in cand_groups}
        # Ensure soft cap >= the already taken (so we don't "undo" first pass)
        for g in cand_groups:
            soft_cap[g] = max(soft_cap[g], taken_per_g[g])

        # Build a unified leftover pool (still sorted by score)
        leftover = []
        for g in cand_groups:
            if by_g[g]:
                # We will respect remaining room = soft_cap[g] - taken_per_g[g] (can be 0)
                # But if everyone is full w.r.t. cap, we will relax caps below.
                for it_sc in by_g[g]:
                    leftover.append((g, it_sc[0], it_sc[1]))
        # Sort by score desc globally
        leftover.sort(key=lambda x: x[2], reverse=True)

        # Try to fill respecting caps first
        filled = 0
        for g, it, sc in leftover:
            if filled >= remaining: 
                break
            if taken_per_g[g] < soft_cap[g]:
                chosen.append((it, sc))
                taken_per_g[g] += 1
                filled += 1

        # If still short, relax caps and take best remaining by score
        if filled < remaining:
            used_items = {it for it, _ in chosen}
            rest = [(g, it, sc) for (g, it, sc) in leftover if it not in used_items]
            rest.sort(key=lambda x: x[2], reverse=True)
            to_add = min(remaining - filled, len(rest))
            chosen.extend([(it, sc) for _, it, sc in rest[:to_add]])

    # If over-selected due to any unforeseen edge, trim by score
    if len(chosen) > k:
        chosen.sort(key=lambda x: x[1], reverse=True)
        chosen = chosen[:k]

    # Stable output: sort by prediction desc
    chosen.sort(key=lambda x: x[1], reverse=True)
    return pd.DataFrame({
        "user_id":   [user_id] * len(chosen),
        "item_id":   [it for it, _ in chosen],
        "prediction":[float(sc) for _, sc in chosen]
    })

# ============ save result helper =============
# def _save_run(args, metrics: dict, out_dir: str = "runs"):
#     """
#     Save one row with metrics + selected args into runs/runs_log.csv (append),
#     and also dump a per-run JSON (runs/<run_id>.json).
#     """
#     import os, json, time
#     import pandas as pd

#     os.makedirs(out_dir, exist_ok=True)
#     run_id = time.strftime("%Y%m%d-%H%M%S")

#     # Keep exactly the arg choices you asked for (+ a few helpful IDs)
#     row = {
#         "run_id": run_id,
#         "data_path": args.data_path,
#         "model": args.model,
#         "balancing": args.balancing,
#         "pre_method": args.pre_method,
#         "in_method": args.in_method,
#         "post_method": args.post_method,
#         "topk": int(args.topk),
#         "seed": int(args.seed),
#         # metrics (cast to float for JSON/CSV cleanliness)
#         "precision": float(metrics["precision"]),
#         "recall": float(metrics["recall"]),
#         "f1": float(metrics["f1"]),
#         "ndcg": float(metrics["ndcg"]),
#         "gini": float(metrics["gini"]),
#         "kl": float(metrics["kl"]),
#         "recall_disp": float(metrics["recall_disp"]),
#         "kl_genre": float(metrics["kl_genre"]),
#         "recall_disp_genre": float(metrics["recall_disp_genre"]),
#     }

#     # Append to CSV log
#     csv_path = os.path.join(out_dir, "runs_log.csv")
#     df_row = pd.DataFrame([row])
#     if os.path.exists(csv_path):
#         df_row.to_csv(csv_path, mode="a", header=False, index=False)
#     else:
#         df_row.to_csv(csv_path, index=False)

#     # Also write a per-run JSON
#     with open(os.path.join(out_dir, f"{run_id}.json"), "w") as f:
#         json.dump(row, f, indent=2)

import csv, os, json, time, hashlib
from dataclasses import asdict, is_dataclass

def _to_serializable(v):
    if is_dataclass(v): return asdict(v)
    if isinstance(v, (set,)): return list(v)
    return v

def _args_to_dict(args):
    if isinstance(args, dict): return args
    if hasattr(args, "__dict__"): 
        d = {k: getattr(args, k) for k in vars(args)}
        return d
    # argparse.Namespace is also dict-like via vars(...)
    try:
        return vars(args)
    except Exception:
        return {}

def _save_run(args, metrics, out_dir="runs", save_mode="both"):
    os.makedirs(out_dir, exist_ok=True)

    # flatten args & metrics
    args_d = _args_to_dict(args)
    row = {
        "run_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": hashlib.md5(
            json.dumps({**args_d, **metrics}, sort_keys=True, default=_to_serializable).encode()
        ).hexdigest()[:10],
        # ---- config you care about ----
        "data_path": args_d.get("data_path"),
        "model": args_d.get("model"),
        "balancing": args_d.get("balancing"),
        "pre_method": args_d.get("pre_method"),
        "in_method": args_d.get("in_method"),
        "post_method": args_d.get("post_method"),
        # optional: core training knobs (helps debugging sweeps)
        # "epochs": args_d.get("epochs"),
        # "latent_dim": args_d.get("latent_dim"),
        # "batch_size": args_d.get("batch_size"),
        # "lr": args_d.get("lr"),
        # "topk": args_d.get("topk"),
        # "seed": args_d.get("seed"),
        # ---- metrics ----
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "ndcg": metrics.get("ndcg"),
        "gini": metrics.get("gini"),
        "kl": metrics.get("kl"),
        "recall_disp": metrics.get("recall_disp"),
        "kl_genre": metrics.get("kl_genre"),
        "recall_disp_genre": metrics.get("recall_disp_genre"),
    }

    # 1) Append to summary.csv
    summary_csv = os.path.join(out_dir, "summary.csv")
    file_exists = os.path.isfile(summary_csv)
    with open(summary_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # 2) Optionally save per-run JSON
    if save_mode in ("both", "per_run"):
        run_dir = os.path.join(out_dir, row["run_id"])
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "args.json"), "w") as fa:
            json.dump(args_d, fa, indent=2, default=_to_serializable)
        with open(os.path.join(run_dir, "metrics.json"), "w") as fm:
            json.dump(metrics, fm, indent=2, default=_to_serializable)

# ============ Main ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True, help="CSV with columns: user_id,item_id,rating")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--latent_dim", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.6)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--balancing", type=str, default=None)
    ap.add_argument("--model", type=str, choices=["mf","lightgcn"], default="mf",
                    help="Choose model: 'mf' for Matrix Factorization, 'lightgcn' for LightGCN")
    ap.add_argument("--pre_method", type=str, choices=["genre","user"], default=None,
                    help="Enable pre-processing & weighting: 'genre' or 'user'. Omit for none.")
    ap.add_argument("--user_activity_percentile", type=float, default=50.0,
                    help="Percentile split for user activity when --pre_method=user")
    ap.add_argument("--in_method", type=str, default=None,
                    choices=["unineg","fairneg"])
    ap.add_argument("--fairneg_beta", type=float, default=0.5, help="mixup β (Eq. 8)")
    ap.add_argument("--fairneg_tau", type=float, default=0.2, help="softmax temperature τ (Eq. 7)")
    ap.add_argument("--fairneg_lr", type=float, default=0.05, help="outer LR α (Eq. 5)")
    ap.add_argument("--fairneg_momentum", type=float, default=0.9, help="momentum γ (Eq. 5)")
    ap.add_argument("--fairneg_pool", type=int, default=64, help="candidate pool size per user")
    ap.add_argument("--post_method", type=str, default=None,
                choices=[None, "rerank"],
                help="Optional post-process step. 'rerank' applies stochastic re-ranking for exposure fairness.")
    # in main()
    ap.add_argument("--save_mode", type=str, default="both",
                    choices=["both","summary_only","per_run"],
                    help="both: write per-run JSON + append to summary.csv; "
                         "summary_only: only append to summary.csv; "
                         "per_run: only per-run JSON.")
    args = ap.parse_args()
    np.random.seed(args.seed)

    print("\n=== Config ===")
    print(f"Dataset: {args.data_path}, model: {args.model}, balancing: {args.balancing}, pre: {args.pre_method}, in: {args.in_method}, post: {args.post_method}")

    df = pd.read_csv(args.data_path)[["user_id","item_id","rating","genre"]].copy()
    df["user_id"] = df["user_id"].astype(int); df["item_id"] = df["item_id"].astype(int)
    df = df.drop_duplicates(subset=["user_id","item_id"]).reset_index(drop=True)

    # ---- pre-processing -------

    # after loading & deduping df, before random_split(...)
    if args.pre_method == "genre":
        df = preprocess_by_genre(df, enable_weights=True)
    elif args.pre_method == "user":
        df = preprocess_by_user_activity(df, percentile=args.user_activity_percentile, enable_weights=True)
    
    # ---- data balancing -------
    
    if args.balancing == 'random':
        df = random_sampling(df)
    
    if args.balancing == 'cluster':
        df = cluster_based_sampling(df)
        
    # ---- data balancing end ----

    # train_df, _, test_df = leave_one_out(df, per_user_val=1, per_user_test=1, seed=args.seed)

    # trainer = MFTrainer(dim=args.latent_dim, lr=args.lr, epochs=args.epochs,
    #                     batch_size=args.batch_size, device="auto")
    # trainer.fit(train_df)

    train_df, val_df, test_df = random_split(
        df, train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )

    # Build seen_ui map from TRAIN ONLY (for masking)
    seen_ui = {}
    for _, r in train_df.iterrows():
        # local indices will be built inside trainer; here keep globals then map later
        pass

    # Initialize model
    if args.model == "mf":
        trainer = MFTrainer(dim=args.latent_dim, lr=args.lr, epochs=args.epochs,
                            batch_size=args.batch_size, device="auto",
                            in_method=args.in_method, fairneg_beta=args.fairneg_beta, 
                            fairneg_tau=args.fairneg_tau, fairneg_lr=args.fairneg_lr,
                            fairneg_momentum=args.fairneg_momentum, fairneg_pool=args.fairneg_pool
                            )
    else:  # LightGCN
        trainer = LightGCNTrainer(dim=args.latent_dim, lr=args.lr, epochs=args.epochs,
                                  batch_size=args.batch_size, n_layers=3, device="auto",
                                  in_method=args.in_method, fairneg_beta=args.fairneg_beta, 
                                  fairneg_tau=args.fairneg_tau, fairneg_lr=args.fairneg_lr,
                                  fairneg_momentum=args.fairneg_momentum, fairneg_pool=args.fairneg_pool)
    trainer.fit(train_df)

    # Rebuild seen_ui in LOCAL indices after maps exist
    seen_ui = {}
    for _, r in train_df.iterrows():
        u = trainer.user2idx.get(int(r.user_id)); i = trainer.item2idx.get(int(r.item_id))
        if u is not None and i is not None:
            seen_ui.setdefault(u, set()).add(i)
    # ---------------
    users = sorted(test_df.user_id.unique().tolist())
    rows = []
    for u in users:
        for item_id, score in trainer.recommend_topk(u, k=args.topk):
            rows.append({"user_id": u, "item_id": item_id, "prediction": score})
    preds_df = pd.DataFrame(rows)

    # --- Post-process: stochastic re-ranking ---
    if args.post_method == "rerank":
        item2group   = _item2group_map(train_df if "genre" in train_df.columns else test_df)
        if not item2group.empty:
            target_props = _target_props(train_df)
            reranked=[]
            for uid,g in preds_df.groupby("user_id",as_index=False):
                reranked.append(_stochastic_rerank_user(g,item2group,target_props,k=args.topk))
            preds_df=pd.concat(reranked,ignore_index=True)

    # Quality
    def f1_score(p, r):
        return 2 * p * r / (p + r) if p + r > 0 else 0
    p = precision_at_k(preds_df, test_df, args.topk)
    r = recall_at_k(preds_df, test_df, args.topk)
    f1 = f1_score(p, r)
    n = ndcg_at_k(preds_df, test_df, args.topk)
    # Fairness
    g = gini_index(preds_df, args.topk)
    kl = kl_divergence_items(preds_df, test_df, args.topk)
    rd = recall_dispersion(preds_df, test_df, args.topk)
    # Fairness (genre disparity)
    kl_gen = kl_genre_divergence(preds_df, test_df, args.topk)
    rd_gen = recall_dispersion_genre(preds_df, test_df, args.topk)

    metrics = {
        "precision": p,
        "recall": r,
        "f1": f1,
        "ndcg": n,
        "gini": g,
        "kl": kl,
        "recall_disp": rd,
        "kl_genre": kl_gen,
        "recall_disp_genre": rd_gen,
    }

    # _save_run(args, metrics, out_dir="runs")
    _save_run(args, metrics, out_dir="runs", save_mode=args.save_mode)
    
    print("\n=== Quality Metrics ===")
    print(f"Precision@{args.topk}: {p:.6f}")
    print(f"Recall@{args.topk}:    {r:.6f}")
    print(f"F1@{args.topk}:        {f1:.6f}")
    print(f"NDCG@{args.topk}:      {n:.6f}")
    print("\n=== Fairness Metrics ===")
    print(f"Gini:                  {g:.6f}")
    print(f"KL(topK||GT):          {kl:.6f}")
    print(f"Recall Dispersion:     {rd:.6f}")
    print("\n=== Fairness Metrics (Genre Disparity) ===")
    print(f"KL(genre_topK||genre_GT): {kl_gen:.6f}")
    print(f"Recall Dispersion (by genre): {rd_gen:.6f}")

    # Diagnostics
    print("\n=== Diagnostics ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Train users/items: {train_df.user_id.nunique()}/{train_df.item_id.nunique()}")
    print(f"Val   users/items: {val_df.user_id.nunique()}/{val_df.item_id.nunique()}")
    print(f"Test  users/items: {test_df.user_id.nunique()}/{test_df.item_id.nunique()}")
    print(f"Train pairs: {len(train_df)}, Val pairs: {len(val_df)}, Test pairs: {len(test_df)}")
    print(f"Users with preds: {preds_df.user_id.nunique()} / {test_df.user_id.nunique()}")


if __name__ == "__main__":
    main()