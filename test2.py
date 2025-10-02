# test2.py
# MF & LightGCN with in-process fairness constraints:
#   --in_method ugf  : user-oriented group fairness regularizer
#   --in_method rb   : rating-bias (评分偏差) constraint
#   --in_method none : vanilla training (default)
#
# Expected CSV columns (at least): user_id, item_id
# Optional: rating (for RB), genre/item_group (for UGF)

from __future__ import annotations
import argparse, math, random
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from src.data_preparation.dataset_balancing import *
# from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# ------------------ utils ------------------

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def device_from_flag(flag: str) -> torch.device:
    if flag == "cuda" or (flag == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")

def safe_softmax(x: torch.Tensor, dim: int = -1, tau: float = 1.0) -> torch.Tensor:
    x = x / max(tau, 1e-6)
    x = x - x.max(dim=dim, keepdim=True).values
    return F.softmax(x, dim=dim)

def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps); q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    return 0.5 * (p * (p / m).log()).sum(dim=1) + 0.5 * (q * (q / m).log()).sum(dim=1)

def l2_dist(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return ((p - q) ** 2).sum(dim=1)

def build_item_group_tensor(item2idx: Dict[int,int], item_group_map: Dict[int,str]) -> Tuple[torch.Tensor, Dict[str,int]]:
    names = {}
    for iid in item2idx.keys():
        names[str(item_group_map.get(int(iid), "unknown"))] = 1
    name2id = {name: gid for gid, name in enumerate(sorted(names.keys()))}
    gid_list = []
    for iid, local in sorted(item2idx.items(), key=lambda kv: kv[1]):
        gid_list.append(name2id[str(item_group_map.get(int(iid), "unknown"))])
    gid = torch.tensor(gid_list, dtype=torch.long)
    return gid, name2id

def make_target_mix_for_batch(
    users_tensor: torch.Tensor,
    idx2user: Dict[int,int],
    user_hist_mix: Optional[Dict[int,torch.Tensor]],
    global_mix_vec: torch.Tensor,
    beta: float
) -> torch.Tensor:
    tgt = []
    for u_local in users_tensor.tolist():
        u_raw = idx2user[int(u_local)]
        if (beta > 0.0) and (user_hist_mix is not None) and (u_raw in user_hist_mix):
            tgt.append(((1.0 - beta) * global_mix_vec + beta * user_hist_mix[u_raw]).unsqueeze(0))
        else:
            tgt.append(global_mix_vec.unsqueeze(0))
    return torch.cat(tgt, dim=0)

# ------------------ MF ------------------

class MFNet(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def score(self, u_idx: torch.Tensor, i_idx: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(u_idx)  # [B,d]
        if i_idx.dim() == 2:
            i = self.item_emb(i_idx)  # [B,C,d]
            return (i * u.unsqueeze(1)).sum(dim=-1)  # [B,C]
        i = self.item_emb(i_idx)                   # [B,d]
        return (i * u).sum(dim=-1)                 # [B]

class MFTrainer:
    def __init__(
        self,
        dim=64, lr=1e-3, epochs=50, batch_size=1024, reg_weight=0.0, neg_per_pos=1,
        in_method: str = "none",
        # UGF
        ugf_lambda: float = 0.2, ugf_tau: float = 0.7, ugf_use_js: bool = True,
        ugf_beta_user: float = 0.0, ugf_target_mix: Optional[str] = None,  # "uniform" or "from_data"
        # RB
        rb_lambda: float = 0.1, rb_alpha: float = 1.2, rb_beta: float = 0.5,
        device: str = "auto",
        # ==== Early Stopping (new) ====
        es_patience: int = 8,     # stop if no improvement for N epochs
        es_min_delta: float = 1e-4, # required improvement margin
        es_warmup: int = 3,        # ignore ES for first N epochs
        es_cooldown: int = 1       # after an improvement, wait N epochs before counting patience
    ):
        self.dim, self.lr, self.epochs, self.bs = dim, lr, epochs, batch_size
        self.reg_weight, self.neg_per_pos = reg_weight, neg_per_pos
        self.in_method = in_method.lower()
        self.device = device_from_flag(device)

        # maps
        self.user2idx: Dict[int,int] = {}
        self.item2idx: Dict[int,int] = {}
        self.idx2user: Dict[int,int] = {}
        self.idx2item: Dict[int,int] = {}

        self.user_pos: Dict[int,set] = defaultdict(set)
        self.net: Optional[MFNet] = None

        # UGF cfg/state
        self.ugf_lambda = ugf_lambda
        self.ugf_tau = ugf_tau
        self.ugf_use_js = ugf_use_js
        self.ugf_beta_user = ugf_beta_user
        self.ugf_target_mix_mode = ugf_target_mix or "from_data"
        self.item_gid: Optional[torch.Tensor] = None
        self.gname2id: Dict[str,int] = {}
        self.num_groups = 0
        self.global_mix: Optional[torch.Tensor] = None
        self.user_hist_mix: Optional[Dict[int,torch.Tensor]] = None

        # RB cfg/state
        self.rb_lambda = rb_lambda
        self.rb_alpha = rb_alpha
        self.rb_beta = rb_beta
        self.global_avg_rating: float = 0.0
        self.user_rating_bias: Dict[int,float] = {}

        # ==== Early Stopping state (new) ====
        self.es_patience   = int(es_patience)
        self.es_min_delta  = float(es_min_delta)
        self.es_warmup     = int(es_warmup)
        self.es_cooldown   = int(es_cooldown)

    # ---------- fit ----------
    def fit(self, train_df: pd.DataFrame):
        users = train_df.user_id.astype(int).unique().tolist()
        items = train_df.item_id.astype(int).unique().tolist()
        self.user2idx = {u:i for i,u in enumerate(users)}
        self.item2idx = {v:j for j,v in enumerate(items)}
        self.idx2user = {i:u for u,i in self.user2idx.items()}
        self.idx2item = {j:v for v,j in self.item2idx.items()}

        # interactions for filtering
        for _, r in train_df[['user_id','item_id']].astype(int).iterrows():
            self.user_pos[int(r.user_id)].add(int(r.item_id))

        self.net = MFNet(len(self.user2idx), len(self.item2idx), self.dim).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        # ----- UGF prepare -----
        if self.in_method == "ugf":
            # build item groups from 'genre' or 'item_group' if available
            if 'genre' in train_df.columns:
                ig_map = {int(i): str(g) for i, g in train_df[['item_id','genre']].drop_duplicates().values}
            elif 'item_group' in train_df.columns:
                ig_map = {int(i): str(g) for i, g in train_df[['item_id','item_group']].drop_duplicates().values}
            else:
                ig_map = {int(i): "unknown" for i in items}
            self.item_gid, self.gname2id = build_item_group_tensor(self.item2idx, ig_map)
            self.item_gid = self.item_gid.to(self.device)
            self.num_groups = len(self.gname2id)

            # global target mix
            if self.ugf_target_mix_mode == "uniform" or self.num_groups == 0:
                gm = torch.ones(self.num_groups, dtype=torch.float32)
            else:
                # from data by group frequency
                counts = torch.zeros(self.num_groups, dtype=torch.float32)
                g_series = [self.gname2id[str(ig_map[int(i)])] for i in items]
                for gid in g_series: counts[gid] += 1.0
                gm = counts.clamp_min(1.0)
            self.global_mix = (gm / gm.sum()).to(self.device)

            # per-user historical mix (over train positives)
            if self.ugf_beta_user > 0.0:
                self.user_hist_mix = {}
                # make sure a column exists for groups
                if 'genre' in train_df.columns:
                    grp_col = 'genre'
                elif 'item_group' in train_df.columns:
                    grp_col = 'item_group'
                else:
                    grp_col = None
                if grp_col is not None:
                    for uid, g in train_df.groupby('user_id'):
                        cnt = torch.zeros(self.num_groups, dtype=torch.float32)
                        for name, gid in self.gname2id.items():
                            cnt[gid] = (g[grp_col].astype(str) == name).sum()
                        if cnt.sum() <= 0: cnt += 1.0
                        self.user_hist_mix[int(uid)] = (cnt / cnt.sum())

        # ----- RB prepare -----
        if self.in_method == "rb":
            if 'rating' not in train_df.columns:
                raise ValueError("RB constraint requires a 'rating' column in the training data.")
            self.global_avg_rating = float(train_df['rating'].astype(float).mean())
            avg_by_user = train_df.groupby('user_id')['rating'].mean().astype(float)
            for uid, uavg in avg_by_user.items():
                self.user_rating_bias[int(uid)] = abs(float(uavg) - self.global_avg_rating)

        # samples
        data = train_df[['user_id','item_id']].astype(int).values
        user_idx = np.array([self.user2idx[u] for u,_ in data], dtype=np.int64)
        item_idx = np.array([self.item2idx[i] for _,i in data], dtype=np.int64)
        all_items_local = np.arange(len(self.item2idx), dtype=np.int64)

        n = len(user_idx)
        steps = max(1, math.ceil(n / self.bs))

        # ==== Early stopping bookkeeping (new) ====
        best_loss = float("inf")
        best_state = None
        epochs_no_improve = 0
        cooldown_left = 0

        for ep in range(1, self.epochs+1):
            perm = np.random.permutation(n)
            user_idx = user_idx[perm]; item_idx = item_idx[perm]
            ep_loss = 0.0

            for t in range(steps):
                st, ed = t*self.bs, min(n, (t+1)*self.bs)
                if st >= ed: continue
                u_b = torch.as_tensor(user_idx[st:ed], device=self.device, dtype=torch.long)  # [B]
                i_pos = torch.as_tensor(item_idx[st:ed], device=self.device, dtype=torch.long) # [B]

                # negatives [B, Nneg]
                neg = np.random.choice(all_items_local, size=(i_pos.size(0), self.neg_per_pos), replace=True)
                i_neg = torch.as_tensor(neg, device=self.device, dtype=torch.long)

                self.net.train()
                s_pos = self.net.score(u_b, i_pos)          # [B]
                s_neg = self.net.score(u_b, i_neg)          # [B, Nneg]

                bpr = -F.logsigmoid(s_pos.unsqueeze(1) - s_neg).mean()

                reg = 0.0
                if self.reg_weight > 0:
                    reg = self.reg_weight * (
                        self.net.user_emb(u_b).pow(2).mean()
                        + self.net.item_emb(i_pos).pow(2).mean()
                        + self.net.item_emb(i_neg).pow(2).mean()
                    )

                loss = bpr + (reg if isinstance(reg, torch.Tensor) else torch.tensor(reg, device=self.device))

                # ---- UGF penalty ----
                if self.in_method == "ugf" and self.num_groups >= 2:
                    cand = torch.cat([i_pos.unsqueeze(1), i_neg], dim=1)         # [B, 1+Nneg]
                    scores = self.net.score(u_b, cand)                           # [B, 1+Nneg]
                    probs  = safe_softmax(scores, dim=1, tau=self.ugf_tau)       # [B, 1+Nneg]
                    gid    = self.item_gid[cand]                                 # [B, 1+Nneg]
                    mask   = F.one_hot(gid, num_classes=self.num_groups).float()# [B, 1+Nneg, G]
                    exp_g  = (probs.unsqueeze(-1) * mask).sum(dim=1)            # [B, G]
                    exp_g  = exp_g / (exp_g.sum(dim=1, keepdim=True) + 1e-12)

                    tgt = make_target_mix_for_batch(
                        u_b, self.idx2user, self.user_hist_mix, self.global_mix, self.ugf_beta_user
                    ).to(self.device)                                            # [B, G]

                    ugf_loss = js_divergence(exp_g, tgt).mean() if self.ugf_use_js else l2_dist(exp_g, tgt).mean()
                    loss = loss + self.ugf_lambda * ugf_loss

                # ---- RB penalty ----
                if self.in_method == "rb":
                    # map local->raw ids
                    raw_ids = [self.idx2user[int(u)] for u in u_b.detach().cpu().tolist()]
                    cap = torch.tensor(
                        [self.rb_alpha * (self.user_rating_bias.get(uid, 0.0) + self.rb_beta) for uid in raw_ids],
                        dtype=torch.float32, device=self.device
                    )
                    diff = (s_pos - self.global_avg_rating).abs()
                    rb_loss = (diff - cap).clamp_min(0.0).mean()
                    loss = loss + self.rb_lambda * rb_loss

                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += float(loss.detach().cpu())

            avg_loss = ep_loss / max(1, steps)
            print(f"[MF] Epoch {ep}/{self.epochs} loss={avg_loss:.4f}")

            # ==== Early stopping check (new) ====
            improved = (best_loss - avg_loss) > self.es_min_delta
            if improved:
                best_loss = avg_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                epochs_no_improve = 0
                cooldown_left = self.es_cooldown
                print(f"[MF][ES] ✓ New best loss: {best_loss:.6f}")
            else:
                if ep > self.es_warmup:
                    if cooldown_left > 0:
                        cooldown_left -= 1
                        print(f"[MF][ES] cooldown {cooldown_left} left (no patience count this epoch)")
                    else:
                        epochs_no_improve += 1
                        print(f"[MF][ES] no improve: {epochs_no_improve}/{self.es_patience}")
                        if epochs_no_improve >= self.es_patience:
                            print(f"[MF][ES] Early stopping at epoch {ep}. Best loss: {best_loss:.6f}")
                            break

        # ==== Restore best weights (new) ====
        if best_state is not None:
            self.net.load_state_dict(best_state)
            self.net.to(self.device)
            print("[MF][ES] Restored best model weights.")

    @torch.no_grad()
    def recommend_topk(self, user_id: int, k: int = 20, seen_ui=None):
        if self.net is None or user_id not in self.user2idx:
            return []
    
        # Support both MFNet variants: (user_emb/item_emb) or (user/item)
        U = getattr(self.net, "user_emb", getattr(self.net, "user", None))
        V = getattr(self.net, "item_emb", getattr(self.net, "item", None))
        if U is None or V is None:
            raise AttributeError("MFNet must expose user_emb/item_emb (or user/item) nn.Embedding tables.")
    
        u_local = self.user2idx[user_id]
        u = torch.tensor([u_local], dtype=torch.long, device=self.device)   # [1]
        # scores over all items
        scores = (U(u) * V.weight).sum(dim=1).squeeze(0)                    # (I,)
    
        # optional: mask training positives (expects LOCAL item ids in seen_ui[u_local])
        if seen_ui is not None and u_local in seen_ui:
            idx = torch.tensor(list(seen_ui[u_local]), dtype=torch.long, device=scores.device)
            if idx.numel() > 0:
                scores.index_fill_(0, idx, float("-inf"))
    
        k = min(k, scores.numel())
        topk_idx = torch.topk(scores, k).indices.tolist()
        return [(self.idx2item[i], float(scores[i].detach().cpu())) for i in topk_idx]

# ------------------ LightGCN ------------------

import scipy.sparse as sp

def _sparse_coo_torch(mx):
    mx = mx.tocoo().astype(np.float32)
    idx = torch.from_numpy(np.vstack((mx.row, mx.col)).astype(np.int64))
    val = torch.from_numpy(mx.data)
    return torch.sparse_coo_tensor(idx, val, torch.Size(mx.shape))

class LightGCNNet(nn.Module):
    def __init__(self, n_users, n_items, dim, n_layers):
        super().__init__()
        self.n_users, self.n_items, self.dim, self.n_layers = n_users, n_items, dim, n_layers
        self.user = nn.Embedding(n_users, dim); self.item = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user.weight, std=0.01); nn.init.normal_(self.item.weight, std=0.01)
        self.cached_user = None; self.cached_item = None

    def _propagate_fp32(self, norm_adj: torch.Tensor):
        with torch.amp.autocast('cuda', enabled=False):
            all_emb = torch.cat([self.user.weight, self.item.weight], dim=0).float()
            embs = [all_emb]
            for _ in range(self.n_layers):
                all_emb = torch.sparse.mm(norm_adj.float(), all_emb)
                embs.append(all_emb)
            out = torch.stack(embs, dim=1).mean(dim=1)
        return out[:self.n_users], out[self.n_users:]

    def update_cache(self, norm_adj: torch.Tensor):
        u, i = self._propagate_fp32(norm_adj)
        self.cached_user, self.cached_item = u, i

    def score(self, u_idx: torch.Tensor, i_idx: torch.Tensor) -> torch.Tensor:
        if self.cached_user is None or self.cached_item is None:
            raise RuntimeError("Call update_cache(norm_adj) before score().")
        if i_idx.dim() == 2:
            u = self.cached_user[u_idx]                 # [B,d]
            i = self.cached_item[i_idx]                 # [B,C,d]
            return (i * u.unsqueeze(1)).sum(dim=-1)     # [B,C]
        u = self.cached_user[u_idx]                     # [B,d]
        i = self.cached_item[i_idx]                     # [B,d]
        return (i * u).sum(dim=-1)                      # [B]

class LightGCNTrainer(MFTrainer):
    def __init__(self, n_layers=3, **kwargs):
        super().__init__(**kwargs)
        self.n_layers = n_layers
        self.norm_adj: Optional[torch.Tensor] = None

    def _build_norm_adj(self, df: pd.DataFrame):
        rows, cols, data = [], [], []
        for _, r in df[['user_id','item_id']].astype(int).iterrows():
            u = self.user2idx.get(int(r.user_id)); i = self.item2idx.get(int(r.item_id))
            if u is None or i is None: continue
            rows.append(u); cols.append(i); data.append(1.0)
        A = sp.coo_matrix((data, (rows, cols)), shape=(len(self.user2idx), len(self.item2idx)), dtype=np.float32)
        top = sp.hstack([sp.csr_matrix((A.shape[0], A.shape[0])), A.tocsr()], format='csr')
        bottom = sp.hstack([A.T.tocsr(), sp.csr_matrix((A.shape[1], A.shape[1]))], format='csr')
        adj = sp.vstack([top, bottom], format='csr')
        deg = np.array(adj.sum(1)).flatten()
        d_inv_sqrt = np.power(deg, -0.5, where=deg>0); d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
        D = sp.diags(d_inv_sqrt)
        norm = D @ adj @ D
        return _sparse_coo_torch(norm).coalesce().to(self.device).float()

    def fit(self, train_df: pd.DataFrame):
        # build maps then graph
        users = train_df.user_id.astype(int).unique().tolist()
        items = train_df.item_id.astype(int).unique().tolist()
        self.user2idx = {u:i for i,u in enumerate(users)}
        self.item2idx = {v:j for j,v in enumerate(items)}
        self.idx2user = {i:u for u,i in self.user2idx.items()}
        self.idx2item = {j:v for v,j in self.item2idx.items()}
        for _, r in train_df[['user_id','item_id']].astype(int).iterrows():
            self.user_pos[int(r.user_id)].add(int(r.item_id))

        self.norm_adj = self._build_norm_adj(train_df)
        self.net = LightGCNNet(len(self.user2idx), len(self.item2idx), self.dim, self.n_layers).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=0.0)

        # UGF/RB preparation – reuse MFTrainer logic
        # we’ll temporarily construct a tiny MF wrapper for convenient computations (global mix etc.)
        mf_like = MFTrainer(
            dim=self.dim, lr=self.lr, epochs=1, batch_size=self.bs, reg_weight=self.reg_weight,
            neg_per_pos=self.neg_per_pos, in_method=self.in_method,
            ugf_lambda=self.ugf_lambda, ugf_tau=self.ugf_tau, ugf_use_js=self.ugf_use_js,
            ugf_beta_user=self.ugf_beta_user, ugf_target_mix=self.ugf_target_mix_mode,
            rb_lambda=self.rb_lambda, rb_alpha=self.rb_alpha, rb_beta=self.rb_beta,
            device=str(self.device).split(':')[0]
        )
        # copy maps then call only the “prep” sections by invoking internal helpers:
        mf_like.user2idx, mf_like.item2idx = self.user2idx, self.item2idx
        mf_like.idx2user, mf_like.idx2item = self.idx2user, self.idx2item

        if self.in_method == "ugf":
            if 'genre' in train_df.columns:
                ig_map = {int(i): str(g) for i, g in train_df[['item_id','genre']].drop_duplicates().values}
            elif 'item_group' in train_df.columns:
                ig_map = {int(i): str(g) for i, g in train_df[['item_id','item_group']].drop_duplicates().values}
            else:
                ig_map = {int(i): "unknown" for i in items}
            self.item_gid, self.gname2id = build_item_group_tensor(self.item2idx, ig_map)
            self.item_gid = self.item_gid.to(self.device)
            self.num_groups = len(self.gname2id)
            if self.ugf_target_mix_mode == "uniform" or self.num_groups == 0:
                gm = torch.ones(self.num_groups, dtype=torch.float32)
            else:
                counts = torch.zeros(self.num_groups, dtype=torch.float32)
                g_series = [self.gname2id[str(ig_map[int(i)])] for i in items]
                for gid in g_series: counts[gid] += 1.0
                gm = counts.clamp_min(1.0)
            self.global_mix = (gm / gm.sum()).to(self.device)
            self.user_hist_mix = None
            if self.ugf_beta_user > 0.0:
                self.user_hist_mix = {}
                grp_col = 'genre' if 'genre' in train_df.columns else ('item_group' if 'item_group' in train_df.columns else None)
                if grp_col is not None:
                    for uid, g in train_df.groupby('user_id'):
                        cnt = torch.zeros(self.num_groups, dtype=torch.float32)
                        for name, gid in self.gname2id.items():
                            cnt[gid] = (g[grp_col].astype(str) == name).sum()
                        if cnt.sum() <= 0: cnt += 1.0
                        self.user_hist_mix[int(uid)] = (cnt / cnt.sum())

        if self.in_method == "rb":
            if 'rating' not in train_df.columns:
                raise ValueError("RB constraint requires a 'rating' column in the training data.")
            self.global_avg_rating = float(train_df['rating'].astype(float).mean())
            avg_by_user = train_df.groupby('user_id')['rating'].mean().astype(float)
            self.user_rating_bias = {int(uid): abs(float(avg) - self.global_avg_rating) for uid, avg in avg_by_user.items()}

        # arrays
        data = train_df[['user_id','item_id']].astype(int).values
        user_idx = np.array([self.user2idx[u] for u,_ in data], dtype=np.int64)
        item_idx = np.array([self.item2idx[i] for _,i in data], dtype=np.int64)
        all_items_local = np.arange(len(self.item2idx), dtype=np.int64)

        n = len(user_idx); steps = max(1, math.ceil(n / self.bs))

        # ===== Early stopping bookkeeping (new) =====
        best_loss = float("inf")
        best_state = None
        epochs_no_improve = 0
        cooldown_left = 0

        for ep in range(1, self.epochs+1):
            perm = np.random.permutation(n)
            user_idx = user_idx[perm]; item_idx = item_idx[perm]
            # self.net.update_cache(self.norm_adj)
            ep_loss = 0.0
            for t in range(steps):
                st, ed = t*self.bs, min(n, (t+1)*self.bs)
                if st >= ed: continue
                u_b = torch.as_tensor(user_idx[st:ed], device=self.device, dtype=torch.long)
                i_pos = torch.as_tensor(item_idx[st:ed], device=self.device, dtype=torch.long)

                self.net.update_cache(self.norm_adj)

                neg = np.random.choice(all_items_local, size=(i_pos.size(0), self.neg_per_pos), replace=True)
                i_neg = torch.as_tensor(neg, device=self.device, dtype=torch.long)

                s_pos = self.net.score(u_b, i_pos)
                s_neg = self.net.score(u_b, i_neg)
                bpr = -F.logsigmoid(s_pos.unsqueeze(1) - s_neg).mean()

                # tiny L2 on cached embeddings used this step
                reg = 0.0
                if self.reg_weight > 0:
                    reg = self.reg_weight * (
                        self.net.cached_user[u_b].pow(2).mean()
                        + self.net.cached_item[i_pos].pow(2).mean()
                        + self.net.cached_item[i_neg].pow(2).mean()
                    )

                loss = bpr + (reg if isinstance(reg, torch.Tensor) else torch.tensor(reg, device=self.device))

                if self.in_method == "ugf" and self.num_groups >= 2:
                    cand = torch.cat([i_pos.unsqueeze(1), i_neg], dim=1)           # [B,1+Nneg]
                    scores = self.net.score(u_b, cand)                              # [B,1+Nneg]
                    probs  = safe_softmax(scores, dim=1, tau=self.ugf_tau)
                    gid    = self.item_gid[cand]
                    mask   = F.one_hot(gid, num_classes=self.num_groups).float()
                    exp_g  = (probs.unsqueeze(-1) * mask).sum(dim=1)
                    exp_g  = exp_g / (exp_g.sum(dim=1, keepdim=True) + 1e-12)

                    tgt = make_target_mix_for_batch(u_b, self.idx2user, self.user_hist_mix, self.global_mix, self.ugf_beta_user).to(self.device)
                    ugf_loss = js_divergence(exp_g, tgt).mean() if self.ugf_use_js else l2_dist(exp_g, tgt).mean()
                    loss = loss + self.ugf_lambda * ugf_loss

                if self.in_method == "rb":
                    raw_ids = [self.idx2user[int(u)] for u in u_b.detach().cpu().tolist()]
                    cap = torch.tensor(
                        [self.rb_alpha * (self.user_rating_bias.get(uid, 0.0) + self.rb_beta) for uid in raw_ids],
                        dtype=torch.float32, device=self.device
                    )
                    diff = (s_pos - self.global_avg_rating).abs()
                    rb_loss = (diff - cap).clamp_min(0.0).mean()
                    loss = loss + self.rb_lambda * rb_loss

                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += float(loss.detach().cpu())

            avg_loss = ep_loss/max(1,steps)
            print(f"[LightGCN] Epoch {ep}/{self.epochs} loss={avg_loss:.6f}")

            # ===== Early stopping check (new) =====
            improved = (best_loss - avg_loss) > self.es_min_delta
            if improved:
                best_loss = avg_loss
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

    @torch.no_grad()
    def recommend_topk(self, user_id: int, k: int = 20, seen_ui=None):
        # user_id is RAW id here
        if self.net is None or user_id not in self.user2idx:
            return []
    
        # ensure cache is ready (don’t recompute if already cached)
        if self.net.cached_user is None or self.net.cached_item is None:
            self.net.update_cache(self.norm_adj)
    
        u_local = self.user2idx[user_id]
        u_vec   = self.net.cached_user[u_local].unsqueeze(0)          # (1, D)
        scores  = torch.matmul(u_vec, self.net.cached_item.T).squeeze(0)  # (I,)
    
        # --- mask seen items (supports either LOCAL or RAW seen_ui) ---
        if seen_ui is not None:
            # try local first
            mask_items = None
            if u_local in seen_ui:
                mask_items = list(seen_ui[u_local])
                # assume they are LOCAL item ids already
                idx = torch.tensor(mask_items, dtype=torch.long, device=scores.device)
            else:
                # maybe RAW map: {raw_user_id: {raw_item_id,...}}
                raw_items = seen_ui.get(user_id, None)
                if raw_items:
                    # map raw item ids -> local
                    li = [self.item2idx[i] for i in raw_items if i in self.item2idx]
                    if li:
                        idx = torch.tensor(li, dtype=torch.long, device=scores.device)
                    else:
                        idx = None
                else:
                    idx = None
            if idx is not None and idx.numel() > 0:
                scores.index_fill_(0, idx, float("-inf"))
    
        k = min(k, scores.numel())
        # all-masked safety: if everything is -inf, return empty
        if not torch.isfinite(scores).any():
            return []
    
        topk_idx = torch.topk(scores, k).indices.tolist()
        return [(self.idx2item[i], float(scores[i].detach().cpu())) for i in topk_idx]

# ------- save runs -------

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

# ------------------ argparse / main ------------------

def build_parser():
    p = argparse.ArgumentParser("test2.py — MF/LightGCN with UGF or Rating-Bias constraints")
    # data
    p.add_argument("--data_path", type=str, required=True, help="CSV with columns user_id,item_id[,rating,genre/item_group]")
    # model
    p.add_argument("--model", choices=["mf","lightgcn"], default="mf")
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=3, help="LightGCN layers")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--reg_weight", type=float, default=0.0)
    p.add_argument("--neg_per_pos", type=int, default=1)
    p.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--train_ratio", type=float, default=0.6)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--balancing", type=str, default=None)

    p.add_argument("--pre_method", type=str, choices=["genre","user"], default=None,
                    help="Enable pre-processing & weighting: 'genre' or 'user'. Omit for none.")

    # fairness in-process
    p.add_argument("--in_method", choices=["ugf","rb"], default="ugf")

    # UGF knobs
    p.add_argument("--ugf_lambda", type=float, default=0.2)
    p.add_argument("--ugf_tau", type=float, default=0.7)
    p.add_argument("--ugf_use_js", action="store_true", default=True)
    p.add_argument("--ugf_l2", dest="ugf_use_js", action="store_false", help="use L2 instead of JS divergence")
    p.add_argument("--ugf_beta_user", type=float, default=0.0, help="mix global target with per-user history (0~1)")
    p.add_argument("--ugf_target_mix", choices=["from_data","uniform"], default="from_data")

    # Rating-bias knobs
    p.add_argument("--rb_lambda", type=float, default=0.1)
    p.add_argument("--rb_alpha", type=float, default=1.2)
    p.add_argument("--rb_beta", type=float, default=0.5)

    # in main()
    p.add_argument("--save_mode", type=str, default="both",
                    choices=["both","summary_only","per_run"],
                    help="both: write per-run JSON + append to summary.csv; "
                         "summary_only: only append to summary.csv; "
                         "per_run: only per-run JSON.")

    return p

def main():
    args = build_parser().parse_args()
    print("\n=== Config ===")
    print(f"Dataset: {args.data_path}, model: {args.model}, balancing: {args.balancing}, in: {args.in_method}")
    set_seed(args.seed)

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

    common_kwargs = dict(
        dim=args.latent_dim, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
        reg_weight=args.reg_weight, neg_per_pos=args.neg_per_pos,
        in_method=args.in_method,
        ugf_lambda=args.ugf_lambda, ugf_tau=args.ugf_tau, ugf_use_js=args.ugf_use_js,
        ugf_beta_user=args.ugf_beta_user, ugf_target_mix=args.ugf_target_mix,
        rb_lambda=args.rb_lambda, rb_alpha=args.rb_alpha, rb_beta=args.rb_beta,
        device=args.device,
    )

    if args.model == "mf":
        trainer = MFTrainer(**common_kwargs)
    else:
        trainer = LightGCNTrainer(n_layers=args.n_layers, **common_kwargs)

    print(f"Device: {trainer.device} | Model: {args.model} | in_method: {args.in_method}")
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
        for item_id, score in trainer.recommend_topk(u, k=20, seen_ui=seen_ui):
            rows.append({"user_id": u, "item_id": item_id, "prediction": score})
    preds_df = pd.DataFrame(rows)

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

    # after computing `metrics` in main():
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