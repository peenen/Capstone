from collections import defaultdict
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

def _device_from_config(cfg: Dict[str, Any]) -> torch.device:
    dev = cfg.get("model", {}).get("device", "auto")
    if dev == "cuda" or (dev == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def _safe_softmax(x: torch.Tensor, dim: int = -1, tau: float = 1.0) -> torch.Tensor:
    x = x / max(tau, 1e-6)
    x = x - x.max(dim=dim, keepdim=True).values
    return F.softmax(x, dim=dim)


def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(dim=1)
    kl_qm = (q * (q / m).log()).sum(dim=1)
    return 0.5 * (kl_pm + kl_qm)


def _l2_dist(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return ((p - q) ** 2).sum(dim=1)


def _build_item_group_id(item2idx: Dict[int, int], item_group_map: Dict[int, str]) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    把 item_id→group_name 的映射，转成：
      - gid_tensor: [num_items]，按“本模型内 item 索引”的顺序排列的 group id
      - group_name2id: {'popular':0, 'unpopular':1, 'unknown':2, ...}
    """
    group_names = {}
    for iid in item2idx.keys():
        gname = item_group_map.get(int(iid), "unknown")
        group_names[gname] = 1
    name2id = {name: gid for gid, name in enumerate(sorted(group_names.keys()))}
    gid_list = []
    for iid, idx in sorted(item2idx.items(), key=lambda kv: kv[1]):
        gname = item_group_map.get(int(iid), "unknown")
        gid_list.append(name2id[gname])
    gid_tensor = torch.tensor(gid_list, dtype=torch.long)
    return gid_tensor, name2id


def _make_target_mix_tensor(users_tensor: torch.Tensor,
                            user_hist_mix: Optional[Dict[int, torch.Tensor]],
                            global_mix_vec: torch.Tensor,
                            beta: float) -> torch.Tensor:
    """
    users_tensor: [B]（原始 user_id）
    user_hist_mix: dict[user_id] -> torch[G]（和 global_mix_vec 的组顺序一致）
    返回 [B, G] 的目标分布
    """
    B = users_tensor.shape[0]
    tgt = []
    for u in users_tensor.tolist():
        if (beta > 0.0) and (user_hist_mix is not None) and (u in user_hist_mix):
            mix = (1.0 - beta) * global_mix_vec + beta * user_hist_mix[u]
        else:
            mix = global_mix_vec
        tgt.append(mix.unsqueeze(0))
    return torch.cat(tgt, dim=0)  # [B, G]


# =============== MF（含 UGF） ===============

class MFNet(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int):
        super().__init__()
        self.user_emb = nn.Embedding(num_embeddings=n_users, embedding_dim=dim)
        self.item_emb = nn.Embedding(num_embeddings=n_items, embedding_dim=dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def score(self, u_idx: torch.Tensor, i_idx: torch.Tensor) -> torch.Tensor:
        """
        u_idx: [B]
        i_idx: [B, C] 或 [B]
        返回：若 [B, C] -> [B, C]；若 [B] -> [B]
        """
        u = self.user_emb(u_idx)  # [B, d]
        if i_idx.dim() == 2:
            i = self.item_emb(i_idx)  # [B, C, d]
            return (i * u.unsqueeze(1)).sum(dim=-1)
        else:
            i = self.item_emb(i_idx)  # [B, d]
            return (i * u).sum(dim=-1)


class MFRecommender:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = _device_from_config(config)
        # 映射
        self.user2idx: Dict[int, int] = {}
        self.item2idx: Dict[int, int] = {}
        self.idx2user: Dict[int, int] = {}
        self.idx2item: Dict[int, int] = {}
        # 训练缓存
        self.user_pos = defaultdict(set)  # 用户 -> 训练集中已交互 item_id
        # 模型
        self.net: Optional[MFNet] = None
        # UGF
        self.enable_ugf = False
        self.lambda_ugf = 0.0
        self.tau = 0.7
        self.use_js = True
        self.beta = 0.0
        self.item_gid_tensor: Optional[torch.Tensor] = None
        self.group_name2id: Dict[str, int] = {}
        self.global_target_vec: Optional[torch.Tensor] = None
        self.user_hist_mix: Optional[Dict[int, torch.Tensor]] = None
        self.num_groups = 0
        # ------------ 评分偏差约束：相关参数--------------
        self.global_avg_rating = 0.0  # 全局平均评分
        self.user_rating_bias = {}    # 用户评分偏差：{user_id: 偏差值}
        self.enable_rating_bias_constraint = False  # 是否启用偏差约束
        self.alpha_bias = 1.2         # 约束宽松度（默认值，可从config读取）
        self.beta_bias = 0.5          # 约束偏移量（默认值，可从config读取）

    # ----- 公共接口 -----

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, group_info: Optional[Dict[str, Any]] = None):
        cfg_m = self.config.get("model", {})
        dim = int(cfg_m.get("latent_dim", 64))
        lr = float(cfg_m.get("lr", 1e-3))
        epochs = int(cfg_m.get("epochs", 10))
        batch_size = int(cfg_m.get("batch_size", 256))
        reg_weight = float(cfg_m.get("reg_weight", 0.0))
        neg_per_pos = int(cfg_m.get("neg_per_pos", 1))

        # 构建索引映射
        users = train_df["user_id"].astype(int).unique().tolist()
        items = train_df["item_id"].astype(int).unique().tolist()
        self.user2idx = {u: i for i, u in enumerate(users)}
        self.item2idx = {v: j for j, v in enumerate(items)}
        self.idx2user = {i: u for u, i in self.user2idx.items()}
        self.idx2item = {j: v for v, j in self.item2idx.items()}

        # -------------------------- 评分偏差约束：统计评分偏差 --------------------------
        # 1. 计算全局平均评分（需确保train_df有"rating"列）
        self.global_avg_rating = train_df["rating"].astype(float).mean()
        # 2. 计算每个用户的评分偏差（用户平均评分 - 全局平均评分的绝对值）
        user_avg_rating = train_df.groupby("user_id")["rating"].mean().reset_index()
        for _, row in user_avg_rating.iterrows():
            uid = int(row["user_id"])
            user_avg = float(row["rating"])
            self.user_rating_bias[uid] = abs(user_avg - self.global_avg_rating)  # 相对偏差
        # ------------------------------------------------------------------------

        # 记录已交互集合（训练用 & 推荐时过滤）
        for _, r in train_df[["user_id", "item_id"]].astype(int).iterrows():
            self.user_pos[int(r.user_id)].add(int(r.item_id))

        # 模型
        self.net = MFNet(len(self.user2idx), len(self.item2idx), dim).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)

        # ====== UGF 初始化（来自 in_process 透传的 group_info / ugf_config） ======
        ugf = (group_info or {}).get("ugf_config", {}) if group_info is not None else {}
        self.enable_ugf = bool(ugf.get("enable_ugf", False))
        self.lambda_ugf = float(ugf.get("lambda_ugf", 0.2))
        self.tau = float(ugf.get("tau", 0.7))
        self.use_js = bool(ugf.get("use_js_divergence", True))
        self.beta = float(ugf.get("beta_user_personalization", 0.0))

        item_group_map = (group_info or {}).get("item_group_map", {}) if group_info is not None else {}
        self.item_gid_tensor, self.group_name2id = _build_item_group_id(self.item2idx, item_group_map)
        self.item_gid_tensor = self.item_gid_tensor.to(self.device)
        self.num_groups = int(len(self.group_name2id))

        target_mix = ugf.get("target_mix", {"popular": 0.6, "unpopular": 0.4})
        global_vec = torch.zeros(self.num_groups, dtype=torch.float32)
        for name, gid in self.group_name2id.items():
            global_vec[gid] = float(target_mix.get(name, 0.0))
        if global_vec.sum() <= 0:
            global_vec += 1.0
        self.global_target_vec = (global_vec / global_vec.sum()).to(self.device)

        self.user_hist_mix = None
        if self.beta > 0:
            df_hist = train_df.copy()
            if "item_group" not in df_hist.columns:
                m = {int(k): v for k, v in item_group_map.items()}
                df_hist["item_group"] = df_hist["item_id"].astype(int).map(lambda i: m.get(int(i), "unknown"))
            self.user_hist_mix = {}
            for uid, g in df_hist.groupby("user_id"):
                counts = torch.zeros(self.num_groups)
                for name, gid in self.group_name2id.items():
                    counts[gid] = (g["item_group"] == name).sum()
                if counts.sum() <= 0:
                    counts += 1.0
                self.user_hist_mix[int(uid)] = (counts / counts.sum())

        # ====== 训练循环（BPR / 简单点积） ======
        # 准备训练样本索引
        data = train_df[["user_id", "item_id"]].astype(int).values
        user_idx = np.array([self.user2idx[u] for u, _ in data], dtype=np.int64)
        item_idx = np.array([self.item2idx[i] for _, i in data], dtype=np.int64)

        # 负采样的物品全集
        all_items_local = np.arange(len(self.item2idx), dtype=np.int64)

        # 若启用“公平负采样”可在 config['in_process']['method']=='negative_sampling' 下调整概率（此处保留均匀）
        sampler_probs = None  # 可扩展：混合均匀 + 逆流行度

        n = len(user_idx)
        steps_per_epoch = max(1, (n + batch_size - 1) // batch_size)

        for epoch in range(1, epochs + 1):
            # 打乱
            perm = np.random.permutation(n)
            user_idx = user_idx[perm]
            item_idx = item_idx[perm]

            epoch_loss = 0.0
            for step in range(steps_per_epoch):
                st = step * batch_size
                ed = min(n, st + batch_size)
                if st >= ed:
                    continue

                u_b = torch.as_tensor(user_idx[st:ed], dtype=torch.long, device=self.device)   # [B]
                i_pos_b = torch.as_tensor(item_idx[st:ed], dtype=torch.long, device=self.device)  # [B]

                # 负采样：对每个正例采 neg_per_pos 个负例
                B = u_b.shape[0]
                if sampler_probs is None:
                    neg_items = np.random.choice(all_items_local, size=(B, neg_per_pos), replace=True)
                else:
                    neg_items = np.random.choice(all_items_local, size=(B, neg_per_pos), replace=True, p=sampler_probs)
                i_neg_b = torch.as_tensor(neg_items, dtype=torch.long, device=self.device)  # [B, Nneg]

                # 主损失：BPR（s(u,i+) > s(u,i-)）
                self.net.train()
                s_pos = self.net.score(u_b, i_pos_b)                 # [B]
                s_neg = self.net.score(u_b, i_neg_b)                 # [B, Nneg]
                # max-margin: log-sigmoid
                bpr = -F.logsigmoid(s_pos.unsqueeze(1) - s_neg).mean()

                # L2 正则（可选）
                if reg_weight > 0:
                    reg = reg_weight * (
                        self.net.user_emb(u_b).pow(2).mean()
                        + self.net.item_emb(i_pos_b).pow(2).mean()
                        + self.net.item_emb(i_neg_b).pow(2).mean()
                    )
                else:
                    reg = 0.0

                loss_main = bpr + (reg if isinstance(reg, torch.Tensor) else torch.tensor(reg, device=self.device))

                # === UGF：批内候选的软曝光分布，与目标分布做散度 ===
                if self.enable_ugf and self.num_groups >= 2:
                    cand_items = torch.cat([i_pos_b.unsqueeze(1), i_neg_b], dim=1)   # [B, 1+Nneg]
                    scores = self.net.score(u_b, cand_items)                         # [B, 1+Nneg]
                    probs = _safe_softmax(scores, dim=1, tau=self.tau)              # [B, 1+Nneg]

                    gid = self.item_gid_tensor[cand_items]                           # [B, 1+Nneg]
                    group_mask = F.one_hot(gid, num_classes=self.num_groups).float()# [B, 1+Nneg, G]
                    exp_by_group = (probs.unsqueeze(-1) * group_mask).sum(dim=1)    # [B, G]
                    exp_by_group = exp_by_group / (exp_by_group.sum(dim=1, keepdim=True) + 1e-12)

                    target = _make_target_mix_tensor(
                        u_b.detach().cpu(), self.user_hist_mix,
                        self.global_target_vec.detach().cpu(), self.beta
                    ).to(self.device)                                                # [B, G]

                    ugf_loss = _js_divergence(exp_by_group, target).mean() if self.use_js \
                               else _l2_dist(exp_by_group, target).mean()

                    loss = loss_main + self.lambda_ugf * ugf_loss
                else:
                    loss = loss_main
                # -------------------------- 评分偏差约束：损失 --------------------------
                if self.enable_rating_bias_constraint:
                    # 1. 获取当前批次的原始user_id（从本地索引映射回原始ID）
                    batch_user_ids = [self.idx2user[int(u_idx)] for u_idx in u_b.detach().cpu().tolist()]
                    # 2. 计算每个用户的约束上限：alpha*(用户偏差 + beta)
                    constraint_upper = torch.tensor(
                        [self.alpha_bias * (self.user_rating_bias.get(uid, 0.0) + self.beta_bias) 
                         for uid in batch_user_ids],
                        dtype=torch.float32,
                        device=self.device
                    )  # [B]
                    # 3. 计算预测评分与全局平均的差距（绝对值）
                    pred_scores = self.net.score(u_b, i_pos_b)  # 正例预测评分 [B]
                    pred_diff = torch.abs(pred_scores - self.global_avg_rating)  # [B]
                    # 4. Hinge损失：当差距超过约束上限时，施加惩罚
                    bias_loss = torch.max(torch.tensor(0.0, device=self.device), pred_diff - constraint_upper).mean()
                    # 5. 加入总损失（lambda_bias控制约束强度，从config读取）
                    lambda_bias = float(ugf.get("lambda_bias", 0.1))
                    loss = loss + lambda_bias * bias_loss
                # ------------------------------------------------------------------------

                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += float(loss.detach().cpu())

            print(f"[MF] Epoch {epoch}/{epochs} loss={epoch_loss/steps_per_epoch:.4f}")

    def _popular_fallback(self, k: int = 20):
        # 用 item embedding 范数做一个简单的热门榜
        with torch.no_grad():
            vec = self.net.item_emb.weight.detach().cpu().numpy()
            scores = (vec ** 2).sum(axis=1) ** 0.5
            topk = np.argsort(-scores)[:k]
            return [(self.idx2item[i], float(scores[i])) for i in topk]

    def recommend_topk(self, user_id: int, k: int = 20):
        if self.net is None:
            raise RuntimeError("Model not trained.")
        # 冷启动用户：回退
        if user_id not in self.user2idx:
            return self._popular_fallback(k=k)

        self.net.eval()
        with torch.no_grad():
            u_local = self.user2idx[user_id]
            u_t = torch.tensor([u_local], dtype=torch.long, device=self.device)  # [1]
            # 对全部 item 打分
            I = torch.arange(len(self.item2idx), dtype=torch.long, device=self.device)  # [N]
            scores = self.net.score(u_t, I.unsqueeze(0)).squeeze(0)  # [N]

            # 过滤训练中已交互
            seen = self.user_pos.get(user_id, set())
            if seen:
                seen_local = [self.item2idx[i] for i in seen if i in self.item2idx]
                scores[torch.tensor(seen_local, dtype=torch.long, device=self.device)] = -1e9

            top_idx = torch.topk(scores, k=min(k, scores.shape[0]))[1].tolist()
            return [(self.idx2item[i], float(scores[i].item())) for i in top_idx]


# =============== LightGCN（简化版） ===============

class LightGCNRecommender(MFRecommender):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, group_info: Optional[Dict[str, Any]] = None):
        super().fit(train_df, val_df, group_info)
