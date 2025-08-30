import torch
import torch.nn as nn
import torch.nn.functional as F

class MF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64, in_method=None, reg_weight=1e-4):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self.in_method = in_method
        self.reg_weight = reg_weight
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        user_vecs = self.user_emb(user_ids)
        item_vecs = self.item_emb(item_ids)
        scores = (user_vecs * item_vecs).sum(dim=1)
        return scores

    def loss(self, user_ids, pos_items, neg_items=None):
        if self.in_method == "negative_sampling":
            pos_scores = self.forward(user_ids, pos_items)
            neg_scores = self.forward(user_ids, neg_items)
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        else:
            preds = self.forward(user_ids, pos_items)
            loss = F.mse_loss(preds, torch.ones_like(preds))
        
        # Add regularization if requested
        if self.in_method == "regularization":
            reg = (self.user_emb(user_ids).norm(2).pow(2) +
                   self.item_emb(pos_items).norm(2).pow(2)).mean()
            loss += self.reg_weight * reg
        return loss


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64, n_layers=3, in_method=None, reg_weight=1e-4):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.in_method = in_method
        self.reg_weight = reg_weight

        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def propagate(self, adj):
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(adj, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1).mean(dim=1)
        user_embs, item_embs = torch.split(embs, [self.num_users, self.num_items])
        return user_embs, item_embs

    def forward(self, user_ids, item_ids, adj):
        user_embs, item_embs = self.propagate(adj)
        u = user_embs[user_ids]
        i = item_embs[item_ids]
        scores = (u * i).sum(dim=1)
        return scores

    def loss(self, user_ids, pos_items, adj, neg_items=None):
        if self.in_method == "negative_sampling":
            pos_scores = self.forward(user_ids, pos_items, adj)
            neg_scores = self.forward(user_ids, neg_items, adj)
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        else:
            preds = self.forward(user_ids, pos_items, adj)
            loss = F.mse_loss(preds, torch.ones_like(preds))

        if self.in_method == "regularization":
            reg = (self.user_emb(user_ids).norm(2).pow(2) +
                   self.item_emb(pos_items).norm(2).pow(2)).mean()
            loss += self.reg_weight * reg
        return loss