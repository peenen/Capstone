import numpy as np
import pandas as pd

class MFModel:
    def __init__(self, config):
        self.latent_dim = config.get("latent_dim", 64)
        self.lr = config.get("learning_rate", 0.01)
        self.epochs = config.get("epochs", 20)
        self.batch_size = config.get("batch_size", 256)
        self.in_method = config.get("in_method", None)
        self.user_factors = None
        self.item_factors = None
        self.user_map = {}
        self.item_map = {}

    def fit(self, train_df, val_df=None):
        users = train_df['user_id'].unique()
        items = train_df['item_id'].unique()
        self.user_map = {u:i for i,u in enumerate(users)}
        self.item_map = {i:i_ for i_,i in enumerate(items)}
        n_users = len(users)
        n_items = len(items)
        self.user_factors = np.random.normal(0,0.1,(n_users,self.latent_dim))
        self.item_factors = np.random.normal(0,0.1,(n_items,self.latent_dim))

        interactions = train_df[['user_id','item_id','rating']].copy()
        # negative sampling
        if self.in_method == "negative_sampling":
            all_items = set(items)
            neg_rows = []
            for u in users:
                pos_items = set(interactions[interactions['user_id']==u]['item_id'])
                neg_items = list(all_items - pos_items)
                sampled_neg = np.random.choice(neg_items, size=len(pos_items), replace=False)
                for i in sampled_neg:
                    neg_rows.append({'user_id':u,'item_id':i,'rating':0})
            interactions = pd.concat([interactions, pd.DataFrame(neg_rows)], ignore_index=True)

        # simplified SGD
        for epoch in range(self.epochs):
            for idx,row in interactions.iterrows():
                u_idx = self.user_map[row['user_id']]
                i_idx = self.item_map[row['item_id']]
                pred = self.user_factors[u_idx] @ self.item_factors[i_idx].T
                err = row['rating'] - pred
                # regularization
                reg = 0.01 if self.in_method=="regularization" else 0
                self.user_factors[u_idx] += self.lr*(err*self.item_factors[i_idx] - reg*self.user_factors[u_idx])
                self.item_factors[i_idx] += self.lr*(err*self.user_factors[u_idx] - reg*self.item_factors[i_idx])

    def predict(self, test_df):
        preds = []
        for idx,row in test_df.iterrows():
            u_idx = self.user_map.get(row['user_id'], None)
            i_idx = self.item_map.get(row['item_id'], None)
            if u_idx is not None and i_idx is not None:
                score = self.user_factors[u_idx] @ self.item_factors[i_idx].T
            else:
                score = 0.0
            preds.append(score)
        test_df['prediction'] = preds
        return test_df

class LightGCNModel:
    def __init__(self, config):
        self.latent_dim = config.get("latent_dim", 64)
        self.lr = config.get("learning_rate", 0.01)
        self.epochs = config.get("epochs", 20)
        self.batch_size = config.get("batch_size", 256)
        self.in_method = config.get("in_method", None)
        self.user_factors = None
        self.item_factors = None
        self.user_map = {}
        self.item_map = {}
        self.adj = None
        self.num_layers = 2

    def fit(self, train_df, val_df=None):
        users = train_df['user_id'].unique()
        items = train_df['item_id'].unique()
        self.user_map = {u:i for i,u in enumerate(users)}
        self.item_map = {i:i_ for i_,i in enumerate(items)}
        n_users = len(users)
        n_items = len(items)
        self.user_factors = np.random.normal(0,0.1,(n_users,self.latent_dim))
        self.item_factors = np.random.normal(0,0.1,(n_items,self.latent_dim))
        # build adjacency matrix
        adj = np.zeros((n_users+n_items, n_users+n_items))
        for idx,row in train_df.iterrows():
            u_idx = self.user_map[row['user_id']]
            i_idx = self.item_map[row['item_id']]+n_users
            adj[u_idx,i_idx]=1
            adj[i_idx,u_idx]=1
        D_inv_sqrt = np.diag(1/np.sqrt(adj.sum(axis=1)+1e-8))
        self.adj = D_inv_sqrt @ adj @ D_inv_sqrt
        # propagate embeddings
        all_emb = np.vstack([self.user_factors, self.item_factors])
        embeddings = [all_emb]
        for layer in range(self.num_layers):
            all_emb = self.adj @ all_emb
            embeddings.append(all_emb)
        final_emb = np.mean(embeddings, axis=0)
        self.user_factors = final_emb[:n_users]
        self.item_factors = final_emb[n_users:]
        # TODO: in_method can add negative sampling or regularization during fine-tuning
        # simplified: skip extra fine-tuning for brevity

    def predict(self, test_df):
        preds = []
        for idx,row in test_df.iterrows():
            u_idx = self.user_map.get(row['user_id'], None)
            i_idx = self.item_map.get(row['item_id'], None)
            if u_idx is not None and i_idx is not None:
                score = self.user_factors[u_idx] @ self.item_factors[i_idx].T
            else:
                score = 0.0
            preds.append(score)
        test_df['prediction'] = preds
        return test_df