import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..datasets import load_dataset_with_splits, build_id_mappings

class MFRecommender(nn.Module):
    def __init__(self, num_users, num_items, k=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, k)
        self.item_emb = nn.Embedding(num_items, k)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, u_idxs, i_idxs):
        u = self.user_emb(u_idxs)
        i = self.item_emb(i_idxs)
        dot = (u * i).sum(dim=1, keepdim=True)
        b = self.user_bias(u_idxs) + self.item_bias(i_idxs)
        return (dot + b).squeeze(1)  # logits

class ImplicitPairs(Dataset):
    def __init__(self, ratings_df, uid2idx, iid2idx, min_rating=4.0, negative_per_positive=1, max_users=None, max_items=None):
        df = ratings_df.copy()
        if max_users is not None and len(df["userId"].unique()) > max_users:
            keep_users = set(df["userId"].value_counts().head(max_users).index.tolist())
            df = df[df["userId"].isin(keep_users)]
        if max_items is not None and len(df["movieId"].unique()) > max_items:
            keep_items = set(df["movieId"].value_counts().head(max_items).index.tolist())
            df = df[df["movieId"].isin(keep_items)]

        pos = df[df["rating"] >= min_rating][["userId","movieId"]].reset_index(drop=True)
        self.user_pos = {}
        for u, g in pos.groupby("userId"):
            self.user_pos[u] = set(g["movieId"].tolist())
        self.all_items = list(iid2idx.keys())
        self.uid2idx = uid2idx
        self.iid2idx = iid2idx
        self.negative_per_positive = negative_per_positive
        self.pos_pairs = pos.values.tolist()

    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, idx):
        u, i = self.pos_pairs[idx]
        return int(u), int(i)

    def collate_fn(self, batch):
        users = [u for (u, _) in batch]
        pos_items = [i for (_, i) in batch]
        neg_users, neg_items = [], []
        import numpy as np
        for u in users:
            seen = self.user_pos.get(u, set())
            for _ in range(self.negative_per_positive):
                while True:
                    j = np.random.choice(self.all_items)
                    if j not in seen:
                        neg_users.append(u); neg_items.append(int(j))
                        break
        import torch
        u_idx = torch.tensor([self.uid2idx[u] for u in users + neg_users], dtype=torch.long)
        i_idx = torch.tensor([self.iid2idx[i] for i in pos_items + neg_items], dtype=torch.long)
        y = torch.tensor([1]*len(users) + [0]*len(neg_users), dtype=torch.float32)
        return u_idx, i_idx, y

class CollabTrainer:
    def __init__(self, model, ratings_df, uid2idx, iid2idx, min_rating=4.0, negative_per_positive=1, max_users=None, max_items=None, batch_size=4096, lr=0.05, device=None):
        self.model = model
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.dataset = ImplicitPairs(ratings_df, uid2idx, iid2idx, min_rating, negative_per_positive, max_users, max_items)
        from torch.utils.data import DataLoader
        self.dl = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, collate_fn=self.dataset.collate_fn)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        from torch import nn
        self.loss_fn = nn.BCEWithLogitsLoss()

    def train(self, epochs=3, verbose=True):
        self.model.train()
        total_seen = 0
        for e in range(epochs):
            total = 0.0
            for u_idx, i_idx, y in self.dl:
                u_idx = u_idx.to(self.device); i_idx = i_idx.to(self.device); y = y.to(self.device)
                self.opt.zero_grad()
                logits = self.model(u_idx, i_idx)
                loss = self.loss_fn(logits, y)
                loss.backward()
                self.opt.step()
                total += loss.item() * len(y)
                total_seen += len(y)
            if verbose:
                print(f"epoch {e+1}/{epochs} - loss {total/max(1,len(self.dataset)):.4f}")

@torch.no_grad()
def recommend_collab(model, user_id, uid2idx, idx2iid, seen, top_k=None, return_scores=False):
    model.eval()
    device = next(model.parameters()).device
    if user_id not in uid2idx:
        return pd.DataFrame(columns=["movieId","title","score"])
    import torch
    uidx = torch.tensor([uid2idx[user_id]], dtype=torch.long, device=device)
    n_items = len(idx2iid)
    i_idx = torch.arange(n_items, dtype=torch.long, device=device)
    u_vec = uidx.repeat(n_items)
    logits = model(u_vec, i_idx).cpu().numpy()
    movie_ids = [idx2iid[i] for i in range(n_items)]
    df = pd.DataFrame({"movieId": movie_ids, "score": logits})
    df = df.sort_values("score", ascending=False)
    if top_k is not None:
        df = df.head(int(top_k))
    df["title"] = ""
    return df

def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--neg", type=int, default=1)
    args = parser.parse_args()

    movies, ratings_train, ratings_val = load_dataset_with_splits(args.data_dir)
    uid2idx, idx2uid, iid2idx, idx2iid = build_id_mappings(ratings_train, movies)
    model = MFRecommender(len(uid2idx), len(iid2idx), k=args.k)
    trainer = CollabTrainer(model, ratings_train, uid2idx, iid2idx, negative_per_positive=args.neg)
    trainer.train(epochs=args.epochs, verbose=True)

if __name__ == "__main__":
    main_cli()
