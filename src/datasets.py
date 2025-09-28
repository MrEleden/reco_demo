import os
import pandas as pd


def _split_train_test_val(
    df,
    ratios=(0.8, 0.1, 0.1),  # (train, val, test)
    min_val=1,
    min_test=1,
):
    # split rating using the timestamp per user
    _, va_ratio, te_ratio = ratios
    parts_tr, parts_va, parts_te = [], [], []

    for _, g in df.groupby("userId", sort=False):
        g = g.sort_values("timestamp", kind="mergesort")
        n = len(g)

        if n <= 1:
            parts_tr.append(g)
            parts_va.append(g.iloc[0:0])
            parts_te.append(g.iloc[0:0])
            continue
        if n == 2:
            parts_tr.append(g.iloc[:1])
            parts_va.append(g.iloc[0:0])
            parts_te.append(g.iloc[1:])
            continue

        n_test = max(int(n * te_ratio), min_test if te_ratio > 0 else 0)
        n_val = max(int(n * va_ratio), min_val if va_ratio > 0 else 0)

        n_test = max(int(n * te_ratio), 1)
        n_val = max(int(n * va_ratio), min_val)
        train_end = n - n_val - n_test
        val_end = n - n_test
        parts_tr.append(g.iloc[:train_end])
        parts_va.append(g.iloc[train_end:val_end])
        parts_te.append(g.iloc[val_end:])

    train_df = pd.concat(parts_tr).reset_index(drop=True)
    val_df = pd.concat(parts_va).reset_index(drop=True)
    test_df = pd.concat(parts_te).reset_index(drop=True)
    return train_df, val_df, test_df


def _std_movies_hard(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["movieId"] = df["movieId"].astype(int)
    df["title"] = df["title"].fillna("")
    df["genres"] = df["genres"].fillna("")
    return df


def _std_ratings_hard(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df["userId"] = df["userId"].astype(int)
    df["movieId"] = df["movieId"].astype(int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    return df


def load_dataset_with_splits(data_dir):
    # Load movies.csv
    movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
    movies = _std_movies_hard(movies)
    ratings = _std_ratings_hard(ratings)
    train_df, val_df, test_df = _split_train_test_val(ratings, ratios=(0.8, 0.1, 0.1), min_val=1, min_test=1)
    return movies, train_df, val_df, test_df


def build_id_mappings(ratings_df, movies_df):
    user_ids = sorted(pd.unique(ratings_df["userId"]).tolist())
    item_ids = sorted(pd.unique(movies_df["movieId"]).tolist())
    uid2idx = {u: i for i, u in enumerate(user_ids)}
    idx2uid = {i: u for u, i in uid2idx.items()}
    iid2idx = {m: i for i, m in enumerate(item_ids)}
    idx2iid = {i: m for m, i in iid2idx.items()}
    return uid2idx, idx2uid, iid2idx, idx2iid


def get_seen_dict(ratings_df):
    seen = {}
    if not len(ratings_df):
        return seen
    for u, g in ratings_df.groupby("userId"):
        seen[int(u)] = set(int(x) for x in g["movieId"].tolist())
    return seen
