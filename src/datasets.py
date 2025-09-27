import os
import pandas as pd
import numpy as np

RATINGS_ALIASES = {
    "userId": {"user", "userid", "user_id", "u", "uid"},
    "movieId": {"movie", "movieid", "item_id", "itemid", "mid", "iid"},
    "rating": {"rating", "score", "stars", "preference"},
    "timestamp": {"timestamp", "ts", "time", "date"},
}

MOVIES_ALIASES = {
    "movieId": {"movie", "movieid", "item_id", "itemid", "mid", "iid"},
    "title": {"title", "name"},
    "genres": {"genres", "genre", "tags", "category", "categories"},
}


def _synthetic():
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5, 6],
            "title": ["The Matrix", "Toy Story", "Inception", "Finding Nemo", "The Godfather", "Interstellar"],
            "genres": [
                "Action|Sci-Fi",
                "Animation|Comedy",
                "Action|Sci-Fi",
                "Animation|Family",
                "Crime|Drama",
                "Sci-Fi|Drama",
            ],
        }
    )
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 3, 3, 3, 3],
            "movieId": [1, 2, 5, 2, 3, 1, 3, 4, 6],
            "rating": [5, 4, 2, 5, 4, 4, 5, 3, 5],
            "timestamp": [1, 2, 3, 1, 2, 1, 2, 3, 4],
        }
    )
    return movies, ratings


def _standardize_columns(df, mapping):
    cols = {c.lower(): c for c in df.columns}
    out = {}
    for std, aliases in mapping.items():
        found = None
        for a in aliases:
            if a.lower() in cols:
                found = cols[a.lower()]
                break
        if std in df.columns:
            found = std
        if found is not None:
            out[std] = found
    return df.rename(columns=out)


def _has_min_columns(df, required):
    return all(c in df.columns for c in required)


def _load_movies(data_dir):
    movies_p = os.path.join(data_dir, "movies.csv")
    if os.path.exists(movies_p):
        movies = pd.read_csv(movies_p)
        movies = _standardize_columns(movies, MOVIES_ALIASES)
        if not _has_min_columns(movies, ["movieId", "title"]):
            raise ValueError("movies.csv must have columns that map to movieId,title")
        movies["movieId"] = movies["movieId"].astype(int)
        movies["title"] = movies["title"].fillna("")
        if "genres" in movies.columns:
            movies["genres"] = movies["genres"].fillna("")
        else:
            movies["genres"] = ""
        return movies
    else:
        m, _ = _synthetic()
        return m


def _load_ratings(path):
    df = pd.read_csv(path)
    df = _standardize_columns(df, RATINGS_ALIASES)
    if not _has_min_columns(df, ["userId", "movieId", "rating"]):
        raise ValueError(f"{os.path.basename(path)} must map to userId,movieId,rating")
    df["userId"] = df["userId"].astype(int)
    df["movieId"] = df["movieId"].astype(int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
    if "timestamp" not in df.columns:
        df["timestamp"] = np.arange(len(df))  # deterministic filler
    return df


def _split_train_val(ratings, val_ratio=0.2, min_val=1):
    parts = []
    for u, g in ratings.groupby("userId", sort=False):
        g = g.sort_values("timestamp")
        n = len(g)
        if n <= 1:
            parts.append((g, g.iloc[0:0]))
            continue
        n_val = max(min_val, int(round(n * val_ratio)))
        n_val = min(n_val, n - 1)
        val = g.tail(n_val).copy()
        trn = g.head(n - n_val).copy()
        parts.append((trn, val))
    trains = [p[0] for p in parts]
    vals = [p[1] for p in parts]
    train = pd.concat(trains, ignore_index=True) if len(trains) else ratings.iloc[0:0].copy()
    val = pd.concat(vals, ignore_index=True) if len(vals) else ratings.iloc[0:0].copy()
    return train, val


def load_dataset_with_splits(data_dir):
    movies = _load_movies(data_dir)
    ratings_p = os.path.join(data_dir, "ratings.csv")
    ratings_val_p = os.path.join(data_dir, "ratings_val.csv")
    if os.path.exists(ratings_p):
        try:
            ratings = _load_ratings(ratings_p)
            if os.path.exists(ratings_val_p):
                ratings_val = _load_ratings(ratings_val_p)
                # derive train by removing exact matches (userId,movieId,timestamp) in val
                key = ["userId", "movieId", "timestamp"]
                if not all(c in ratings_val.columns for c in key):
                    ratings_val["timestamp"] = -1
                merged = ratings.merge(ratings_val[key].drop_duplicates(), on=key, how="left", indicator=True)
                ratings_train = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
                return movies, ratings_train, ratings_val
            else:
                ratings_train, ratings_val = _split_train_val(ratings)
                return movies, ratings_train, ratings_val
        except Exception as e:
            m, r = _synthetic()
            tr, va = _split_train_val(r)
            m.attrs["__warning__"] = f"Fell back to synthetic dataset due to: {e}"
            return m, tr, va
    else:
        m, r = _synthetic()
        tr, va = _split_train_val(r)
        return m, tr, va


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
