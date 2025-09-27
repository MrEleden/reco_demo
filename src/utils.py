import numpy as np
import pandas as pd

def minmax_norm(x, eps=1e-8):
    x = np.asarray(x, dtype=float)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn + eps)

def to_display_df(df, score_col="score", extra_cols=None):
    df = df.copy()
    df = df.reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df)+1))
    cols = ["rank", "movieId", "title", score_col]
    if extra_cols:
        for c in extra_cols:
            if c not in cols and c in df.columns:
                cols.append(c)
    df = df[cols].rename(columns={score_col: "score"})
    df["score"] = df["score"].round(4)
    if "val_rating" in df.columns:
        df["val_rating"] = pd.to_numeric(df["val_rating"], errors="coerce").round(2)
    return df
