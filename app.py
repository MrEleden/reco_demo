import os
import pandas as pd
import numpy as np
import gradio as gr

from src.datasets import load_dataset_with_splits, build_id_mappings, get_seen_dict
from src.models.collab import MFRecommender, CollabTrainer, recommend_collab
from src.models.content import ContentIndexer, recommend_content
from src.utils import minmax_norm, to_display_df

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(ASSETS_DIR, exist_ok=True)

warning_banner = None

# ---- Load dataset with train/val split ----
movies_df, ratings_train, ratings_val = load_dataset_with_splits(DATA_DIR)
warning_banner = getattr(movies_df, "__warning__", None)

# ID mappings from TRAIN set
uid2idx, idx2uid, iid2idx, idx2iid = build_id_mappings(ratings_train, movies_df)
seen_train = get_seen_dict(ratings_train)

# ---- COLLAB: init + quick train on TRAIN ----
collab = MFRecommender(num_users=len(uid2idx), num_items=len(iid2idx), k=32)
trainer = CollabTrainer(
    model=collab,
    ratings_df=ratings_train,
    uid2idx=uid2idx,
    iid2idx=iid2idx,
    min_rating=4.0,
    negative_per_positive=1,
    max_users=20000,
    max_items=50000,
    batch_size=4096,
    lr=0.05,
)

if len(ratings_train) > 0 and len(uid2idx) > 0 and len(iid2idx) > 0:
    trainer.train(epochs=3, verbose=False)

# ---- CONTENT: build TF-IDF index ----
content = ContentIndexer(movies_df)
content.build()

def list_users():
    if "userId" not in ratings_train.columns or ratings_train.empty:
        return []
    users = sorted(pd.unique(ratings_train["userId"]).tolist())
    return [str(u) for u in users]

def _merge_scores(collab_df, content_df):
    c = collab_df[["movieId","score"]].rename(columns={"score":"collab_score"})
    t = content_df[["movieId","title","score"]].rename(columns={"score":"content_score"})
    merged = pd.merge(c, t, on="movieId", how="left")
    merged["content_score"] = merged["content_score"].fillna(0.0)

    # Ensure titles by joining with movies_df
    mv = movies_df[["movieId","title"]].rename(columns={"title":"_mv_title"})
    merged = merged.merge(mv, on="movieId", how="left")
    if "title" not in merged:
        merged["title"] = merged["_mv_title"]
    else:
        merged["title"] = merged["title"].fillna(merged["_mv_title"])
    if "_mv_title" in merged.columns:
        merged = merged.drop(columns=["_mv_title"])

    return merged

def _history_tables(user_id, min_positive=4.0):
    trn = ratings_train[ratings_train["userId"] == user_id].copy()
    val = ratings_val[ratings_val["userId"] == user_id].copy()

    def _prep(d):
        if d.empty:
            return pd.DataFrame(columns=["rank","movieId","title","rating"])
        d = d.merge(movies_df[["movieId","title"]], on="movieId", how="left")
        d["title"] = d["title"].fillna("")
        d = d.sort_values(["rating","movieId"], ascending=[False, True]).copy()
        d = d[["movieId","title","rating"]]
        d.insert(0, "rank", np.arange(1, len(d)+1))
        d["rating"] = d["rating"].round(2)
        return d

    liked_train = trn[trn["rating"] >= float(min_positive)]
    return _prep(liked_train), _prep(trn), _prep(val)

def _annotate_with_val(df, user_id):
    val_u = ratings_val[ratings_val["userId"] == user_id][["movieId","rating"]]
    if val_u.empty:
        df["in_val"] = False
        df["val_rating"] = np.nan
        return df
    val_map = dict(zip(val_u["movieId"].tolist(), val_u["rating"].tolist()))
    df["val_rating"] = df["movieId"].map(val_map)
    df["in_val"] = df["val_rating"].notna()
    return df

def recommend(user_id_str, alpha, top_k):
    try:
        user_id = int(user_id_str)
    except:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # collab scores for all items
    collab_scores = recommend_collab(
        model=collab,
        user_id=user_id,
        uid2idx=uid2idx,
        idx2iid=idx2iid,
        seen=seen_train,
        return_scores=True,
    )
    # content scores (use TRAIN interactions to build profile)
    content_scores = recommend_content(
        content=content,
        user_id=user_id,
        ratings_df=ratings_train,
        seen=seen_train,
        min_positive=4.0,
    )

    # combine + blend
    merged = _merge_scores(collab_scores, content_scores)
    c_norm = minmax_norm(merged["collab_score"].values)
    t_norm = minmax_norm(merged["content_score"].values)
    merged["hybrid_score"] = alpha * c_norm + (1 - alpha) * t_norm

    # mask only TRAIN seen (allow validation items to appear)
    seen_items = seen_train.get(user_id, set())
    if seen_items:
        merged = merged[~merged["movieId"].isin(seen_items)]

    # annotate with validation info
    merged = _annotate_with_val(merged, user_id)

    merged = merged.sort_values("hybrid_score", ascending=False).head(int(top_k))
    recs_df = to_display_df(merged, score_col="hybrid_score", extra_cols=["in_val","val_rating"])

    # History tables
    liked_train_df, train_df, val_df = _history_tables(user_id)
    return recs_df, liked_train_df, train_df, val_df

with gr.Blocks(title="Movie Recommender (Hybrid/Collab/Content)") as demo:
    gr.Markdown("# 🎬 Movie Recommender\nPick a user and get movies they haven't watched yet.")
    if warning_banner:
        gr.Markdown(f"> ⚠️ **Notice:** {warning_banner}")
    with gr.Row():
        choices = list_users()
        user_dd = gr.Dropdown(choices=choices, label="User", value=choices[0] if choices else None)
        alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Hybrid weight α (collab vs content)")
        topk = gr.Slider(1, 50, value=10, step=1, label="Top-K")
    btn = gr.Button("Recommend")

    with gr.Tabs():
        with gr.Tab("Recommendations"):
            out_recs = gr.Dataframe(headers=["rank","movieId","title","score","in_val","val_rating"], interactive=False)
        with gr.Tab("User history (Train)"):
            gr.Markdown("**Liked in Train (rating ≥ 4)**")
            out_liked_train = gr.Dataframe(headers=["rank","movieId","title","rating"], interactive=False)
            gr.Markdown("**All Train**")
            out_train = gr.Dataframe(headers=["rank","movieId","title","rating"], interactive=False)
        with gr.Tab("User history (Validation)"):
            out_val = gr.Dataframe(headers=["rank","movieId","title","rating"], interactive=False)

    btn.click(fn=recommend, inputs=[user_dd, alpha, topk], outputs=[out_recs, out_liked_train, out_train, out_val])

if __name__ == "__main__":
    demo.launch()
