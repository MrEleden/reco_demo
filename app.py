# Streamlit: Movie Likes Viewer (CSV-only, max 10 per user)
# --------------------------------------------------------
# Run:
#   pip install streamlit pandas
#   streamlit run app.py

import os, math
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Movie Likes Viewer", page_icon="🎬", layout="centered")


def find_csv_dir() -> str | None:
    env_dir = os.getenv("MOVIELENS_DIR")
    candidates = [
        env_dir,
        os.path.join("ml-latest-small", "ml-latest-small"),  # ton layout
        os.path.join("ml-latest-small"),
        os.getcwd(),
    ]
    for p in candidates:
        if p and all(os.path.exists(os.path.join(p, f)) for f in ("movies.csv", "ratings.csv", "links.csv")):
            return p
    return None


CSV_DIR = find_csv_dir()
if CSV_DIR is None:
    st.error(
        "Impossible de trouver les CSV. Place-les dans ml-latest-small/ml-latest-small/ "
        "ou définis MOVIELENS_DIR vers le dossier contenant movies.csv/ratings.csv/links.csv."
    )
    st.stop()


@st.cache_data(show_spinner=True)
def load_data(csv_dir: str):
    movies = pd.read_csv(os.path.join(csv_dir, "movies.csv"))
    ratings = pd.read_csv(os.path.join(csv_dir, "ratings.csv"))
    links = pd.read_csv(os.path.join(csv_dir, "links.csv"))
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")[0].astype("float")
    imdb_map = dict(zip(links["movieId"], links["imdbId"]))
    return movies, ratings, imdb_map


movies, ratings, imdb_map = load_data(CSV_DIR)

st.title("🎬 Movie Likes Viewer (max 10)")
st.caption("Sélectionne un userId et un seuil : on affiche au **maximum 10** films qu'il/elle a likés.")

user_ids = sorted(ratings["userId"].unique().tolist())
if not user_ids:
    st.warning("Aucun userId dans ratings.csv.")
    st.stop()

c1, c2 = st.columns([2, 1])
with c1:
    user_id = st.selectbox("User", options=user_ids, index=0)
with c2:
    min_thr, max_thr = st.slider("Plage de notes", 0.5, 5.0, (4.0, 5.0), 0.5)

user_ratings = ratings[ratings["userId"] == user_id].copy()
user_ratings["in_range"] = (user_ratings["rating"] >= min_thr) & (user_ratings["rating"] <= max_thr)
liked = user_ratings[user_ratings["in_range"]].merge(movies, on="movieId", how="left")

if "timestamp" in liked.columns:
    liked["rated_at"] = pd.to_datetime(liked["timestamp"], unit="s")


def imdb_url(imdb_id) -> str | None:
    try:
        if math.isnan(imdb_id):
            return None
    except Exception:
        pass
    try:
        return f"https://www.imdb.com/title/tt{int(imdb_id):07d}/"
    except Exception:
        return None


liked["imdb"] = liked["movieId"].map(imdb_map).map(imdb_url)

cols = ["movieId", "title", "year", "rating", "rated_at", "imdb"]
liked_view = (
    liked.reindex(columns=[c for c in cols if c in liked.columns])
    .sort_values(by=["rating", "year", "title"], ascending=[False, False, True])
    .head(10)  # <<<<<< cap à 10 films max
)

st.markdown("---")
st.subheader("👍 Films likés")
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Total notes (user)", int(user_ratings.shape[0]))
with m2:
    st.metric("Likes (≥ seuil)", int((user_ratings["liked"]).sum()))
with m3:
    st.metric("Affichés", int(liked_view.shape[0]))

st.dataframe(liked_view.reset_index(drop=True), use_container_width=True, hide_index=True)

st.download_button(
    label="⬇️ Exporter (CSV, max 10)",
    data=liked_view.to_csv(index=False),
    file_name=f"user_{user_id}_liked_movies_top10.csv",
    mime="text/csv",
)
