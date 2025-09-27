# Streamlit: Movie Likes Viewer (CSV-only)
# --------------------------------------
# Shows, for a selected user, the movies they "liked" based on a rating threshold.
# Uses local MovieLens CSVs only (no downloads). Point to your CSV folder if needed.
#
# Project layout example:
#   app.py
#   ml-latest-small/                 # or
#     ml-latest-small/
#       movies.csv
#       ratings.csv
#       links.csv
#
# Run:
#   pip install streamlit pandas
#   streamlit run app.py

import os
import math
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Movie Likes Viewer", page_icon="🎬", layout="centered")

# -----------------------------
# Locate local CSVs
# -----------------------------


def find_csv_dir() -> str | None:
    """Try a few common locations; return the directory containing the CSVs or None."""
    env_dir = os.getenv("MOVIELENS_DIR")
    candidates = [
        env_dir,
        os.path.join("ml-latest-small", "ml-latest-small"),
        os.path.join("ml-latest-small"),
        os.getcwd(),
    ]
    for p in candidates:
        if not p:
            continue
        if all(os.path.exists(os.path.join(p, f)) for f in ("movies.csv", "ratings.csv", "links.csv")):
            return p
    return None


CSV_DIR = find_csv_dir()
if CSV_DIR is None:
    st.error(
        "CSV folder not found. Put your files under `ml-latest-small/ml-latest-small` "
        "or set the env var MOVIELENS_DIR to the folder containing movies.csv, ratings.csv, links.csv."
    )
    st.stop()


# -----------------------------
# Load data (cached)
# -----------------------------
@st.cache_data(show_spinner=True)
def load_data(csv_dir: str):
    movies = pd.read_csv(os.path.join(csv_dir, "movies.csv"))
    ratings = pd.read_csv(os.path.join(csv_dir, "ratings.csv"))
    links = pd.read_csv(os.path.join(csv_dir, "links.csv"))

    # Parse year from title like "Toy Story (1995)"
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")[0].astype("float")

    # Map imdbId for quick URL building
    imdb_map = dict(zip(links["movieId"], links["imdbId"]))

    return movies, ratings, imdb_map


movies, ratings, imdb_map = load_data(CSV_DIR)

# -----------------------------
# UI Controls
# -----------------------------
st.title("🎬 Movie Likes Viewer")
st.caption("Pick a user, choose a threshold, and list their liked movies from local CSVs.")

user_ids = sorted(ratings["userId"].unique().tolist())
if not user_ids:
    st.warning("No users found in ratings.csv.")
    st.stop()

colA, colB = st.columns([2, 1])
with colA:
    user_id = st.selectbox("User", options=user_ids, index=0)
with colB:
    threshold = st.slider("Like threshold", min_value=0.5, max_value=5.0, value=4.0, step=0.5)

# Optional: limit rows
limit = st.number_input("Max rows to show", min_value=5, max_value=500, value=100, step=5)

# -----------------------------
# Compute liked movies for the user
# -----------------------------
user_ratings = ratings[ratings["userId"] == user_id].copy()
user_ratings["liked"] = user_ratings["rating"] >= threshold
liked = user_ratings[user_ratings["liked"]]

# Join with movies
liked = liked.merge(movies, on="movieId", how="left")

# Timestamp → datetime (MovieLens uses Unix seconds)
if "timestamp" in liked.columns:
    liked["rated_at"] = pd.to_datetime(liked["timestamp"], unit="s")


# IMDb URL
def to_imdb_url(imdb_id) -> str | None:
    try:
        if math.isnan(imdb_id):
            return None
    except Exception:
        pass
    try:
        return f"https://www.imdb.com/title/tt{int(imdb_id):07d}/"
    except Exception:
        return None


liked["imdb"] = liked["movieId"].map(imdb_map).map(to_imdb_url)

# Select display columns
cols = [
    "movieId",
    "title",
    "year",
    "rating",
    "rated_at",
    "imdb",
]
liked_view = liked.reindex(columns=[c for c in cols if c in liked.columns]).sort_values(
    by=["rating", "year", "title"], ascending=[False, False, True]
)

st.markdown("---")
st.subheader("👍 Liked movies")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total ratings", int(user_ratings.shape[0]))
with c2:
    st.metric("Liked (≥ threshold)", int(liked_view.shape[0]))
with c3:
    st.metric("Threshold", f"{threshold:.1f}")

st.dataframe(liked_view.head(int(limit)).reset_index(drop=True), use_container_width=True, hide_index=True)

st.download_button(
    label="⬇️ Download liked list (CSV)",
    data=liked_view.to_csv(index=False),
    file_name=f"user_{user_id}_liked_movies.csv",
    mime="text/csv",
)

st.caption(
    "Tip: Change the threshold to 3.5 or 4.5 to see stricter/looser likes. "
    "This app uses only your local CSVs—no external calls."
)
