import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentIndexer:
    def __init__(self, movies_df):
        self.movies = movies_df.copy()
        self.movies["__text__"] = (self.movies["title"].fillna("") + " " + self.movies["genres"].fillna("")).str.lower()
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        self.movie_vectors = None  # sparse matrix

    def build(self):
        self.movie_vectors = self.vectorizer.fit_transform(self.movies["__text__"].values)

    def user_profile(self, liked_movie_ids):
        if len(liked_movie_ids) == 0:
            return None
        idx = self.movies[self.movies["movieId"].isin(liked_movie_ids)].index.values
        if len(idx) == 0:
            return None
        vecs = self.movie_vectors[idx]
        prof = vecs.mean(axis=0)
        prof = np.asarray(prof).reshape(1, -1)  # ensure ndarray row
        return prof

def recommend_content(content, user_id, ratings_df, seen, min_positive=4.0, top_k=None):
    liked = ratings_df[(ratings_df["userId"]==user_id) & (ratings_df["rating"]>=min_positive)]["movieId"].tolist()
    prof = content.user_profile(liked)
    if prof is None:
        scores = np.zeros(content.movie_vectors.shape[0])
    else:
        scores = cosine_similarity(prof, content.movie_vectors).ravel()

    df = pd.DataFrame({
        "movieId": content.movies["movieId"].values,
        "title": content.movies["title"].values,
        "score": scores
    })
    df = df.sort_values("score", ascending=False)
    if top_k is not None:
        df = df.head(int(top_k))
    return df
