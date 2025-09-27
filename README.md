# 🎬 Movie Recommender (Train/Val, PyTorch + Gradio)

Now supports **validation awareness**:
- Train/Val split (uses `data/ratings_val.csv` if present, otherwise per-user chronological split).
- Recs table shows whether a movie is **in the user's validation set** and the **held-out rating**.
- History tabs for **Train** and **Validation**.

## Expected files

```
data/movies.csv      # movieId,title,genres
data/ratings.csv     # userId,movieId,rating,timestamp
# optional:
data/ratings_val.csv # userId,movieId,rating,timestamp
```

If `ratings_val.csv` is missing, a per-user chronological split (~20% val) is performed.

## Run

```bash
pip install -r requirements.txt
python app.py
```
