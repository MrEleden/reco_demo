# Movie Recommender System - AI Assistant Guide

## Architecture Overview

This is a hybrid movie recommendation system built with PyTorch and Gradio that combines collaborative filtering (matrix factorization) and content-based filtering. The system is designed around **train/validation/test splits** with validation awareness in recommendations.

### Key Components

- **`app.py`**: Gradio web interface orchestrating the entire system
- **`src/datasets.py`**: Handles MovieLens-style data loading and chronological train/val/test splitting  
- **`src/models/collab.py`**: PyTorch collaborative filtering using matrix factorization with implicit feedback
- **`src/models/content.py`**: TF-IDF content-based filtering on movie titles/genres
- **`src/utils.py`**: Score normalization and display formatting utilities

## Critical Data Flow

1. **Dataset Loading**: `load_dataset_with_splits()` performs per-user chronological splits (80/10/10) using timestamps
2. **ID Mapping**: All models use integer indices internally via `build_id_mappings()` from the TRAIN set only  
3. **Training**: Collaborative model trains on positive ratings (≥4.0) with negative sampling
4. **Hybrid Scoring**: Recommendations blend normalized collaborative and content scores with user-defined alpha
5. **Validation Awareness**: UI shows which recommended movies appear in user's validation/test sets with actual ratings

## Development Patterns

### Model Training Convention
- Models train only on the training split (`ratings_train`) 
- Use `uid2idx/iid2idx` mappings built from training data for all ID conversions
- Collaborative model uses implicit feedback: ratings ≥4.0 as positive, negative sampling for others

### Recommendation Pipeline  
```python
# Standard pattern for generating recommendations:
collab_scores = recommend_collab(model, user_id, uid2idx, idx2iid, seen_train)
content_scores = recommend_content(content, user_id, ratings_train, seen_train)
merged = _merge_scores(collab_scores, content_scores)  # Handle missing titles
hybrid_score = alpha * normalized_collab + (1-alpha) * normalized_content
```

### Data Assumptions
- Input: MovieLens format (`userId,movieId,rating,timestamp` and `movieId,title,genres`)
- Missing `ratings_val.csv` triggers automatic chronological splitting
- All dataframes standardized via `_std_movies_hard()` and `_std_ratings_hard()`

## Key Implementation Details

- **Implicit Feedback**: `ImplicitPairs` dataset uses BCEWithLogitsLoss, not MSE regression
- **Memory Management**: `CollabTrainer` supports `max_users`/`max_items` for large datasets
- **Score Normalization**: Always use `minmax_norm()` before blending collab/content scores
- **Seen Items**: Only TRAIN interactions are filtered out - validation/test items can appear in recommendations

## Running the System

```bash
pip install -r requirements.txt
python app.py  # Launches Gradio interface on http://localhost:7860
```

For CLI training: `python -m src.models.collab --data_dir data --epochs 5 --k 64`

## Common Pitfalls

- Don't use validation/test data for ID mappings - only training split  
- Remember to call `.eval()` on PyTorch models before inference
- Content filtering requires movies to exist in the movies DataFrame for TF-IDF
- Alpha=1.0 is pure collaborative, alpha=0.0 is pure content-based