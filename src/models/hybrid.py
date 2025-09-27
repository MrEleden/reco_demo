def blend_scores(collab_scores, content_scores, alpha=0.5):
    return alpha * collab_scores + (1 - alpha) * content_scores
