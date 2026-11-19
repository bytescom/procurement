"""Optional semantic taxonomy scoring using only an already installed local model."""

import os
from pathlib import Path


def predict_semantic(texts: list[str], taxonomy_paths: list[tuple[str, ...]], *, model_path: str | None = None):
    """Return cosine-ranked taxonomy paths, or ``None`` when no local model exists.

    No model is downloaded. Configure ``SENTENCE_TRANSFORMER_MODEL`` to a local
    model directory and install sentence-transformers to enable this stage.
    """
    model_path = model_path or os.environ.get("SENTENCE_TRANSFORMER_MODEL")
    if not model_path or not Path(model_path).exists():
        return None
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return None
    model = SentenceTransformer(model_path, local_files_only=True)
    taxonomy_texts = [" > ".join(path) for path in taxonomy_paths]
    item_embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    taxonomy_embeddings = model.encode(taxonomy_texts, normalize_embeddings=True, show_progress_bar=False)
    scores = cosine_similarity(item_embeddings, taxonomy_embeddings)
    return [{"path": taxonomy_paths[int(i)], "confidence": float(scores[row, i])}
            for row, i in enumerate(scores.argmax(axis=1))]
