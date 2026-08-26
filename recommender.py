"""Preference aggregation and cosine-similarity recommendation logic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EPSILON = 1e-8


class RecommendationError(ValueError):
    """Raised when ratings or embedding indexes cannot produce recommendations."""


@dataclass(frozen=True)
class EmbeddingIndex:
    """A row-aligned manifest and normalized embedding matrix."""

    manifest: pd.DataFrame
    embeddings: np.ndarray

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise RecommendationError("Embedding matrix must be two-dimensional.")
        if len(self.manifest) != self.embeddings.shape[0]:
            raise RecommendationError(
                "Manifest and embedding matrix have different row counts."
            )


def normalize_rows(embeddings: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    """Return float32 row-wise L2-normalized embeddings."""
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise RecommendationError("Embeddings must be a two-dimensional matrix.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms < epsilon):
        raise RecommendationError("At least one embedding has a near-zero norm.")
    return matrix / norms


def aggregate_preference(
    liked_embeddings: np.ndarray,
    disliked_embeddings: np.ndarray,
    lambda_negative: float = 1.0,
    epsilon: float = EPSILON,
) -> np.ndarray:
    """Compute normalize(mean(likes) - lambda * mean(dislikes))."""
    if not np.isfinite(lambda_negative) or lambda_negative < 0:
        raise RecommendationError("Lambda must be a finite, non-negative number.")

    liked = np.asarray(liked_embeddings, dtype=np.float32)
    disliked = np.asarray(disliked_embeddings, dtype=np.float32)
    if liked.ndim != 2 or disliked.ndim != 2:
        raise RecommendationError("Liked and disliked embeddings must be matrices.")
    if liked.shape[1] != disliked.shape[1]:
        raise RecommendationError("Liked and disliked embeddings must have equal width.")
    if len(liked) == 0 and len(disliked) == 0:
        raise RecommendationError("At least one rating is required.")

    liked = normalize_rows(liked) if len(liked) else liked
    disliked = normalize_rows(disliked) if len(disliked) else disliked
    dimension = liked.shape[1]
    positive_mean = liked.mean(axis=0) if len(liked) else np.zeros(dimension)
    negative_mean = disliked.mean(axis=0) if len(disliked) else np.zeros(dimension)
    preference = positive_mean - lambda_negative * negative_mean
    norm = float(np.linalg.norm(preference))
    if norm < epsilon:
        raise RecommendationError(
            "The ratings cancel each other out; change a rating or increase lambda."
        )
    return np.asarray(preference / norm, dtype=np.float32)


def choose_outfit_indices(
    manifest: pd.DataFrame,
    gender: str,
    count: int = 15,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Choose distinct, gender-matching manifest row positions."""
    if count <= 0:
        raise RecommendationError("Outfit count must be positive.")
    if "gender" not in manifest:
        raise RecommendationError("Outfit manifest has no gender column.")

    eligible = np.flatnonzero(manifest["gender"].to_numpy() == gender)
    if len(eligible) < count:
        raise RecommendationError(
            f"Only {len(eligible)} outfits are available for {gender}; {count} required."
        )
    generator = rng if rng is not None else np.random.default_rng()
    return np.asarray(generator.choice(eligible, size=count, replace=False), dtype=int)


def rank_candidates(
    preference: np.ndarray,
    index: EmbeddingIndex,
    gender: str,
    top_k: int = 10,
) -> pd.DataFrame:
    """Return the highest cosine-similarity candidates for one gender."""
    if top_k <= 0:
        raise RecommendationError("top_k must be positive.")
    if "gender" not in index.manifest:
        raise RecommendationError("Candidate manifest has no gender column.")

    query = np.asarray(preference, dtype=np.float32)
    if query.ndim != 1 or query.shape[0] != index.embeddings.shape[1]:
        raise RecommendationError("Preference vector has the wrong dimensions.")
    query_norm = float(np.linalg.norm(query))
    if query_norm < EPSILON:
        raise RecommendationError("Preference vector has a near-zero norm.")
    query = query / query_norm

    candidate_positions = np.flatnonzero(
        index.manifest["gender"].to_numpy() == gender
    )
    if not len(candidate_positions):
        raise RecommendationError(f"No recommendation candidates exist for {gender}.")

    candidate_embeddings = np.asarray(
        index.embeddings[candidate_positions], dtype=np.float32
    )
    scores = candidate_embeddings @ query
    result_count = min(top_k, len(scores))
    if result_count == len(scores):
        local_positions = np.arange(len(scores))
    else:
        local_positions = np.argpartition(scores, -result_count)[-result_count:]
    local_positions = local_positions[
        np.argsort(scores[local_positions], kind="stable")[::-1]
    ]
    selected_positions = candidate_positions[local_positions]

    results = index.manifest.iloc[selected_positions].copy().reset_index(drop=True)
    results.insert(0, "similarity", scores[local_positions].astype(float))
    return results

