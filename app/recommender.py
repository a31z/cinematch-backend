from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------------
# CONFIG
# -------------------------------
DEFAULT_DATA_PATH = Path("data/movies.csv")

FEATURE_COLS = [
    "cinematography_rating",
    "direction_rating",
    "pacing_rating",
    "music_rating",
    "plot_rating",
]

# Feature order:
# [cinematography, direction, pacing_adj, music, plot]
N_FEATURES = 5


# -------------------------------
# DATA LOADING
# -------------------------------
def load_movies(filepath: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Loads movie dataset and prepares genres column (pipe-separated -> list[str]).
    """
    fp = Path(filepath)
    df = pd.read_csv(fp)

    df["genres"] = df["genres"].apply(lambda x: x.split("|") if isinstance(x, str) else [])

    required = {"id", "title", "genres", *FEATURE_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"movies.csv missing required columns: {sorted(missing)}")

    return df


# -------------------------------
# MASKING + NORMALIZATION
# -------------------------------
def mask_from_user_weights(user_weights: List[Optional[float]]) -> np.ndarray:
    """True where criterion is considered important; False where user set None."""
    if len(user_weights) != N_FEATURES:
        raise ValueError(f"user_weights must have length {N_FEATURES}")
    return np.array([w is not None for w in user_weights], dtype=bool)


def apply_mask(vector: np.ndarray, user_weights: List[Optional[float]]) -> np.ndarray:
    """Zero-out dimensions the user marked as unimportant (None)."""
    v = np.array(vector, dtype=float)
    m = mask_from_user_weights(user_weights).astype(float)
    return v * m


def normalize_weights(user_weights: List[Optional[float]]) -> np.ndarray:
    """
    Normalizes importance rankings into weights that sum to 1.
    Supports None -> treated as 0 (excluded).
    """
    weights = np.array([w if w is not None else 0.0 for w in user_weights], dtype=float)
    s = float(weights.sum())
    if s <= 0:
        return np.zeros_like(weights)
    return weights / s


# -------------------------------
# PACING
# -------------------------------
def adjust_pacing(movie_pacing: float, pacing_pref: str) -> float:
    """
    pacing_rating scale = 1..5
    1 = very slow
    5 = very fast
    """
    if str(pacing_pref).lower() == "fast":
        return float(movie_pacing)
    return float(6 - movie_pacing)  # flip for "slow"


# -------------------------------
# VECTORS + SIMILARITY
# -------------------------------
def movie_vector(row: pd.Series, pacing_pref: str) -> np.ndarray:
    """
    Returns movie's 5D feature vector with pacing adjusted for user preference.
    Order: [cinematography, direction, pacing_adj, music, plot]
    """
    pacing_score = adjust_pacing(row["pacing_rating"], pacing_pref)
    return np.array(
        [
            row["cinematography_rating"],
            row["direction_rating"],
            pacing_score,
            row["music_rating"],
            row["plot_rating"],
        ],
        dtype=float,
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(np.dot(a, b) / denom)


def compute_genre_score(movie_genres: List[str], user_genres: List[str]) -> float:
    """Score = fraction of user's preferred genres present in the movie."""
    if not user_genres:
        return 0.0
    matches = len(set(movie_genres) & set(user_genres))
    return matches / len(set(user_genres))


# -------------------------------
# LEARNING
# -------------------------------
def dynamic_beta(n_likes: int, K: int = 5) -> float:
    """beta(n) = K / (K + n)"""
    n = max(0, int(n_likes))
    return K / (K + n)


def learned_profile_from_ratings(
    df: pd.DataFrame,
    ratings: Optional[Dict[int, float]],
    pacing_pref: str,
    like_threshold: int = 4,
) -> Tuple[Optional[np.ndarray], int, List[np.ndarray]]:
    """
    Learned profile = average of feature vectors of liked movies.
    Returns: (learned_profile or None, n_likes, liked_vectors)
    """
    if not ratings:
        return None, 0, []

    liked_vectors: List[np.ndarray] = []
    for _, row in df.iterrows():
        mid = int(row["id"])
        if mid in ratings and float(ratings[mid]) >= like_threshold:
            liked_vectors.append(movie_vector(row, pacing_pref))

    n_likes = len(liked_vectors)
    if n_likes == 0:
        return None, 0, []

    learned = np.mean(np.vstack(liked_vectors), axis=0)
    return learned, n_likes, liked_vectors


def learned_importance_from_likes(liked_vectors: List[np.ndarray], user_weights: List[Optional[float]]) -> Optional[np.ndarray]:
    """
    Learn which criteria matter based on liked movies' feature patterns.
    Returns normalized importance weights (sum=1) on allowed dimensions.
    """
    if not liked_vectors:
        return None

    avg = np.mean(np.vstack(liked_vectors), axis=0)
    avg = apply_mask(avg, user_weights)

    s = float(avg.sum())
    if s <= 0:
        return None
    return avg / s


def blend_importance_weights(baseline_w: np.ndarray, learned_w: Optional[np.ndarray], n_likes: int, K: int = 5) -> np.ndarray:
    """final_w = gamma*baseline_w + (1-gamma)*learned_w"""
    if learned_w is None:
        return baseline_w
    gamma = dynamic_beta(n_likes, K=K)
    return gamma * baseline_w + (1 - gamma) * learned_w


def blended_user_profile(baseline: np.ndarray, learned: Optional[np.ndarray], n_likes: int, K: int = 5) -> Tuple[np.ndarray, float]:
    """Option C: final_profile = beta*baseline + (1-beta)*learned"""
    if learned is None:
        return baseline, 1.0
    beta = dynamic_beta(n_likes, K=K)
    return beta * baseline + (1 - beta) * learned, beta


def discovery_user_profile(user_profile: np.ndarray, user_weights: List[Optional[float]], ideal_rating: float = 5.0, alpha: float = 0.85) -> np.ndarray:
    """
    Softens user profile toward a neutral vector on allowed dimensions.
    """
    user_profile = np.array(user_profile, dtype=float)
    m = mask_from_user_weights(user_weights)

    neutral = np.zeros_like(user_profile)
    if m.any():
        neutral[m] = float(ideal_rating) / int(m.sum())

    return alpha * user_profile + (1 - alpha) * neutral


def safe_nonzero_profile(user_profile: np.ndarray, user_weights: List[Optional[float]], ideal_rating: float = 5.0) -> np.ndarray:
    """Guard against all-zero vectors (cosine similarity degeneracy)."""
    if np.linalg.norm(user_profile) != 0:
        return user_profile

    m = mask_from_user_weights(user_weights)
    fallback = np.zeros_like(user_profile, dtype=float)
    if m.any():
        fallback[m] = float(ideal_rating) / int(m.sum())
    return fallback


# -------------------------------
# API
# -------------------------------
@dataclass
class RecommendMeta:
    n_likes: int
    beta: float
    confidence: float
    baseline_importance: List[float]
    learned_importance: Optional[List[float]]
    final_importance: List[float]
    baseline_profile: List[float]
    learned_profile: Optional[List[float]]
    final_profile: List[float]


def recommend(
    df: pd.DataFrame,
    user_weights: List[Optional[float]],
    preferred_genres: List[str],
    pacing_pref: str = "fast",
    discovery_mode: bool = False,
    top_n: int = 10,
    ratings: Optional[Dict[int, float]] = None,  # {movie_id: rating}
    like_threshold: int = 4,
    K: int = 5,
    ideal_rating: float = 5.0,
) -> Tuple[pd.DataFrame, RecommendMeta]:
    """
    returns (recommendations_df, meta)
    """

    df = df.copy()

    learned_profile, n_likes, liked_vectors = learned_profile_from_ratings(
        df, ratings=ratings, pacing_pref=pacing_pref, like_threshold=like_threshold
    )

    baseline_w = normalize_weights(user_weights)
    learned_w = learned_importance_from_likes(liked_vectors, user_weights)
    final_w = blend_importance_weights(baseline_w, learned_w, n_likes=n_likes, K=K)

    baseline_profile = final_w * float(ideal_rating)
    baseline_profile = apply_mask(baseline_profile, user_weights)

    if learned_profile is not None:
        learned_profile = apply_mask(learned_profile, user_weights)

    user_profile, beta = blended_user_profile(baseline_profile, learned_profile, n_likes=n_likes, K=K)

    if discovery_mode:
        user_profile = discovery_user_profile(user_profile, user_weights, ideal_rating=ideal_rating, alpha=0.85)

    user_profile = safe_nonzero_profile(user_profile, user_weights, ideal_rating=ideal_rating)

    def _relevance(row: pd.Series) -> float:
        x = movie_vector(row, pacing_pref)
        x = apply_mask(x, user_weights)
        return cosine_similarity(user_profile, x)

    df["relevance_score"] = df.apply(_relevance, axis=1)

    df["genre_score"] = df["genres"].apply(lambda g: compute_genre_score(g, preferred_genres))

    if discovery_mode:
        df["final_score"] = 0.9 * df["relevance_score"] + 0.1 * df["genre_score"]
    else:
        df["final_score"] = 0.7 * df["relevance_score"] + 0.3 * df["genre_score"]

    recs = df.sort_values("final_score", ascending=False).head(int(top_n))

    confidence = 0.4 + 0.6 * (n_likes / (n_likes + K))
    
    meta = RecommendMeta(
        n_likes=int(n_likes),
        beta=float(beta),
        confidence=float(confidence),
        baseline_importance=[float(x) for x in baseline_w],
        learned_importance=None if learned_w is None else [float(x) for x in learned_w],
        final_importance=[float(x) for x in final_w],
        baseline_profile=[float(x) for x in baseline_profile],
        learned_profile=None if learned_profile is None else [float(x) for x in learned_profile],
        final_profile=[float(x) for x in user_profile],
    )

    return recs, meta
