from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
from functools import lru_cache
import os
from dotenv import load_dotenv
import requests
import pandas as pd

from storage.sqlite_store import init_db, load_profile, save_profile, load_ratings, upsert_rating, delete_rating
from app.recommender import load_movies, recommend

app = FastAPI()
load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()
movies_df = load_movies("data/movies.csv")

TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"
POSTER_FALLBACK = "https://via.placeholder.com/300x450?text=No+Poster"
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BEARER_TOKEN = os.getenv("TMDB_BEARER_TOKEN")

def _tmdb_get(path: str, params: Optional[Dict[str, Any]] = None):
    if TMDB_BEARER_TOKEN:
        headers = {"Authorization": f"Bearer {TMDB_BEARER_TOKEN}", "accept": "application/json"}
        return requests.get(f"{TMDB_API_BASE}{path}", headers=headers, params=params, timeout=10)
    if TMDB_API_KEY:
        merged = dict(params or {})
        merged["api_key"] = TMDB_API_KEY
        return requests.get(f"{TMDB_API_BASE}{path}", params=merged, timeout=10)
    return None

@lru_cache(maxsize=4096)
def _resolve_poster_url(movie_id: str) -> Optional[str]:
    if not TMDB_API_KEY and not TMDB_BEARER_TOKEN:
        return None

    if movie_id.isdigit():
        response = _tmdb_get(f"/movie/{movie_id}")
        if response and response.ok:
            poster_path = response.json().get("poster_path")
            if poster_path:
                return f"{TMDB_IMAGE_BASE}{poster_path}"

    return None

class ProfilePayload(BaseModel):
    user_weights: List[Optional[float]]
    preferred_genres: List[str]
    pacing_pref: str = "fast"
    discovery_mode: bool = False

class RatingPayload(BaseModel):
    movie_id: int
    rating: float

@app.get("/profile")
def get_profile():
    return {"profile": load_profile()}

@app.post("/profile")
def set_profile(payload: ProfilePayload):
    save_profile(payload.user_weights, payload.preferred_genres, payload.pacing_pref, payload.discovery_mode)
    return {"ok": True}

@app.get("/ratings")
def get_ratings():
    return {"ratings": load_ratings()}

@app.post("/ratings")
def set_rating(payload: RatingPayload):
    upsert_rating(payload.movie_id, payload.rating)
    return {"ok": True}

@app.delete("/ratings/{movie_id}")
def remove_rating(movie_id: int):
    delete_rating(movie_id)
    return {"ok": True}

@app.get("/recommend")
def get_recommendations(top_n: int = 10):
    prof = load_profile()
    if prof is None:
        return {"error": "No profile saved. POST /profile first."}

    ratings = load_ratings()

    recs, meta = recommend(
        movies_df,
        user_weights=prof["user_weights"],
        preferred_genres=prof["preferred_genres"],
        pacing_pref=prof["pacing_pref"],
        discovery_mode=prof["discovery_mode"],
        top_n=top_n,
        ratings=ratings if ratings else None,
    )

    safe_recs = recs.astype(object).where(pd.notna(recs), None)

    return {
        "recommendations": safe_recs.to_dict(orient="records"),
        "meta": meta.__dict__ if hasattr(meta, "__dict__") else meta,
        "ratings_count": len(ratings),
    }

@app.get("/movies")
def movies():
    cols = [
        "id",
        "title",
        "genres",
        "keywords",
        "overview",
        "release_date",
        "runtime",
        "cinematography_rating",
        "pacing_rating",
        "music_rating",
        "direction_rating",
        "plot_rating",
        "tags",
        "overall",
        "pacing",
    ]

    safe_df = movies_df[cols].copy()
    safe_df["id"] = safe_df["id"].astype(str)
    safe_df = safe_df.astype(object).where(pd.notna(safe_df), None)
    return safe_df.to_dict("records")

@app.get("/poster/{movie_id}")
def poster(movie_id: str):
    url = _resolve_poster_url(movie_id)
    if not url:
        return RedirectResponse(POSTER_FALLBACK, status_code=307)
    return RedirectResponse(url, status_code=307)
