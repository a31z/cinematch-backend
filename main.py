from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from storage.sqlite_store import init_db, load_profile, save_profile, load_ratings, upsert_rating, delete_rating
from app.recommender import load_movies, recommend

app = FastAPI()

init_db()
movies_df = load_movies("data/movies.csv")

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

    return {
        "recommendations": recs.to_dict(orient="records"),
        "meta": meta.__dict__ if hasattr(meta, "__dict__") else meta,
        "ratings_count": len(ratings),
    }
