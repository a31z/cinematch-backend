import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path("storage/cinematch.db")

def _now_iso() -> str:
    return datetime.utcnow().isoformat()

def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS user_profile (
          id INTEGER PRIMARY KEY CHECK (id = 1),
          weights_json TEXT NOT NULL,
          genres_json  TEXT NOT NULL,
          pacing_pref  TEXT NOT NULL,
          discovery_mode INTEGER NOT NULL DEFAULT 0,
          updated_at TEXT NOT NULL
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS user_ratings (
          movie_id INTEGER PRIMARY KEY,
          rating REAL NOT NULL,
          updated_at TEXT NOT NULL
        );
        """)
        conn.commit()

def load_profile() -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        SELECT weights_json, genres_json, pacing_pref, discovery_mode, updated_at
        FROM user_profile WHERE id=1;
        """)
        row = cur.fetchone()
        if not row:
            return None
        weights_json, genres_json, pacing_pref, discovery_mode, updated_at = row
        return {
            "user_weights": json.loads(weights_json),
            "preferred_genres": json.loads(genres_json),
            "pacing_pref": pacing_pref,
            "discovery_mode": bool(discovery_mode),
            "updated_at": updated_at,
        }

def save_profile(user_weights: List[Optional[float]], preferred_genres: List[str], pacing_pref: str, discovery_mode: bool):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO user_profile (id, weights_json, genres_json, pacing_pref, discovery_mode, updated_at)
        VALUES (1, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          weights_json=excluded.weights_json,
          genres_json=excluded.genres_json,
          pacing_pref=excluded.pacing_pref,
          discovery_mode=excluded.discovery_mode,
          updated_at=excluded.updated_at;
        """, (json.dumps(user_weights), json.dumps(preferred_genres), pacing_pref, int(discovery_mode), _now_iso()))
        conn.commit()

def load_ratings() -> Dict[int, float]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT movie_id, rating FROM user_ratings;")
        rows = cur.fetchall()
        return {int(mid): float(r) for mid, r in rows}

def upsert_rating(movie_id: int, rating: float):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO user_ratings (movie_id, rating, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(movie_id) DO UPDATE SET
          rating=excluded.rating,
          updated_at=excluded.updated_at;
        """, (int(movie_id), float(rating), _now_iso()))
        conn.commit()

def delete_rating(movie_id: int):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM user_ratings WHERE movie_id=?;", (int(movie_id),))
        conn.commit()
