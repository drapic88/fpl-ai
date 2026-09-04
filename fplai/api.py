"""Thin client for the public Fantasy Premier League API, with on-disk caching.

All endpoints here are the same ones the official website calls. Nothing is
scraped and nothing is submitted without you asking for it.

The cache helpers at the bottom are shared with `news` and `external`, which
fetch from sources outside the FPL API.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests

BASE = "https://fantasy.premierleague.com/api"
CACHE_DIR = Path(os.environ.get("FPL_CACHE_DIR", Path.home() / ".cache" / "fpl-ai"))
UA = "Mozilla/5.0 (compatible; fpl-ai/1.0; personal FPL assistant)"


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def cache_read(key: str, ttl: int) -> Any | None:
    """Return the cached payload for `key` if it is younger than ttl seconds."""
    if ttl <= 0:
        return None
    cp = _cache_path(key)
    if not cp.exists() or (time.time() - cp.stat().st_mtime) >= ttl:
        return None
    try:
        return json.loads(cp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def cache_write(key: str, data: Any) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(key).write_text(json.dumps(data), encoding="utf-8")


def _get(path: str, key: str, ttl: int = 3600, session: requests.Session | None = None) -> Any:
    """GET an endpoint, serving from cache if the cached copy is younger than ttl seconds."""
    cached = cache_read(key, ttl)
    if cached is not None:
        return cached

    sess = session or requests
    resp = sess.get(f"{BASE}/{path}", headers={"User-Agent": UA}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    cache_write(key, data)
    return data


def bootstrap(ttl: int = 1800) -> dict:
    """Players, teams, gameweeks, prices, ownership, injury flags."""
    return _get("bootstrap-static/", "bootstrap", ttl)


def fixtures(ttl: int = 3600) -> list[dict]:
    """Every fixture of the season with FPL's own difficulty ratings."""
    return _get("fixtures/", "fixtures", ttl)


def element_summary(element_id: int, ttl: int = 86400) -> dict:
    """Per-player history: this season game by game, plus one row per past season."""
    return _get(f"element-summary/{element_id}/", f"element_{element_id}", ttl)


def entry(entry_id: int, ttl: int = 600) -> dict:
    """Public info about a manager (your team id is in the URL when you view your points)."""
    return _get(f"entry/{entry_id}/", f"entry_{entry_id}", ttl)


def entry_picks(entry_id: int, event: int, ttl: int = 600) -> dict:
    """A manager's picks for a finished gameweek. Public, but has no selling prices."""
    return _get(f"entry/{entry_id}/event/{event}/picks/", f"picks_{entry_id}_{event}", ttl)


def my_team(entry_id: int, cookie: str, ttl: int = 0) -> dict:
    """Your live squad including selling prices, bank and free transfers.

    Needs the `pl_profile` + session cookie string from a logged-in browser
    (DevTools -> Network -> any api request -> Request Headers -> cookie).
    Read-only; nothing is written back.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(
        f"{BASE}/my-team/{entry_id}/",
        headers={"User-Agent": UA, "Cookie": cookie, "X-Requested-With": "XMLHttpRequest"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def clear_cache() -> int:
    """Delete cached responses. Returns how many files were removed."""
    if not CACHE_DIR.exists():
        return 0
    n = 0
    for f in CACHE_DIR.glob("*.json"):
        f.unlink()
        n += 1
    return n
