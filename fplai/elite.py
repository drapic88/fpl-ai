"""Elite managers: who they are, and what they did this season.

FPL keeps a manager's picks, transfers and chips for the current season only.
Earlier seasons survive as one row each -- points and overall rank -- so the
past can say *who* is good, but only this season can say *what they did*.

Two groups are taken from the top of the overall league:

* **current** -- the best ranks this season. Strong now, but a ten-week rank
  carries a lot of luck.
* **consistent** -- the best median finish over past seasons. Fewer of them are
  near the top this year, but their record is not one lucky run.

Everything here only reads public endpoints, slowly, and caches every response.
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd
import requests

from . import api

OVERALL_LEAGUE = 314
FINISHED_TTL = 30 * 86400   # a finished gameweek's picks never change


@dataclass
class EliteConfig:
    scan_pages: int = 20          # 50 managers a page from the top of the overall league
    n_current: int = 100          # best by this season's rank
    n_consistent: int = 100       # best by median past finish
    min_past_seasons: int = 3     # fewer than this and a past record proves little
    recent_seasons: int = 5       # how many past seasons the median looks at
    pause: float = 0.12           # seconds between uncached requests


def _polite(fn, *args, pause: float = 0.12, retries: int = 4, **kwargs):
    """Call an `api` function, backing off on rate limits and transient errors."""
    for attempt in range(retries + 1):
        try:
            out = fn(*args, **kwargs)
            time.sleep(pause)
            return out
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else 0
            if code == 404 or attempt == retries:
                raise
            time.sleep(5.0 * (attempt + 1) if code == 429 else 1.0 + attempt)
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(1.0 + attempt)
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------- who is elite


def consistency_score(past: list[dict], min_seasons: int = 3, recent: int = 5) -> float | None:
    """Median log10 overall rank over the last `recent` seasons; lower is better.

    A log scale keeps one season at rank 50 from outweighing four at rank 5,000:
    the question is "is this manager reliably near the top", not "did they ever win".
    """
    ranks = [int(p["rank"]) for p in past if p.get("rank")]
    if len(ranks) < min_seasons:
        return None
    vals = sorted(math.log10(max(1, r)) for r in ranks[-recent:])
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def select_managers(cfg: EliteConfig | None = None, verbose: bool = True) -> pd.DataFrame:
    """Scan the top of the overall league and pick the current and consistent groups."""
    cfg = cfg or EliteConfig()
    rows = []
    for page in range(1, cfg.scan_pages + 1):
        data = _polite(api.league_standings, OVERALL_LEAGUE, page, pause=cfg.pause)
        for r in data["standings"]["results"]:
            rows.append({"entry": int(r["entry"]), "rank": int(r["rank"]), "total": int(r["total"])})
    if verbose:
        print(f"  scanned {len(rows)} managers from the top of the overall league")

    for i, r in enumerate(rows, 1):
        hist = _polite(api.entry_history, r["entry"], pause=cfg.pause)
        past = hist.get("past", [])
        r["past_seasons"] = len(past)
        r["past_best_rank"] = min((int(p["rank"]) for p in past if p.get("rank")), default=None)
        r["score"] = consistency_score(past, cfg.min_past_seasons, cfg.recent_seasons)
        if verbose and i % 200 == 0:
            print(f"  ...{i}/{len(rows)} past records")

    df = pd.DataFrame(rows)
    current = set(df.nsmallest(cfg.n_current, "rank")["entry"])
    consistent = set(df.dropna(subset=["score"]).nsmallest(cfg.n_consistent, "score")["entry"])
    df["group"] = [
        "both" if e in current and e in consistent
        else "current" if e in current
        else "consistent" if e in consistent
        else ""
        for e in df["entry"]
    ]
    return df[df["group"] != ""].reset_index(drop=True)


# --------------------------------------------------------------------------- what they did


def fetch_decisions(entries: list[int], last_gw: int, cfg: EliteConfig | None = None,
                    verbose: bool = True) -> dict[int, dict]:
    """Picks for every finished gameweek, all transfers, and the season history."""
    cfg = cfg or EliteConfig()
    out: dict[int, dict] = {}
    for i, e in enumerate(entries, 1):
        picks = {}
        for gw in range(1, last_gw + 1):
            try:
                picks[gw] = _polite(api.entry_picks, e, gw, ttl=FINISHED_TTL, pause=cfg.pause)
            except requests.HTTPError:
                continue  # joined after this gameweek
        out[e] = {
            "picks": picks,
            "transfers": _polite(api.entry_transfers, e, pause=cfg.pause),
            "history": _polite(api.entry_history, e, pause=cfg.pause),
        }
        if verbose and i % 20 == 0:
            print(f"  ...{i}/{len(entries)} managers")
    return out


def effective_ownership(picks: list[list[dict]]) -> dict[int, float]:
    """Element -> effective ownership in % across these squads for one gameweek.

    EO counts what a manager actually gets from the player: 0 on the bench, 1 in
    the XI, 2 as captain, 3 with a triple captain. 150% means the average elite
    manager earns one and a half times the player's points.
    """
    if not picks:
        return {}
    tot: Counter = Counter()
    for squad in picks:
        for p in squad:
            tot[int(p["element"])] += int(p.get("multiplier", 1))
    return {el: 100.0 * v / len(picks) for el, v in tot.items()}


def eo_by_gw(decisions: dict[int, dict]) -> dict[int, dict[int, float]]:
    """Gameweek -> element -> elite EO."""
    per_gw: dict[int, list] = defaultdict(list)
    for d in decisions.values():
        for gw, payload in d["picks"].items():
            per_gw[int(gw)].append(payload.get("picks", []))
    return {gw: effective_ownership(squads) for gw, squads in sorted(per_gw.items())}


def manager_gameweeks(decisions: dict[int, dict], points: dict[int, dict[int, float]],
                      groups: dict[int, str] | None = None) -> pd.DataFrame:
    """One row per manager per gameweek: points, transfers, hit, chip, captain."""
    groups = groups or {}
    rows = []
    for e, d in decisions.items():
        current = {int(c["event"]): c for c in d["history"].get("current", [])}
        for gw, payload in d["picks"].items():
            gw = int(gw)
            cap = next((p for p in payload.get("picks", []) if p.get("is_captain")), None)
            h = current.get(gw, {})
            cap_el = int(cap["element"]) if cap else None
            rows.append({
                "entry": e,
                "group": groups.get(e, ""),
                "gw": gw,
                "points": h.get("points"),
                "transfers": h.get("event_transfers", 0),
                "hit": h.get("event_transfers_cost", 0),
                "bench_points": h.get("points_on_bench", 0),
                "chip": payload.get("active_chip") or "",
                "captain": cap_el,
                "captain_points": points.get(gw, {}).get(cap_el) if cap_el else None,
            })
    return pd.DataFrame(rows)


def summarise(mgw: pd.DataFrame) -> dict:
    """The headline habits: transfers, hits, chip timing, captaincy, bench."""
    if mgw.empty:
        return {}
    per_mgr = mgw.groupby("entry").agg(
        transfers=("transfers", "sum"), hits=("hit", lambda s: int((s > 0).sum())),
        hit_cost=("hit", "sum"), gws=("gw", "nunique"),
    )
    chips = mgw[mgw["chip"] != ""].groupby("chip")["gw"].describe()[["count", "min", "50%", "max"]]
    cap = mgw.dropna(subset=["captain"]).groupby("gw")["captain"].agg(
        lambda s: s.value_counts(normalize=True).iloc[0]
    )
    return {
        "managers": int(mgw["entry"].nunique()),
        "transfers_per_gw": float(per_mgr["transfers"].sum() / per_mgr["gws"].sum()),
        "hits_per_manager": float(per_mgr["hits"].mean()),
        "hit_points_per_manager": float(per_mgr["hit_cost"].mean()),
        "share_never_hit": float((per_mgr["hits"] == 0).mean()),
        "chips": chips.round(1).to_dict(orient="index"),
        "captain_consensus": float(cap.mean()) if len(cap) else 0.0,
        "captain_points": float(mgw["captain_points"].mean()),
        "bench_points_per_gw": float(mgw["bench_points"].mean()),
    }
