"""Foreign-league scoring rates for players with no Premier League record.

When a striker arrives from Serie A, the old model had nothing to go on and
fell back on a price-shaped prior: it knew he was expensive, and guessed. That
prior is doing an enormous amount of work in exactly the moment it matters most
-- the pre-season squad pick, when a tenth of the player pool has just moved.

Understat publishes per-player minutes, goals, assists, xG and xA for the top
five European leagues. Its league pages fetch that through a JSON endpoint,
which is what this module calls, cached for a week.

Two deliberate choices about how the numbers are used:

* **Expected goals over actual goals.** A 22-goal season on 12 xG is a fluke
  waiting to regress; npxG and xA travel across leagues far better than the
  finishing that produced them.
* **A league-strength haircut, then treated as weak evidence.** Ligue 1 output
  does not survive contact with the Premier League one-for-one, and even after
  the haircut the minutes are down-weighted so the existing empirical-Bayes
  shrinkage keeps pulling an unproven signing toward his positional prior.

Nothing here is required. If the site is unreachable or a player cannot be
matched, the model falls back to the price prior exactly as before.
"""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
from dataclasses import dataclass, field

import requests

from . import api

ENDPOINT = "https://understat.com/main/getPlayersStats/"

#: League -> multiplier converting that league's output to a Premier League scale.
#: Roughly the consensus translation factors used for cross-league projection;
#: the exact values are arguable, which is why they are config rather than code.
LEAGUE_STRENGTH: dict[str, float] = {
    "EPL": 1.00,
    "La_liga": 0.94,
    "Bundesliga": 0.92,
    "Serie_A": 0.92,
    "Ligue_1": 0.86,
}


@dataclass
class ExternalConfig:
    enabled: bool = True
    leagues: tuple[str, ...] = ("La_liga", "Bundesliga", "Serie_A", "Ligue_1", "EPL")
    #: Starting year of the season to read, e.g. 2025 means the 2025/26 campaign.
    season: int | None = None
    ttl: int = 7 * 86400
    #: Weight on expected (npxG/xA) rather than actual goals and assists.
    xg_weight: float = 0.60
    #: Minutes are scaled down before being handed to the shrinkage step, so a
    #: foreign season counts as weaker evidence than the same minutes in the PL.
    minutes_trust: float = 0.55
    strength: dict = field(default_factory=lambda: dict(LEAGUE_STRENGTH))


def default_season(today: dt.date | None = None) -> int:
    """The most recently completed season's starting year.

    Between August and December we want last season; from January onward the
    current campaign is already the relevant one but incomplete, so last
    season's full record is still the safer read for a new arrival.
    """
    today = today or dt.date.today()
    return today.year - 1 if today.month >= 7 else today.year - 2


# --------------------------------------------------------------------------- fetching


def fetch_league(league: str, season: int, ttl: int) -> list[dict]:
    """Raw per-player season totals for one league, cached on disk."""
    key = f"understat_{league}_{season}"
    cached = api.cache_read(key, ttl)
    if cached is not None:
        return cached

    try:
        resp = requests.post(
            ENDPOINT,
            data={"league": league, "season": str(season)},
            headers={
                "User-Agent": api.UA,
                "X-Requested-With": "XMLHttpRequest",
                "Referer": f"https://understat.com/league/{league}/{season}",
            },
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError):
        return []

    players = payload.get("players") or []
    if not payload.get("success") or not players:
        return []
    api.cache_write(key, players)
    return players


# --------------------------------------------------------------------------- name matching

_STRIP = re.compile(r"[^a-z ]+")


def normalise(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(name))
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", _STRIP.sub(" ", plain.lower())).strip()


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_index(cfg: ExternalConfig, season: int, verbose: bool = False) -> dict[str, dict]:
    """normalised player name -> per-90 rates, already on a Premier League scale.

    A name claimed by two different players across the leagues we read is dropped
    rather than guessed at: a wrong match is worse than no match.
    """
    index: dict[str, dict] = {}
    clashes: set[str] = set()

    for league in cfg.leagues:
        factor = float(cfg.strength.get(league, 0.90))
        rows = fetch_league(league, season, cfg.ttl)
        if verbose:
            print(f"  external: {league} {season}/{str(season + 1)[-2:]} -> {len(rows)} players")
        for r in rows:
            minutes = _num(r.get("time"))
            if minutes < 450:            # half a season of football; below that it is noise
                continue
            name = normalise(r.get("player_name", ""))
            if not name:
                continue
            if name in index:
                clashes.add(name)
                continue

            per90 = minutes / 90.0
            goals = _num(r.get("goals"))
            npg = _num(r.get("npg"), goals)
            npxg = _num(r.get("npxG"), _num(r.get("xG")))
            assists = _num(r.get("assists"))
            xa = _num(r.get("xA"))

            w = cfg.xg_weight
            goal_rate = ((1 - w) * npg + w * npxg) / per90 * factor
            assist_rate = ((1 - w) * assists + w * xa) / per90 * factor

            index[name] = {
                "league": league,
                "season": season,
                "minutes": minutes,
                "games": _num(r.get("games")),
                "goal_rate": max(0.0, goal_rate),
                "assist_rate": max(0.0, assist_rate),
                "team": r.get("team_title", ""),
                "strength_factor": factor,
            }

    for name in clashes:
        index.pop(name, None)
    if verbose and clashes:
        print(f"  external: dropped {len(clashes)} ambiguous name(s)")
    return index


def match_players(elements: list[dict], index: dict[str, dict]) -> dict[int, dict]:
    """FPL player id -> external rates, for the players we can confidently identify.

    Only full-name matches are accepted. Surnames alone are far too collision-prone
    across five leagues, and a mis-attributed scoring rate would be invisible in the
    output while quietly corrupting a squad pick.
    """
    out: dict[int, dict] = {}
    for e in elements:
        candidates = [
            normalise(f"{e.get('first_name', '')} {e.get('second_name', '')}"),
            normalise(e.get("known_name") or ""),
        ]
        for key in candidates:
            if key and key in index:
                out[int(e["id"])] = index[key]
                break
    return out


def load(elements: list[dict], cfg: ExternalConfig | None = None,
         today: dt.date | None = None, verbose: bool = False) -> dict[int, dict]:
    """Top level: fetch, index and match. Returns {} if disabled or unreachable."""
    cfg = cfg or ExternalConfig()
    if not cfg.enabled:
        return {}
    season = cfg.season if cfg.season is not None else default_season(today)
    try:
        index = build_index(cfg, season, verbose=verbose)
    except Exception as exc:  # never let an outside site break a squad pick
        if verbose:
            print(f"  ! external stats unavailable ({type(exc).__name__}), using price priors")
        return {}
    matched = match_players(elements, index)
    if verbose:
        print(f"  external: matched {len(matched)} FPL player(s) to {len(index)} indexed players")
    return matched
