"""Projection engine.

Turns raw FPL data into an expected-points number for every player in every
upcoming gameweek. Deconstructs returns into probabilistic scoring components:
appearance, goals (xG), assists (xA), clean sheets (Poisson), goals conceded
penalties, saves, cards, and bonus points.

Three things gate those components before any of them can be scored, and all
three are modelled per gameweek rather than once for the whole horizon:

* **Availability** (`news`) -- injuries, suspensions, personal absences and
  departures, with a return date where the news gives one.
* **Competition for the shirt** (`transfers.depth_chart`) -- a club only has
  eleven places, and a new arrival takes one off somebody.
* **Settling in** (`transfers.settling_multiplier`) -- a player who signed
  three weeks ago is not yet the player his old league's numbers describe.
"""

from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd

from . import api, news as news_mod, transfers as transfers_mod

POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
DEFENSIVE = {1, 2}

# Scoring values per FPL rulebook
GOAL_PTS = {1: 6.0, 2: 6.0, 3: 5.0, 4: 4.0}
CS_PTS = {1: 4.0, 2: 4.0, 3: 1.0, 4: 0.0}
ASSIST_PTS = 3.0
CARD_PENALTY = 1.0

#: How much a substitute appearance counts toward "he is in the manager's plans",
#: on a scale where a start is 1.0 and being left out of the squad is 0.0.
SUB_APPEARANCE_CREDIT = 0.35


@dataclass
class ModelConfig:
    horizon: int = 5              # how many gameweeks to look ahead
    decay: float = 0.86           # weight of gameweek n+1 relative to n
    shrink_matches: float = 6.0   # prior strength, in matches, for past rate
    shrink_inseason: float = 4.0  # prior strength for current season matches
    home_bonus: float = 1.06
    fdr_weight: float = 0.06      # per point of FPL fixture difficulty away from 3
    strength_exp: float = 0.50    # sensitivity to team strength ratings
    ep_blend: float = 0.25        # weight given to FPL's own ep_next in the first gameweek
    min_minutes_for_rate: int = 250
    seasons_weight: tuple[float, ...] = (0.70, 0.30)  # last season, season before
    inseason_match_decay: float = 0.92  # EWMA decay per match in current season
    base_home_goals: float = 1.55 # Premier League long-term average home goals
    base_away_goals: float = 1.25 # Premier League long-term average away goals
    # Prior strength, in matches, for "is he a starter". At 1.5 the opening
    # weekend's team sheet carries 40% of the weight against the pre-season prior --
    # enough to notice a benching, not so much that one substitution rewrites a role.
    start_prior_matches: float = 1.5
    # How much of a new signing's pre-move starting record still applies at his
    # new club. 1.0 trusts it fully; 0 treats him as a complete unknown.
    new_signing_start_pull: float = 0.6
    overrides: dict = field(default_factory=dict)
    # Sub-models. Each carries its own knobs; see the modules for what they mean.
    news: news_mod.NewsConfig = field(default_factory=news_mod.NewsConfig)
    squad: transfers_mod.SquadConfig = field(default_factory=transfers_mod.SquadConfig)


# --------------------------------------------------------------------------- history fetching


def fetch_history(player_ids: list[int], workers: int = 6, verbose: bool = True) -> dict[int, dict]:
    """Pull past-season totals and current-season match logs for every player."""
    out: dict[int, dict] = {}

    def one(pid: int):
        try:
            summary = api.element_summary(pid)
            return pid, {
                "past": summary.get("history_past", []),
                "this_season": summary.get("history", []),
            }
        except Exception:
            return pid, {"past": [], "this_season": []}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (pid, hist) in enumerate(pool.map(one, player_ids), 1):
            out[pid] = hist
            if verbose and i % 100 == 0:
                print(f"  ...{i}/{len(player_ids)} player histories")
    return out


def _get_history_parts(hist_entry: list | dict | None) -> tuple[list[dict], list[dict]]:
    """Helper to extract past seasons and this season history uniformly."""
    if isinstance(hist_entry, dict):
        return hist_entry.get("past", []), hist_entry.get("this_season", [])
    elif isinstance(hist_entry, list):
        return hist_entry, []
    return [], []


def matches_since_joining(hist_curr: list[dict], joined: str | None) -> float:
    """Appearances a player has made *for his current club* this season.

    A January signing's match log still contains his old club's games. Counting
    only the appearances after the join date is what makes the settling-in
    discount fade at the right rate.
    """
    played = [m for m in hist_curr if float(m.get("minutes", 0) or 0) > 0]
    if not joined:
        return float(len(played))
    count = 0
    for m in played:
        ko = str(m.get("kickoff_time") or "")[:10]
        if not ko or ko >= joined:
            count += 1
    return float(count)


# --------------------------------------------------------------------------- rates & priors


def _rates_from_history(
    hist_past: list[dict],
    hist_curr: list[dict],
    pos_id: int,
    cfg: ModelConfig,
) -> dict[str, float]:
    """Extract EWMA per-90 statistics from past seasons and in-season match logs."""
    # 1. In-season stats (EWMA)
    curr_mins = 0.0
    curr_goals = 0.0
    curr_assists = 0.0
    curr_cs = 0.0
    curr_saves = 0.0
    curr_yc = 0.0
    curr_starts = 0
    curr_matches = len(hist_curr)

    if hist_curr:
        weight = 1.0
        total_w = 0.0
        for m in reversed(hist_curr):  # most recent match first
            mins = float(m.get("minutes", 0) or 0)
            if mins > 0:
                curr_mins += weight * mins
                curr_goals += weight * float(m.get("goals_scored", 0) or 0)
                curr_assists += weight * float(m.get("assists", 0) or 0)
                curr_cs += weight * float(m.get("clean_sheets", 0) or 0)
                curr_saves += weight * float(m.get("saves", 0) or 0)
                curr_yc += weight * float(m.get("yellow_cards", 0) or 0)
                if mins >= 55:
                    curr_starts += 1
                total_w += weight
            weight *= cfg.inseason_match_decay

    # 2. Past seasons stats
    past_mins = 0.0
    past_goals = 0.0
    past_assists = 0.0
    past_cs = 0.0
    past_saves = 0.0
    past_pts = 0.0

    if hist_past:
        seasons = hist_past[-len(cfg.seasons_weight):][::-1]
        for w, s in zip(cfg.seasons_weight, seasons):
            mins = float(s.get("minutes", 0) or 0)
            if mins < cfg.min_minutes_for_rate:
                continue
            past_mins += w * mins
            past_pts += w * float(s.get("total_points", 0) or 0)
            # When specific counts are present in past seasons, use them
            past_goals += w * float(s.get("goals_scored", 0) or 0)
            past_assists += w * float(s.get("assists", 0) or 0)
            past_cs += w * float(s.get("clean_sheets", 0) or 0)
            past_saves += w * float(s.get("saves", 0) or 0)

    return {
        "curr_mins": curr_mins,
        "curr_matches": float(curr_matches),
        "curr_starts": float(curr_starts),
        "curr_goals_p90": (curr_goals / (curr_mins / 90.0)) if curr_mins >= 90 else None,
        "curr_assists_p90": (curr_assists / (curr_mins / 90.0)) if curr_mins >= 90 else None,
        "curr_cs_p90": (curr_cs / (curr_mins / 90.0)) if curr_mins >= 90 else None,
        "curr_saves_p90": (curr_saves / (curr_mins / 90.0)) if curr_mins >= 90 else None,
        "past_mins": past_mins,
        "past_pts_p90": (past_pts / (past_mins / 90.0)) if past_mins >= 90 else None,
        "past_goals_p90": (past_goals / (past_mins / 90.0)) if past_mins >= 90 else None,
        "past_assists_p90": (past_assists / (past_mins / 90.0)) if past_mins >= 90 else None,
        "past_cs_p90": (past_cs / (past_mins / 90.0)) if past_mins >= 90 else None,
        "past_saves_p90": (past_saves / (past_mins / 90.0)) if past_mins >= 90 else None,
    }


def _minutes_profile(
    hist_past: list[dict],
    hist_curr: list[dict],
    price: float,
    ext: dict | None = None,
    is_new_signing: bool = False,
    prior_matches: float = 1.0,
    new_signing_pull: float = 0.6,
) -> tuple[float, float]:
    """Return (p_start_base, avg_mins_when_starting), before availability or competition.

    Availability is applied per gameweek and competition is settled by the depth
    chart, so this function answers only: how much of a regular is this player,
    judging by his own record?

    Whatever he did before this season is the *prior*; what he has done in it is
    the evidence, shrunk toward that prior by sample size. This matters most in
    the first weeks of a season, when there is exactly one match to go on -- and
    "was he in the team for the opener?" is about the most informative single
    fact available. An earlier version required two matches before it would look
    at the current season at all, which meant that through GW1 a player who had
    just been left out still projected off last season's minutes.
    """
    # --- prior: what his record before this season suggests ------------------
    if hist_past:
        past_mins = float(hist_past[-1].get("minutes", 0) or 0)
        prior = min(1.0, past_mins / 2800.0)
        avg_mins_when_start = 82.0
    elif ext and ext.get("minutes"):
        # No Premier League record, but we know what he played abroad last season.
        # That is far better evidence of "is he a starter" than his price tag.
        prior = float(np.clip(float(ext["minutes"]) / 2800.0, 0.1, 0.95))
        games = float(ext.get("games") or 0)
        avg_mins_when_start = float(np.clip(
            float(ext["minutes"]) / games, 55.0, 90.0
        )) if games > 0 else 80.0
    else:
        # Nothing at all to go on: fall back to the price proxy.
        prior = float(np.clip((price - 4.5) / 5.0, 0.15, 0.90))
        avg_mins_when_start = 80.0

    # Minutes racked up elsewhere say much less about a place in *this* team, so
    # pull a new signing's prior toward "unknown" rather than trusting it whole.
    # A keeper who played every week on loan last season is not thereby the
    # number one at the club that just bought him.
    if is_new_signing:
        prior = 0.5 + (prior - 0.5) * new_signing_pull

    # --- evidence: what he has actually done this season ---------------------
    # Graded, not binary. Coming off the bench is weak evidence of a place in the
    # side; being left out of the squad altogether is strong evidence against one.
    # Scoring both as "did not start" throws away the difference between a starter
    # who was substituted early and a player the manager has frozen out.
    recent = hist_curr[-4:] if hist_curr else []
    if recent:
        scores, started = [], []
        for m in recent:
            mins = float(m.get("minutes", 0) or 0)
            if mins >= 55:
                scores.append(1.0)
                started.append(mins)
            elif mins > 0:
                scores.append(SUB_APPEARANCE_CREDIT)
            else:
                scores.append(0.0)
        observed = float(np.mean(scores))
        w = len(recent) / (len(recent) + max(0.0, prior_matches))
        p_start_base = w * observed + (1.0 - w) * prior
        if started:
            avg_mins_when_start = float(np.mean(started))
    else:
        p_start_base = prior

    return float(np.clip(p_start_base, 0.0, 1.0)), avg_mins_when_start


def expected_minutes(p_start: float, avail: float, avg_start_mins: float,
                     settle: float = 1.0) -> float:
    """Expected minutes given a settled start probability and this week's availability.

    `p_start` already carries availability (the depth chart competes on availability-
    weighted probabilities), so the substitute term is the availability left over
    after the chance of starting is accounted for.
    """
    p_sub = float(np.clip((avail - p_start) * 0.45, 0.0, 0.50))
    raw = p_start * avg_start_mins + p_sub * 20.0
    return float(np.clip(raw * settle, 0.0, 90.0))


# --------------------------------------------------------------------------- fixtures & match Poisson


def fixture_table(fixtures: list[dict], start_gw: int, horizon: int) -> dict[int, dict[int, list[dict]]]:
    """team_id -> gw -> list of fixtures."""
    table: dict[int, dict[int, list[dict]]] = {}
    for f in fixtures:
        gw = f.get("event")
        if gw is None or gw < start_gw or gw >= start_gw + horizon:
            continue
        for side, opp_side, home in (("team_h", "team_a", True), ("team_a", "team_h", False)):
            t, o = f[side], f[opp_side]
            diff = f["team_h_difficulty"] if home else f["team_a_difficulty"]
            table.setdefault(t, {}).setdefault(gw, []).append(
                {"opponent": o, "home": home, "difficulty": diff}
            )
    return table


def _team_stat(team: dict, stat_key: str, fallback_key: str) -> float:
    val = float(team.get(stat_key) or 0)
    if val <= 0:
        val = float(team.get(fallback_key) or 0)
    return val if val > 0 else 3.0


def _match_poisson_expectation(
    team: dict,
    opp: dict,
    home: bool,
    avg_att_h: float,
    avg_att_a: float,
    avg_def_h: float,
    avg_def_a: float,
    cfg: ModelConfig,
) -> tuple[float, float, float, float]:
    """Calculate (lambda_team, mu_conceded, p_clean_sheet, p_concede_2plus).

    Uses home/away offensive & defensive strength ratings to estimate expected goals
    scored (lambda) and conceded (mu), then derives exact Poisson probabilities.
    """
    if home:
        att = _team_stat(team, "strength_attack_home", "strength_overall_home") / avg_att_h
        opp_def = avg_def_a / _team_stat(opp, "strength_defence_away", "strength_overall_away")
        lam = cfg.base_home_goals * ((att * opp_def) ** cfg.strength_exp)

        opp_att = _team_stat(opp, "strength_attack_away", "strength_overall_away") / avg_att_a
        team_def = avg_def_h / _team_stat(team, "strength_defence_home", "strength_overall_home")
        mu = cfg.base_away_goals * ((opp_att * team_def) ** cfg.strength_exp)
    else:
        att = _team_stat(team, "strength_attack_away", "strength_overall_away") / avg_att_a
        opp_def = avg_def_h / _team_stat(opp, "strength_defence_home", "strength_overall_home")
        lam = cfg.base_away_goals * ((att * opp_def) ** cfg.strength_exp)

        opp_att = _team_stat(opp, "strength_attack_home", "strength_overall_home") / avg_att_h
        team_def = avg_def_a / _team_stat(team, "strength_defence_away", "strength_overall_away")
        mu = cfg.base_home_goals * ((opp_att * team_def) ** cfg.strength_exp)

    lam = max(0.2, min(4.5, lam))
    mu = max(0.2, min(4.5, mu))

    # Poisson probabilities
    p_cs = math.exp(-mu)
    p_concede_2plus = 1.0 - math.exp(-mu) * (1.0 + mu)

    return lam, mu, p_cs, p_concede_2plus


# --------------------------------------------------------------------------- availability fallback


def _fallback_availability(bootstrap: dict, gws: list[int], cfg: ModelConfig,
                           today: dt.date) -> dict[int, news_mod.Availability]:
    """Availability from the FPL flags alone, with no network calls.

    Used when the caller did not supply one, so `build` behaves sensibly (and
    offline) on its own. The gameweek deadlines still come from the bootstrap:
    without them a parsed return date cannot be placed in the schedule and would
    be read as "out for the whole horizon", which is worse than not parsing it.
    """
    local = news_mod.NewsConfig(
        injury_ramp=cfg.news.injury_ramp,
        unknown_return_curve=cfg.news.unknown_return_curve,
        personal_curve=cfg.news.personal_curve,
        doubt_recovery=cfg.news.doubt_recovery,
        default_suspension_gws=cfg.news.default_suspension_gws,
        use_web=False,
    )
    deadlines = news_mod.event_deadlines(bootstrap.get("events", []))
    out = {}
    for e in bootstrap.get("elements", []):
        sig = news_mod.parse_fpl_news(e, deadlines, today)
        sig.curve = news_mod.availability_curve(sig, gws, local)
        out[sig.player_id] = sig
    return out


# --------------------------------------------------------------------------- main projection engine


def build(
    bootstrap: dict,
    fixtures_raw: list[dict],
    start_gw: int,
    cfg: ModelConfig,
    history: dict[int, dict | list[dict]] | None = None,
    availability: dict[int, news_mod.Availability] | None = None,
    moves: dict[int, dict] | None = None,
    external: dict[int, dict] | None = None,
    today: dt.date | None = None,
) -> tuple[pd.DataFrame, dict[int, dict[int, float]]]:
    """Project every player's points for every gameweek in the horizon.

    `availability`, `moves` and `external` are optional; when omitted the model
    derives availability from the FPL flags and treats everyone as settled, which
    keeps `build` usable offline and in tests.
    """
    teams = {t["id"]: t for t in bootstrap["teams"]}
    elements = bootstrap["elements"]
    history = history or {}
    external = external or {}
    today = today or dt.date.today()
    gws = list(range(start_gw, start_gw + cfg.horizon))

    if availability is None:
        availability = _fallback_availability(bootstrap, gws, cfg, today)
    if moves is None:
        moves = transfers_mod.classify_moves(elements, today, cfg.squad)

    # Calculate league benchmark strength averages
    att_h = np.mean([_team_stat(t, "strength_attack_home", "strength_overall_home") for t in teams.values()])
    att_a = np.mean([_team_stat(t, "strength_attack_away", "strength_overall_away") for t in teams.values()])
    def_h = np.mean([_team_stat(t, "strength_defence_home", "strength_overall_home") for t in teams.values()])
    def_a = np.mean([_team_stat(t, "strength_defence_away", "strength_overall_away") for t in teams.values()])

    avg_att_h = float(att_h) if att_h > 0 else 3.0
    avg_att_a = float(att_a) if att_a > 0 else 3.0
    avg_def_h = float(def_h) if def_h > 0 else 3.0
    avg_def_a = float(def_a) if def_a > 0 else 3.0
    avg_league_goals = (cfg.base_home_goals + cfg.base_away_goals) / 2.0

    rows = []
    for e in elements:
        pid = int(e["id"])
        hist_past, hist_curr = _get_history_parts(history.get(pid))
        pos_id = e["element_type"]
        price = e["now_cost"] / 10.0

        sig = availability.get(pid) or news_mod.Availability(player_id=pid)
        move = moves.get(pid, {})
        ext = external.get(pid)

        p_start_base, avg_start_mins = _minutes_profile(
            hist_past, hist_curr, price, ext,
            is_new_signing=bool(move.get("is_new_signing")),
            prior_matches=cfg.start_prior_matches,
            new_signing_pull=cfg.new_signing_start_pull,
        )
        rates = _rates_from_history(hist_past, hist_curr, pos_id, cfg)
        club_matches = matches_since_joining(hist_curr, move.get("joined"))
        settle = transfers_mod.settling_multiplier(move, club_matches, cfg.squad)

        # A signing from abroad has no Premier League record, so the empirical-Bayes
        # step has nothing but the price prior to shrink toward. Hand it the player's
        # foreign-league rates as (deliberately weak) past-season evidence instead.
        past_mins = rates["past_mins"]
        past_goals_p90 = rates["past_goals_p90"]
        past_assists_p90 = rates["past_assists_p90"]
        ext_league = ""
        if ext and past_mins <= 0:
            ext_league = ext.get("league", "")
            past_mins = float(ext["minutes"])
            past_goals_p90 = ext["goal_rate"]
            past_assists_p90 = ext["assist_rate"]

        rows.append(
            {
                "id": pid,
                "name": e["web_name"],
                "team_id": e["team"],
                "team": teams[e["team"]]["short_name"],
                "pos_id": pos_id,
                "pos": POS[pos_id],
                "price": price,
                "status": e.get("status", "a"),
                "selected_by": float(e.get("selected_by_percent", 0) or 0),
                "ep_next": float(e.get("ep_next", 0) or 0),
                "p_start_raw": p_start_base,
                "avg_start_mins": avg_start_mins,
                "settle": settle,
                "club_matches": club_matches,
                "curr_mins": rates["curr_mins"],
                "past_mins": past_mins,
                "curr_goals_p90": rates["curr_goals_p90"],
                "curr_assists_p90": rates["curr_assists_p90"],
                "curr_saves_p90": rates["curr_saves_p90"],
                "past_pts_p90": rates["past_pts_p90"],
                "past_goals_p90": past_goals_p90,
                "past_assists_p90": past_assists_p90,
                "past_saves_p90": rates["past_saves_p90"],
                "news": e.get("news", ""),
                # --- availability & transfer context -------------------------
                "avail": float(sig.curve.get(start_gw, 1.0)),
                "news_category": sig.category,
                "news_summary": sig.summary(),
                "return_gw": sig.return_gw,
                "web_headline": sig.web_headline,
                "web_category": sig.web_category or "",
                "days_at_club": move.get("days_at_club"),
                "is_new_signing": bool(move.get("is_new_signing")),
                "window": move.get("window", transfers_mod.SETTLED),
                "external_league": ext_league,
                "external_minutes": float(ext["minutes"]) if ext else 0.0,
            }
        )

    df = pd.DataFrame(rows)
    # Keep the historic column name available; some callers and the tests read it.
    df["avail_next"] = df["avail"]

    # ----------------------------------------------------------------------- Priors & Empirical Bayes
    # Position baseline priors as a function of price
    df["prior_goal_rate"] = np.nan
    df["prior_assist_rate"] = np.nan
    df["prior_pts_p90"] = np.nan

    for pos_id in POS:
        m = df["pos_id"] == pos_id
        # Goals prior
        known_g = df[m & df["curr_goals_p90"].notna()]
        if len(known_g) < 10:
            known_g = df[m & df["past_goals_p90"].notna()]

        if len(known_g) >= 10:
            try:
                g_vals = known_g["curr_goals_p90"].fillna(known_g["past_goals_p90"]).clip(0, 1.5)
                coef_g = np.polyfit(np.log(known_g["price"]), g_vals, 1)
                df.loc[m, "prior_goal_rate"] = np.polyval(coef_g, np.log(df.loc[m, "price"])).clip(0.01, 1.2)
            except Exception:
                df.loc[m, "prior_goal_rate"] = 0.05 if pos_id <= 2 else 0.22
        else:
            df.loc[m, "prior_goal_rate"] = 0.04 if pos_id <= 2 else (0.20 if pos_id == 3 else 0.35)

        # Assists prior
        known_a = df[m & df["curr_assists_p90"].notna()]
        if len(known_a) < 10:
            known_a = df[m & df["past_assists_p90"].notna()]
        if len(known_a) >= 10:
            try:
                a_vals = known_a["curr_assists_p90"].fillna(known_a["past_assists_p90"]).clip(0, 1.2)
                coef_a = np.polyfit(np.log(known_a["price"]), a_vals, 1)
                df.loc[m, "prior_assist_rate"] = np.polyval(coef_a, np.log(df.loc[m, "price"])).clip(0.01, 0.8)
            except Exception:
                df.loc[m, "prior_assist_rate"] = 0.08 if pos_id <= 2 else 0.18
        else:
            df.loc[m, "prior_assist_rate"] = 0.06 if pos_id <= 2 else (0.18 if pos_id == 3 else 0.15)

        # Total points p90 prior (for backup calibration)
        known_pts = df[m & df["past_pts_p90"].notna()]
        if len(known_pts) >= 10:
            try:
                coef_pts = np.polyfit(np.log(known_pts["price"]), known_pts["past_pts_p90"].clip(1, 12), 1)
                df.loc[m, "prior_pts_p90"] = np.polyval(coef_pts, np.log(df.loc[m, "price"])).clip(1.5, 9.0)
            except Exception:
                df.loc[m, "prior_pts_p90"] = 3.0 + (pos_id * 0.4)
        else:
            df.loc[m, "prior_pts_p90"] = 3.0 + (pos_id * 0.4)

    df["prior_goal_rate"] = df["prior_goal_rate"].fillna(0.10).clip(lower=0.01)
    df["prior_assist_rate"] = df["prior_assist_rate"].fillna(0.10).clip(lower=0.01)
    df["prior_pts_p90"] = df["prior_pts_p90"].fillna(3.5).clip(lower=1.0)

    # Multi-tier shrinkage. Foreign-league minutes are discounted here rather than at
    # the source, so the rates stay readable while the *confidence* in them drops.
    trust = np.where(df["external_league"].astype(bool) & (df["curr_mins"] <= 0), 0.55, 1.0)
    curr_matches = (df["curr_mins"] / 90.0).fillna(0.0)
    past_matches = (df["past_mins"] / 90.0).fillna(0.0) * trust

    w_curr = curr_matches / (curr_matches + cfg.shrink_inseason)
    w_past = (1.0 - w_curr) * (past_matches / (past_matches + cfg.shrink_matches))
    w_prior = 1.0 - w_curr - w_past

    # Blended goal and assist rates per 90
    df["goal_rate"] = (
        w_curr * df["curr_goals_p90"].fillna(df["prior_goal_rate"])
        + w_past * df["past_goals_p90"].fillna(df["prior_goal_rate"])
        + w_prior * df["prior_goal_rate"]
    ).fillna(df["prior_goal_rate"]).clip(0.01, 1.5)

    df["assist_rate"] = (
        w_curr * df["curr_assists_p90"].fillna(df["prior_assist_rate"])
        + w_past * df["past_assists_p90"].fillna(df["prior_assist_rate"])
        + w_prior * df["prior_assist_rate"]
    ).fillna(df["prior_assist_rate"]).clip(0.01, 1.2)

    df["saves_rate"] = np.where(
        df["pos_id"] == 1,
        (w_curr * df["curr_saves_p90"].fillna(3.0) + (1 - w_curr) * df["past_saves_p90"].fillna(3.0)).clip(1.5, 5.5),
        0.0,
    )

    ftab = fixture_table(fixtures_raw, start_gw, cfg.horizon)

    # ----------------------------------------------------------------------- Depth chart per gameweek
    # Competition for a starting place is settled separately in each gameweek, because
    # availability changes: while the first choice is injured his deputy starts, and
    # when he comes back the deputy's minutes go away again.
    start_probs: dict[int, pd.Series] = {}
    minutes: dict[int, pd.Series] = {}
    for gw in gws:
        df["_avail_gw"] = [
            float((availability.get(int(pid)).curve.get(gw, 1.0)) if availability.get(int(pid)) else 1.0)
            for pid in df["id"]
        ]
        ps = transfers_mod.depth_chart(df, cfg.squad, "p_start_raw", "_avail_gw")
        start_probs[gw] = ps
        minutes[gw] = pd.Series(
            [
                expected_minutes(p, a, m, s)
                for p, a, m, s in zip(ps, df["_avail_gw"], df["avg_start_mins"], df["settle"])
            ],
            index=df.index,
        )
    df.drop(columns=["_avail_gw"], inplace=True)

    # Headline columns describe the coming gameweek.
    df["p_start"] = start_probs[start_gw]
    df["xmins"] = minutes[start_gw]

    # ----------------------------------------------------------------------- Component Projection
    proj: dict[int, dict[int, float]] = {}

    for idx, r in enumerate(df.itertuples()):
        team = teams[r.team_id]
        per_gw: dict[int, float] = {}

        for gw in gws:
            xmins_gw = float(minutes[gw].iloc[idx])
            p_start_gw = float(start_probs[gw].iloc[idx])
            total_pts = 0.0

            # Someone who cannot play scores nothing, full stop. Stated once here
            # rather than relied upon to fall out of every component below.
            if xmins_gw <= 0.0:
                per_gw[gw] = 0.0
                continue

            for fx in ftab.get(r.team_id, {}).get(gw, []):
                opp = teams[fx["opponent"]]
                home = fx["home"]

                lam, mu, p_cs, p_gc2 = _match_poisson_expectation(
                    team, opp, home, avg_att_h, avg_att_a, avg_def_h, avg_def_a, cfg
                )

                # Minutes probabilities
                p_play_60 = float(np.clip(p_start_gw * 0.94, 0.0, 1.0))
                p_play_sub = float(np.clip((xmins_gw - p_play_60 * 80.0) / 25.0, 0.0, 1.0)) if p_play_60 < 0.95 else 0.0
                p_play_sub = max(0.0, min(1.0 - p_play_60, p_play_sub))

                # 1. Appearance Points
                pts_app = 2.0 * p_play_60 + 1.0 * p_play_sub

                # 2. Attacking Returns (scaled by team goal expectation ratio)
                att_scale = (lam / avg_league_goals) ** cfg.strength_exp
                exp_goals = r.goal_rate * (xmins_gw / 90.0) * att_scale
                exp_assists = r.assist_rate * (xmins_gw / 90.0) * att_scale
                pts_attack = (exp_goals * GOAL_PTS[r.pos_id]) + (exp_assists * ASSIST_PTS)

                # 3. Clean Sheet Points
                pts_cs = CS_PTS[r.pos_id] * p_cs * p_play_60

                # 4. Goals Conceded Penalty (for DEF and GKP with 60+ mins)
                pts_concede = 0.0
                if r.pos_id in DEFENSIVE:
                    pts_concede = -1.0 * p_gc2 * p_play_60

                # 5. Saves Points (for GKP)
                pts_saves = 0.0
                if r.pos_id == 1:
                    pts_saves = (r.saves_rate / 3.0) * (xmins_gw / 90.0)

                # 6. Yellow Card Penalty
                pts_cards = -0.15 * (xmins_gw / 90.0)

                # 7. Bonus Points Model (BPS regression on baseline + returns).
                # Every term is gated on actually being on the pitch: a clean sheet
                # earns a midfielder nothing if he did not play in it.
                cs_bonus = (0.25 if r.pos_id in DEFENSIVE else 0.05) * p_cs * p_play_60
                pts_bonus = (
                    0.38 * exp_goals
                    + 0.20 * exp_assists
                    + cs_bonus
                    + 0.08 * p_play_60
                )

                match_pts = pts_app + pts_attack + pts_cs + pts_concede + pts_saves + pts_cards + pts_bonus
                if np.isfinite(match_pts) and match_pts > 0:
                    total_pts += match_pts

            per_gw[gw] = float(total_pts) if np.isfinite(total_pts) else 0.0

        # Blend FPL official ep_next for start_gw if enabled. Only where we agree the
        # player is available -- ep_next lags badly on players who have just been ruled
        # out, and blending it back in would undo the news we just read.
        if cfg.ep_blend > 0 and r.ep_next > 0 and per_gw.get(start_gw, 0) > 0 and r.avail > 0.5:
            per_gw[start_gw] = (1.0 - cfg.ep_blend) * per_gw[start_gw] + cfg.ep_blend * r.ep_next

        proj[r.id] = {g: (float(v) if np.isfinite(v) else 0.0) for g, v in per_gw.items()}

    # Apply manual overrides if specified
    for pid, val in cfg.overrides.items():
        if pid not in proj:
            continue
        if isinstance(val, dict):
            proj[pid].update({int(g): float(v) for g, v in val.items()})
        else:
            proj[pid] = {g: float(val) for g in gws}

    df["proj_next"] = df["id"].map(lambda i: proj[i].get(start_gw, 0.0)).fillna(0.0)
    df["proj_horizon"] = df["id"].map(lambda i: sum(proj[i].values())).fillna(0.0)
    df["value"] = np.where(df["price"] > 0, df["proj_horizon"] / df["price"], 0.0)
    return df, proj


def load_overrides(path: str, df: pd.DataFrame | None = None) -> dict:
    """CSV with columns: player (id or web_name), points, and optionally gw."""
    raw = pd.read_csv(path)
    name_to_id = {}
    if df is not None:
        name_to_id = {str(n).lower(): i for n, i in zip(df["name"], df["id"])}
    out: dict = {}
    for r in raw.itertuples():
        key = str(r.player)
        pid = int(key) if key.isdigit() else name_to_id.get(key.lower())
        if pid is None:
            print(f"  ! override skipped, unknown player: {key}")
            continue
        if "gw" in raw.columns and not pd.isna(r.gw):
            out.setdefault(pid, {})[int(r.gw)] = float(r.points)
        else:
            out[pid] = float(r.points)
    return out
