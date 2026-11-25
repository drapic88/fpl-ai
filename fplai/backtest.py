"""Backtest: replay past gameweeks and score the projections against what happened.

The question asked of every gameweek is the one that decides transfers and the
armband -- *who will score the most this week* -- not who has the best season
average. So the headline metrics are about the top of the ranking:

* `pick_pts`    mean actual points of the model's top five per position
* `captain_pts` actual points of the model's number one
* `hit10`       share of the model's top ten that finished in the real top ten
* `haul_prec`   share of the model's top twenty that hauled (10+ points)
* `spearman`    rank agreement over the whole pool, for completeness

Each gameweek is projected point-in-time: match logs, prices and team strength
only from before that gameweek. Two things cannot be replayed and are left out
of every predictor alike: team news (everyone is assumed available, so the
model's minutes come from its own record alone) and FPL's fixture-difficulty
ratings (team strength is rebuilt from rolling xG instead).

Data: past seasons from the per-gameweek CSVs of the public
vaastav/Fantasy-Premier-League repository (`merged_gw.csv`), the current season
from the FPL API's per-player histories.
"""

from __future__ import annotations

import copy
import datetime as dt
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import api, model

HISTORY_DIR = api.CACHE_DIR / "history"
POS_ID = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
STAT_COLS = [
    "minutes", "goals_scored", "assists", "clean_sheets", "saves", "yellow_cards",
    "total_points", "expected_goals", "expected_assists", "defensive_contribution",
    "starts", "bonus",
]
HAUL = 10


@dataclass
class Season:
    """One season of per-fixture player rows plus each player's earlier seasons."""
    name: str
    rows: pd.DataFrame                       # one row per player per fixture
    past: dict[int, list[dict]]              # element -> season totals, oldest first
    team_names: dict[int, str]               # team id -> name
    team_prior: dict[str, tuple[float, float]] = field(default_factory=dict)  # name -> (xG for, xG against) per match


# --------------------------------------------------------------------------- loading


def _fixture_teams(rows: pd.DataFrame) -> dict[int, tuple[int, int]]:
    """fixture -> (home team id, away team id), read off the opponent columns."""
    out = {}
    for fx, g in rows.groupby("fixture"):
        home = g[g["was_home"]]
        away = g[~g["was_home"]]
        if home.empty or away.empty:
            continue
        out[int(fx)] = (int(away["opponent_team"].iloc[0]), int(home["opponent_team"].iloc[0]))
    return out


def season_totals(rows: pd.DataFrame, season_name: str) -> dict[str, dict]:
    """Player name -> one `history_past`-style row for that season."""
    cols = [c for c in STAT_COLS if c in rows]
    agg = rows.groupby("name")[cols].sum()
    out = {}
    for name, r in agg.iterrows():
        d = {c: float(r[c]) for c in cols}
        d["season_name"] = season_name
        out[str(name)] = d
    return out


def team_match_xg(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per team per fixture: GW, kickoff, xG for and against."""
    g = rows.groupby(["fixture", "team_id"], as_index=False).agg(
        gw=("GW", "first"), kickoff=("kickoff_time", "first"), xg=("expected_goals", "sum"),
        opp=("opponent_team", "first"),
    )
    against = g[["fixture", "team_id", "xg"]].rename(columns={"team_id": "opp", "xg": "xga"})
    return g.merge(against, on=["fixture", "opp"], how="left").fillna({"xga": 0.0})


def team_season_prior(rows: pd.DataFrame, team_names: dict[int, str]) -> dict[str, tuple[float, float]]:
    """Team name -> (xG for, xG against) per match over a whole season."""
    t = team_match_xg(rows)
    m = t.groupby("team_id")[["xg", "xga"]].mean()
    return {team_names[int(i)]: (float(r["xg"]), float(r["xga"])) for i, r in m.iterrows()}


def load_vaastav(season: str, previous: list[Season] | None = None,
                 directory: Path = HISTORY_DIR) -> Season:
    """Read one season's `merged_gw.csv`; `previous` supplies the players' past seasons."""
    df = pd.read_csv(directory / f"merged_gw_{season}.csv")
    df = df[df["position"].isin(POS_ID)].copy()
    df["pos_id"] = df["position"].map(POS_ID)
    df["was_home"] = df["was_home"].astype(str).str.lower().isin(["true", "1"])
    for c in STAT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0) if c in df else 0.0
    df["xP"] = pd.to_numeric(df.get("xP"), errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    fx = _fixture_teams(df)
    df["team_id"] = [fx.get(int(f), (0, 0))[0 if h else 1] for f, h in zip(df["fixture"], df["was_home"])]
    df = df[df["team_id"] > 0]
    team_names = {}
    for name, tid in df.groupby("team")["team_id"].agg(lambda s: s.mode().iloc[0]).items():
        team_names[int(tid)] = str(name)

    past: dict[int, list[dict]] = {}
    totals = [season_totals(p.rows, p.name) for p in (previous or [])]
    for el, name in df.groupby("element")["name"].first().items():
        seasons = [t[name] for t in totals if name in t]
        if seasons:
            past[int(el)] = seasons
    prior = team_season_prior(previous[-1].rows, previous[-1].team_names) if previous else {}
    return Season(season, df.reset_index(drop=True), past, team_names, prior)


def load_api_season(name: str, bootstrap: dict, fixtures: list[dict], histories: dict[int, dict],
                    previous: Season | None = None) -> Season:
    """The current season from the FPL API, shaped like a `load_vaastav` season."""
    fx = {int(f["id"]): (int(f["team_h"]), int(f["team_a"])) for f in fixtures}
    els = {int(e["id"]): e for e in bootstrap["elements"]}
    teams = {int(t["id"]): t["name"] for t in bootstrap["teams"]}
    rows = []
    for pid, h in histories.items():
        e = els.get(int(pid))
        if not e:
            continue
        for m in h.get("this_season", []):
            if int(m.get("fixture", 0)) not in fx:
                continue
            home, away = fx[int(m["fixture"])]
            r = {c: float(m.get(c) or 0) for c in STAT_COLS}
            r.update({
                "element": int(pid),
                "name": f"{e['first_name']} {e['second_name']}",
                "pos_id": int(e["element_type"]),
                "team_id": home if m["was_home"] else away,
                "team": teams[home if m["was_home"] else away],
                "fixture": int(m["fixture"]),
                "opponent_team": int(m["opponent_team"]),
                "was_home": bool(m["was_home"]),
                "kickoff_time": m.get("kickoff_time"),
                "GW": int(m["round"]),
                "value": float(m.get("value") or e["now_cost"]),
                "xP": np.nan,
            })
            rows.append(r)
    df = pd.DataFrame(rows)
    past = {int(pid): [
        {**s, "season_name": s.get("season_name")} for s in h.get("past", [])
    ] for pid, h in histories.items() if h.get("past")}
    prior = team_season_prior(previous.rows, previous.team_names) if previous else {}
    return Season(name, df, past, teams, prior)


# --------------------------------------------------------------------------- point-in-time inputs


def team_strengths(season: Season, gw: int, prior_matches: float = 8.0,
                   decay: float = 0.9) -> dict[int, dict]:
    """FPL-style strength ratings rebuilt from xG before `gw`.

    The model reads a rating as `(rating / league average) ** strength_exp`, with
    strength_exp at 0.5, so ratings are set to the square of the xG ratio: a team
    creating 20% more than average then scores 20% more in the Poisson step.
    """
    t = team_match_xg(season.rows[season.rows["GW"] < gw])
    league_xg = (
        float(np.mean([v[0] for v in season.team_prior.values()])) if season.team_prior else 1.35
    )
    out = {}
    for tid, name in season.team_names.items():
        p_xg, p_xga = season.team_prior.get(name, (0.8 * league_xg, 1.2 * league_xg))
        g = t[t["team_id"] == tid].sort_values("gw")
        w = decay ** np.arange(len(g))[::-1] if len(g) else np.array([])
        xg = (prior_matches * p_xg + float((w * g["xg"]).sum())) / (prior_matches + float(w.sum()))
        xga = (prior_matches * p_xga + float((w * g["xga"]).sum())) / (prior_matches + float(w.sum()))
        att = 1000.0 * (xg / league_xg) ** 2
        dfn = 1000.0 * (league_xg / max(0.3, xga)) ** 2
        out[tid] = {
            "id": tid, "name": name, "short_name": name[:3].upper(),
            "strength_attack_home": att, "strength_attack_away": att,
            "strength_defence_home": dfn, "strength_defence_away": dfn,
            "strength_overall_home": 1000.0, "strength_overall_away": 1000.0,
        }
    return out


def inputs_at(season: Season, gw: int) -> tuple[dict, list[dict], dict[int, dict], dt.date]:
    """(bootstrap, fixtures, histories, today) as they stood before `gw`."""
    rows = season.rows
    now = rows[rows["GW"] == gw]
    before = rows[rows["GW"] < gw].sort_values("kickoff_time")
    kick = pd.to_datetime(now["kickoff_time"], utc=True, errors="coerce").min()
    today = (kick - pd.Timedelta(hours=2)).date() if pd.notna(kick) else dt.date.today()

    elements = []
    for el, g in now.groupby("element"):
        r = g.iloc[0]
        xp = g["xP"].iloc[0]
        elements.append({
            "id": int(el), "web_name": r["name"], "first_name": "", "second_name": r["name"],
            "team": int(r["team_id"]), "element_type": int(r["pos_id"]),
            "now_cost": int(r["value"]), "status": "a", "news": "",
            "chance_of_playing_next_round": None, "team_join_date": None,
            "selected_by_percent": 0.0, "ep_next": float(xp) if pd.notna(xp) else 0.0,
        })
    deadline = (kick - pd.Timedelta(hours=2)) if pd.notna(kick) else pd.Timestamp.now(tz="UTC")
    events = [{"id": gw, "is_next": True, "finished": False,
               "deadline_time": deadline.strftime("%Y-%m-%dT%H:%M:%SZ")}]
    bs = {"teams": list(team_strengths(season, gw).values()), "elements": elements, "events": events}

    fixtures = []
    for fx, g in now.groupby("fixture"):
        h = g[g["was_home"]]
        a = g[~g["was_home"]]
        if h.empty or a.empty:
            continue
        fixtures.append({"id": int(fx), "event": gw, "team_h": int(h["team_id"].iloc[0]),
                         "team_a": int(a["team_id"].iloc[0]),
                         "team_h_difficulty": 3, "team_a_difficulty": 3})

    keep = ["minutes", "goals_scored", "assists", "clean_sheets", "saves", "yellow_cards",
            "expected_goals", "expected_assists", "kickoff_time", "total_points"]
    if season.rows["defensive_contribution"].sum() > 0:
        keep.append("defensive_contribution")
    hist: dict[int, dict] = {}
    by_el = {int(el): g[keep].to_dict(orient="records") for el, g in before.groupby("element")}
    for e in elements:
        hist[e["id"]] = {"past": season.past.get(e["id"], []), "this_season": by_el.get(e["id"], [])}
    return bs, fixtures, hist, today


def project(season: Season, gw: int, cfg: model.ModelConfig) -> pd.DataFrame:
    """The model's view of `gw`, from data before it."""
    bs, fixtures, hist, today = inputs_at(season, gw)
    c = copy.deepcopy(cfg)
    c.horizon = 1
    df, _ = model.build(bs, fixtures, gw, c, hist, moves={}, external={}, today=today)
    return df


def actual_points(season: Season, gw: int) -> pd.Series:
    return season.rows[season.rows["GW"] == gw].groupby("element")["total_points"].sum()


def pool(season: Season, gw: int, recent: int = 3) -> set[int]:
    """Players with a match this week who played in at least one of their last `recent`."""
    rows = season.rows
    now = set(rows.loc[rows["GW"] == gw, "element"])
    before = rows[rows["GW"] < gw].sort_values("kickoff_time")
    played = before.groupby("element").tail(recent)
    active = set(played.loc[played["minutes"] > 0, "element"])
    return {int(e) for e in now & active}


# --------------------------------------------------------------------------- scoring


def evaluate(score: pd.Series, actual: pd.Series, pos: pd.Series) -> dict[str, float]:
    """Top-of-the-ranking metrics for one predictor in one gameweek."""
    d = pd.DataFrame({"s": score, "a": actual, "p": pos}).dropna()
    if len(d) < 30:
        return {}
    d = d.sample(frac=1.0, random_state=0)            # ties broken at random, not by id
    top = d.sort_values("s", ascending=False)
    tenth = d["a"].nlargest(10).min()
    top5 = top.groupby("p").head(5)
    hauls = int((d["a"] >= HAUL).sum())
    return {
        "pick_pts": float(top5["a"].mean()),
        "captain_pts": float(top["a"].iloc[0]),
        "top3_pts": float(top["a"].iloc[:3].mean()),
        "hit10": float((top["a"].iloc[:10] >= tenth).mean()),
        "haul_prec": float((top["a"].iloc[:20] >= HAUL).mean()),
        "haul_recall": float((top["a"].iloc[:20] >= HAUL).sum() / hauls) if hauls else np.nan,
        "spearman": float(d["s"].rank().corr(d["a"].rank())),   # no scipy needed
        "oracle_pick_pts": float(d.sort_values("a", ascending=False).groupby("p").head(5)["a"].mean()),
    }


def baselines(season: Season, gw: int, n: int = 4) -> dict[str, pd.Series]:
    """Naive rankings to measure the model against: recent form and price."""
    before = season.rows[(season.rows["GW"] < gw) & (season.rows["minutes"] > 0)]
    form = before.sort_values("kickoff_time").groupby("element").tail(n)
    now = season.rows[season.rows["GW"] == gw].groupby("element")["value"].first()
    return {
        "form4": form.groupby("element")["total_points"].mean(),
        "price": now.astype(float),
    }


def predictors(df: pd.DataFrame, extra: dict[str, pd.Series] | None = None) -> dict[str, pd.Series]:
    """Named ranking scores built from one projection frame.

    FPL's own `xP` from the historical CSVs is deliberately not among them: it
    is recorded after the gameweek and already contains its points (it tracks
    the points of that gameweek at r=0.48 and of the next one at only 0.19), so
    any comparison with it -- or any `ep_blend` tuned on it -- would be rigged.
    """
    s = df.set_index("id")
    out = {"mean": s["proj_next"], "p_haul": s["p_haul"]}
    for k, v in (extra or {}).items():
        out[k] = v
    return out


def run(season: Season, configs: dict[str, model.ModelConfig], gws: list[int] | None = None,
        extra: dict[int, dict[str, pd.Series]] | None = None, verbose: bool = True) -> pd.DataFrame:
    """Per-gameweek metrics for every config and every predictor it yields."""
    gws = gws or sorted(int(g) for g in season.rows["GW"].unique() if g >= 2)
    out = []
    for gw in gws:
        act = actual_points(season, gw)
        members = pool(season, gw)
        if len(members) < 30:
            continue
        for cname, cfg in configs.items():
            df = project(season, gw, cfg)
            df = df[df["id"].isin(members)]
            pos = df.set_index("id")["pos"]
            first = cname == next(iter(configs))
            more = dict((extra or {}).get(gw, {}))
            if first:
                more.update(baselines(season, gw))
            for pname, score in predictors(df, more).items():
                # A score missing for a player (unowned, say) counts as zero, not as absent:
                # dropping him would shrink the pool and flatter that predictor.
                m = evaluate(score.reindex(pos.index).fillna(0.0), act.reindex(pos.index), pos)
                if m:
                    out.append({"season": season.name, "gw": gw, "config": cname,
                                "predictor": pname, **m})
        if verbose:
            print(f"  {season.name} GW{gw} done")
    return pd.DataFrame(out)


def summary(results: pd.DataFrame) -> pd.DataFrame:
    """Mean of every metric per season, config and predictor."""
    metrics = ["pick_pts", "captain_pts", "top3_pts", "hit10", "haul_prec", "haul_recall",
               "spearman", "oracle_pick_pts"]
    return (results.groupby(["season", "config", "predictor"])[metrics].mean()
            .round(3).reset_index())


def tune(season: Season, base: model.ModelConfig, grid: dict[str, list], gws: list[int] | None = None,
         metric: str = "pick_pts", passes: int = 2, verbose: bool = True) -> tuple[model.ModelConfig, list[dict]]:
    """Coordinate search: one knob at a time, keeping whatever scores best on `metric`.

    Cheap and transparent rather than clever -- each step changes one number and
    the log says what it bought. Run it on one season and check the result on
    another; the tuning season always flatters the tuned config.
    """
    def score(cfg):
        r = run(season, {"c": cfg}, gws, verbose=False)
        r = r[r["predictor"] == "mean"]
        return float(r[metric].mean()), float(r["spearman"].mean())

    best = copy.deepcopy(base)
    best_score = score(best)
    log = [{"step": "start", "value": None, metric: best_score[0], "spearman": best_score[1]}]
    for _ in range(passes):
        improved = False
        for knob, values in grid.items():
            for v in values:
                if math.isclose(float(getattr(best, knob)), float(v)):
                    continue
                trial = copy.deepcopy(best)
                setattr(trial, knob, type(getattr(best, knob))(v))
                sc = score(trial)
                log.append({"step": knob, "value": v, metric: sc[0], "spearman": sc[1]})
                if verbose:
                    print(f"  {knob}={v}: {metric} {sc[0]:.3f}  spearman {sc[1]:.3f}")
                if sc > best_score:
                    best, best_score, improved = trial, sc, True
        if not improved:
            break
    return best, log
