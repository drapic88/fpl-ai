"""Offline checks for the backtest harness and the model's new components.

Run with:  python tests/test_backtest.py
No network needed: a small league is fabricated and written as a merged_gw CSV.
"""

import math
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from fplai import backtest as bt, model  # noqa: E402

N_TEAMS, GWS = 6, 8
TEAMS = [f"Team{i}" for i in range(1, N_TEAMS + 1)]
SHAPE = (("GK", 2), ("DEF", 6), ("MID", 6), ("FWD", 3))


def fake_season(seed: int = 3) -> pd.DataFrame:
    """A round-robin season with players whose quality shows in their numbers."""
    rnd = random.Random(seed)
    players, el = [], 1
    for t in range(1, N_TEAMS + 1):
        for pos, n in SHAPE:
            for k in range(n):
                players.append({"element": el, "name": f"P{el}", "team_id": t, "position": pos,
                                "quality": rnd.uniform(0.2, 1.0), "starter": k < (n + 1) // 2 + (pos == "DEF")})
                el += 1
    rows, fixture = [], 1
    for gw in range(1, GWS + 1):
        order = list(range(1, N_TEAMS + 1))
        rnd.shuffle(order)
        for h, a in zip(order[::2], order[1::2]):
            for p in players:
                if p["team_id"] not in (h, a):
                    continue
                home = p["team_id"] == h
                mins = 90 if p["starter"] and rnd.random() < 0.9 else (rnd.choice([0, 0, 20]))
                xg = p["quality"] * (0.5 if p["position"] == "FWD" else 0.25 if p["position"] == "MID" else 0.05) * mins / 90
                goals = 1 if rnd.random() < xg else 0
                assists = 1 if rnd.random() < xg * 0.6 else 0
                cs = 1 if mins >= 60 and rnd.random() < 0.3 else 0
                dc = rnd.randint(4, 14) if p["position"] == "DEF" and mins >= 60 else 0
                pts = (2 if mins >= 60 else 1 if mins else 0) + goals * {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}[p["position"]] + 3 * assists
                rows.append({
                    "name": p["name"], "position": p["position"], "team": TEAMS[p["team_id"] - 1],
                    "xP": 2.0, "assists": assists, "bonus": 0, "clean_sheets": cs, "element": p["element"],
                    "expected_assists": xg * 0.6, "expected_goals": xg, "fixture": fixture,
                    "goals_scored": goals, "kickoff_time": f"2025-08-{gw:02d}T15:00:00Z",
                    "minutes": mins, "opponent_team": a if home else h, "saves": 3 if p["position"] == "GK" and mins else 0,
                    "starts": int(mins >= 60), "total_points": pts, "value": 45 + int(p["quality"] * 60),
                    "was_home": home, "yellow_cards": 0, "defensive_contribution": dc, "round": gw, "GW": gw,
                })
            fixture += 1
    return pd.DataFrame(rows)


def write(df: pd.DataFrame, directory: Path, season: str) -> None:
    df.to_csv(directory / f"merged_gw_{season}.csv", index=False)


def check_point_in_time(directory: Path) -> None:
    """Changing what happens in GW5 and later must not move the GW5 projection."""
    df = fake_season()
    write(df, directory, "2030-31")
    s = bt.load_vaastav("2030-31", directory=directory)
    before = bt.project(s, 5, model.ModelConfig(ep_blend=0.0)).set_index("id")["proj_next"]

    future = df["GW"] >= 5
    df.loc[future, ["goals_scored", "expected_goals", "total_points"]] *= 9
    write(df, directory, "2030-31")
    s2 = bt.load_vaastav("2030-31", directory=directory)
    after = bt.project(s2, 5, model.ModelConfig(ep_blend=0.0)).set_index("id")["proj_next"]
    assert (before - after).abs().max() < 1e-9, "the GW5 projection read data from GW5 or later"
    print("point-in-time ok: nothing from the target gameweek leaks into its projection")


def check_harness(directory: Path) -> None:
    write(fake_season(), directory, "2030-31")
    s = bt.load_vaastav("2030-31", directory=directory)
    assert len(s.team_names) == N_TEAMS, s.team_names
    res = bt.run(s, {"pure": model.ModelConfig(ep_blend=0.0)}, verbose=False)
    assert not res.empty and set(res["gw"]) <= set(range(2, GWS + 1))
    assert {"mean", "p_haul", "form4", "price"} <= set(res["predictor"])
    assert "fpl_xP" not in set(res["predictor"]), "the leaky historical xP must never be scored"
    summ = bt.summary(res)
    assert (summ["oracle_pick_pts"] >= summ["pick_pts"] - 1e-9).all(), "nothing beats the oracle"
    print(f"harness ok: {len(res)} rows over {res['gw'].nunique()} gameweeks")


def check_evaluate() -> None:
    rnd = random.Random(1)
    idx = list(range(100))
    actual = pd.Series([rnd.randint(0, 15) for _ in idx], index=idx, dtype=float)
    pos = pd.Series([["GKP", "DEF", "MID", "FWD"][i % 4] for i in idx], index=idx)
    perfect = bt.evaluate(actual, actual, pos)
    assert math.isclose(perfect["pick_pts"], perfect["oracle_pick_pts"])
    assert perfect["hit10"] == 1.0 and perfect["captain_pts"] == actual.max()
    assert perfect["spearman"] > 0.999
    reverse = bt.evaluate(-actual, actual, pos)
    assert reverse["pick_pts"] < perfect["pick_pts"] and reverse["spearman"] < -0.999
    print("evaluate ok: a perfect ranking scores the oracle, a reversed one the worst")


def check_team_strengths(directory: Path) -> None:
    df = fake_season()
    df.loc[df["team"] == "Team1", "expected_goals"] += 0.3
    write(df, directory, "2030-31")
    s = bt.load_vaastav("2030-31", directory=directory)
    t = bt.team_strengths(s, 6)
    by_name = {v["name"]: v["strength_attack_home"] for v in t.values()}
    assert by_name["Team1"] == max(by_name.values()), by_name
    print("team strength ok: the team creating the most xG rates the strongest attack")


def check_haul_probability() -> None:
    assert model.haul_probability([], 4) == 0.0
    assert model.haul_probability([(0.0, 0.5, 0.2, 0.3, 0.0)], 4) == 0.0
    low = model.haul_probability([(0.9, 0.2, 0.1, 0.3, 0.0)], 4)
    high = model.haul_probability([(0.9, 0.8, 0.3, 0.3, 0.0)], 4)
    double = model.haul_probability([(0.9, 0.2, 0.1, 0.3, 0.0)] * 2, 4)
    assert 0.0 < low < high <= 1.0, (low, high)
    assert double > low, "two matches must give more chances to haul than one"
    dc = model.haul_probability([(0.9, 0.4, 0.1, 0.5, 0.6)], 3)   # a goal alone is 9 for a midfielder
    no_dc = model.haul_probability([(0.9, 0.4, 0.1, 0.5, 0.0)], 3)
    assert dc > no_dc
    print(f"haul probability ok: {low:.3f} -> {high:.3f} with more xG, {double:.3f} in a double")


def check_xg_and_defcon() -> None:
    cfg0 = model.ModelConfig(ep_blend=0.0)
    cfg1 = model.ModelConfig(ep_blend=0.0, xg_weight=1.0)
    unlucky = [{"minutes": 90, "goals_scored": 0, "assists": 0, "expected_goals": 0.8,
                "expected_assists": 0.2, "clean_sheets": 0, "saves": 0, "yellow_cards": 0}] * 6
    r0 = model._rates_from_history([], unlucky, 4, cfg0)
    r1 = model._rates_from_history([], unlucky, 4, cfg1)
    assert r0["curr_goals_p90"] == 0.0 and math.isclose(r1["curr_goals_p90"], 0.8)

    busy = [{"minutes": 90, "goals_scored": 0, "assists": 0, "clean_sheets": 0, "saves": 0,
             "yellow_cards": 0, "defensive_contribution": 12}] * 5
    rd = model._rates_from_history([], busy, 2, cfg0)
    assert rd["curr_dc_rate"] == 1.0 and rd["curr_dc_n"] == 5
    gk = model._rates_from_history([], busy, 1, cfg0)
    assert gk["curr_dc_rate"] is None, "goalkeepers do not earn defensive contributions"

    with tempfile.TemporaryDirectory() as d:
        write(fake_season(), Path(d), "2030-31")
        s = bt.load_vaastav("2030-31", directory=Path(d))
        off = bt.project(s, 6, cfg0).set_index("id")
        on = bt.project(s, 6, model.ModelConfig(ep_blend=0.0, defcon=True)).set_index("id")
    defs = off.index[(off["pos"] == "DEF") & (off["p_defcon"] > 0.05) & (off["proj_next"] > 0)]
    gks = off.index[off["pos"] == "GKP"]
    assert len(defs) and (on.loc[defs, "proj_next"] > off.loc[defs, "proj_next"]).all()
    assert (on.loc[gks, "proj_next"] - off.loc[gks, "proj_next"]).abs().max() < 1e-9
    print("xG weighting and defensive contributions ok; both are off by default")


def main():
    check_evaluate()
    check_haul_probability()
    check_xg_and_defcon()
    with tempfile.TemporaryDirectory() as d:
        check_point_in_time(Path(d))
        check_harness(Path(d))
        check_team_strengths(Path(d))
    print("\nALL BACKTEST CHECKS PASSED")


if __name__ == "__main__":
    main()
