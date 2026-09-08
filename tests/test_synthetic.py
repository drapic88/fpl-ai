"""Offline smoke test: fabricates a plausible league and checks every FPL rule holds.

Run with:  python tests/test_synthetic.py
No network needed — useful for changing the model without burning API calls.
"""

import datetime as dt
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fplai import model, optimize  # noqa: E402

random.seed(7)
NAMES = ["Haaland", "Fernandes", "Saka", "Palmer", "Isak", "Raya", "Gabriel", "Trippier"]


def fake_bootstrap(n_teams=20, per_team=25):
    teams = []
    for t in range(1, n_teams + 1):
        base = random.randint(1000, 1400)
        teams.append({
            "id": t, "short_name": f"T{t:02d}",
            "strength_attack_home": base + 60, "strength_attack_away": base,
            "strength_defence_home": base + 40, "strength_defence_away": base - 20,
            "strength_overall_home": base, "strength_overall_away": base,
        })
    elements, pid = [], 1
    for t in range(1, n_teams + 1):
        for et, count in ((1, 3), (2, 8), (3, 9), (4, 5)):
            for _ in range(count):
                elements.append({
                    "id": pid, "web_name": f"{random.choice(NAMES)}{pid}", "team": t,
                    "first_name": f"First{pid}", "second_name": f"Surname{pid}",
                    "element_type": et, "now_cost": random.randint(38, 150),
                    "status": random.choice(["a"] * 17 + ["d", "i", "u"]),
                    "chance_of_playing_next_round": None,
                    "team_join_date": random.choice(
                        ["2021-07-01"] * 8 + ["2026-07-20", "2026-01-15"]
                    ),
                    "can_select": True,
                    "selected_by_percent": round(random.uniform(0.1, 40), 1),
                    "ep_next": round(random.uniform(0, 7), 1), "news": "",
                })
                pid += 1
    # Real deadlines matter: return dates in the news are resolved against them.
    events = [
        {
            "id": g,
            "is_next": g == 1,
            "finished": False,
            "deadline_time": (dt.datetime(2026, 8, 21, 17, 30) + dt.timedelta(days=7 * (g - 1)))
            .isoformat() + "Z",
        }
        for g in range(1, 39)
    ]
    return {"teams": teams, "elements": elements, "events": events}


def fake_fixtures(n_teams=20, n_gw=10):
    out, fid = [], 1
    for gw in range(1, n_gw + 1):
        order = list(range(1, n_teams + 1))
        random.shuffle(order)
        for i in range(0, n_teams, 2):
            h, a = order[i], order[i + 1]
            out.append({
                "id": fid, "event": gw, "team_h": h, "team_a": a,
                "team_h_difficulty": random.randint(2, 5),
                "team_a_difficulty": random.randint(2, 5), "finished": False,
            })
            fid += 1
    return out


def fake_history(elements):
    hist = {}
    for e in elements:
        if random.random() < 0.15:          # newly promoted / new signing, no PL history
            hist[e["id"]] = {"past": [], "this_season": []}
            continue
        mins = random.choice([0, 400, 1200, 2200, 3000])
        ppg = 2.5 + (e["now_cost"] / 10 - 4) * 0.55 + random.gauss(0, 0.8)
        past = [{
            "season_name": "2025/26", "minutes": mins,
            "total_points": max(0, int(ppg * mins / 90)),
            "goals_scored": random.randint(0, 15) if e["element_type"] >= 3 else random.randint(0, 3),
            "assists": random.randint(0, 10),
            "clean_sheets": random.randint(0, 14) if e["element_type"] <= 2 else random.randint(0, 5),
            "saves": random.randint(30, 120) if e["element_type"] == 1 else 0,
        }]
        # In-season match logs for 3 synthetic gameweeks
        this_season = []
        for gw in range(1, 4):
            gmins = random.choice([0, 0, 75, 90, 90])
            this_season.append({
                "event": gw,
                "minutes": gmins,
                "goals_scored": 1 if (gmins > 0 and random.random() < 0.25) else 0,
                "assists": 1 if (gmins > 0 and random.random() < 0.15) else 0,
                "clean_sheets": 1 if (gmins >= 60 and random.random() < 0.35) else 0,
                "saves": random.randint(2, 6) if (e["element_type"] == 1 and gmins > 0) else 0,
                "yellow_cards": 1 if random.random() < 0.1 else 0,
                "total_points": random.randint(1, 10) if gmins > 0 else 0,
            })
        hist[e["id"]] = {"past": past, "this_season": this_season}
    return hist


def check(sol, budget=100.0):
    squad = sol["squad"]
    assert len(squad) == 15, "squad must be 15"
    counts = {}
    for r in squad:
        counts[r["pos"]] = counts.get(r["pos"], 0) + 1
    assert counts == {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}, counts
    clubs = {}
    for r in squad:
        clubs[r["team"]] = clubs.get(r["team"], 0) + 1
    assert max(clubs.values()) <= 3, f"too many from one club: {clubs}"
    assert sol["cost"] <= budget + 1e-6, f"over budget: {sol['cost']}"
    assert len(sol["xi"]) == 11 and len(sol["bench"]) == 4
    xi_pos = {}
    for r in sol["xi"]:
        xi_pos[r["pos"]] = xi_pos.get(r["pos"], 0) + 1
    assert xi_pos.get("GKP") == 1
    assert 3 <= xi_pos.get("DEF", 0) <= 5
    assert 1 <= xi_pos.get("FWD", 0) <= 3
    assert sol["captain"] in [r["id"] for r in sol["xi"]]
    assert sol["bench"][-1]["pos"] == "GKP", "reserve keeper should sit last on the bench"


def main():
    bs = fake_bootstrap()
    fx = fake_fixtures()
    hist = fake_history(bs["elements"])
    cfg = model.ModelConfig(horizon=5)
    df, proj = model.build(bs, fx, 1, cfg, hist)
    print(f"built projections for {len(df)} players")
    assert df["proj_horizon"].max() > 0
    # An injured player scores nothing in the gameweek he is injured for...
    assert (df.loc[df["status"] == "i", "proj_next"] == 0).all(),         "injured players must project zero for the coming gameweek"
    # ...but must not be written off for the rest of the horizon. Before the
    # per-gameweek availability curve, one flag zeroed a player for every future
    # gameweek too, which is why the model never bought anyone due back soon.
    injured = df[df["status"] == "i"]
    if len(injured):
        assert injured["proj_horizon"].max() > 0,             "an injury must not zero out gameweeks after the expected return"
    # A departed player is gone for good, in every gameweek.
    assert (df.loc[df["status"] == "u", "proj_horizon"] == 0).all(),         "players who have left the league must project zero throughout"

    gws = [1, 2, 3, 4, 5]
    opt = optimize.SolveOptions(budget=100.0)
    sol = optimize.pick_squad(df, proj, gws, opt)
    check(sol)
    print(f"squad ok: £{sol['cost']}m, projected GW1 {sol['proj_next']} pts")

    # A tighter budget must never produce a better projection than a looser one.
    poor = optimize.pick_squad(df, proj, gws, optimize.SolveOptions(budget=85.0))
    check(poor, 85.0)
    # Compare on the objective the solver maximises -- the decayed total across the
    # whole horizon -- not on gameweek one alone. A bigger budget's feasible set
    # contains the smaller one's, so the optimum cannot get worse; but a squad built
    # for five gameweeks can quite legitimately score less in the first of them.
    assert poor["objective"] <= sol["objective"] + 1e-6
    assert (sum(r["proj_horizon"] for r in poor["squad"])
            <= sum(r["proj_horizon"] for r in sol["squad"]) + 1e-6)
    print(f"budget monotonicity ok (£85m -> {poor['proj_next']} pts)")

    # Forced inclusion must be honoured.
    forced = int(df.sort_values("price").iloc[0]["id"])
    incl = optimize.pick_squad(df, proj, gws, optimize.SolveOptions(budget=100.0, include=(forced,)))
    assert forced in [r["id"] for r in incl["squad"]]
    print("include/exclude ok")

    # Transfers: from the cheap squad, with money in the bank, it should improve things.
    current = [r["id"] for r in poor["squad"]]
    selling = {r["id"]: r["price"] for r in poor["squad"]}
    plan = optimize.plan_transfers(
        df, proj, gws, current, selling,
        optimize.SolveOptions(bank=15.0, free_transfers=1, max_transfers=2),
    )
    check(plan)
    assert plan["transfers"] <= 2
    assert len(plan["in"]) == len(plan["out"])
    assert "future_transfers" in plan
    print(f"transfer plan ok: {plan['transfers']} move(s), hit -{int(plan['hit'])}, "
          f"projected {plan['proj_next']} pts")

    # A wildcard is not rationed by free transfers. The FT-banking constraint used
    # to subtract the wildcard week's moves from the next week's stock, which -- with
    # hits pinned to zero and ft floored at 1 -- silently capped a wildcard at
    # free_transfers moves, so `--wildcard` reproduced the ordinary plan.
    wc_opt = optimize.SolveOptions(bank=15.0, free_transfers=1, max_transfers=1,
                                   wildcard=True)
    wc = optimize.plan_transfers(df, proj, gws, current, selling, wc_opt)
    check(wc)
    assert wc["hit"] == 0, "a wildcard week must never charge a points hit"
    assert wc["transfers"] > 1, (
        "a wildcard must be free to make more than free_transfers moves, got "
        f"{wc['transfers']}"
    )
    # ...and it must be at least as good as the same week played without the chip.
    plain = optimize.plan_transfers(
        df, proj, gws, current, selling,
        optimize.SolveOptions(bank=15.0, free_transfers=1, max_transfers=1),
    )
    assert wc["objective"] >= plain["objective"] - 1e-6, (
        "a wildcard's feasible set contains the one-move plan's, so it cannot score worse"
    )
    print(f"wildcard ok: {wc['transfers']} move(s), hit -{int(wc['hit'])}, "
          f"objective {wc['objective']:.2f} vs plain {plain['objective']:.2f}")

    # With zero budget movement and no free transfers, doing nothing must be allowed.
    idle = optimize.plan_transfers(
        df, proj, gws, current, selling,
        optimize.SolveOptions(bank=0.0, free_transfers=0, max_transfers=1),
    )
    check(idle)
    print(f"idle case ok: {idle['transfers']} transfer(s) recommended")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()

