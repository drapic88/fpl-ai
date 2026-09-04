"""Squad selection and transfer planning as integer programmes.

Both problems are solved exactly (branch and bound via CBC), not greedily.
Supports multi-period transfer trajectory optimization with dynamic Free Transfer
accumulation (up to 5 banked transfers under 2026/27 rules).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pulp

SQUAD_QUOTA = {1: 2, 2: 5, 3: 5, 4: 3}
XI_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
XI_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
EPS = 1e-4


@dataclass
class SolveOptions:
    budget: float = 100.0
    bench_weight: float = 0.12      # bench points are worth something, but not much
    decay: float = 0.86
    max_per_club: int = 3
    include: tuple[int, ...] = ()
    exclude: tuple[int, ...] = ()
    free_transfers: int = 1
    max_transfers: int = 15
    hit_cost: float = 4.0
    bank: float = 0.0
    wildcard: bool = False
    verbose: bool = False


# --------------------------------------------------------------------------- squad builder


def pick_squad(players: pd.DataFrame, proj: dict, gws: list[int], opt: SolveOptions) -> dict:
    """Build an optimal 15-man squad from scratch (season start or wildcard)."""
    prob = pulp.LpProblem("fpl_build", pulp.LpMaximize)
    ids = list(players["id"])
    pos = dict(zip(players["id"], players["pos_id"]))
    club = dict(zip(players["id"], players["team_id"]))
    price = dict(zip(players["id"], players["price"]))
    xmins = dict(zip(players["id"], players["xmins"]))

    squad = pulp.LpVariable.dicts("squad", ids, cat="Binary")

    # Squad structure constraints
    prob += pulp.lpSum(squad.values()) == 15
    for t, n in SQUAD_QUOTA.items():
        prob += pulp.lpSum(squad[p] for p in ids if pos[p] == t) == n
    for c in set(club.values()):
        prob += pulp.lpSum(squad[p] for p in ids if club[p] == c) <= opt.max_per_club

    # Budget constraint
    prob += pulp.lpSum(price[p] * squad[p] for p in ids) <= opt.budget

    for p in opt.include:
        if p in squad:
            prob += squad[p] == 1
    for p in opt.exclude:
        if p in squad:
            prob += squad[p] == 0

    # Lineup variables per gameweek
    obj = []
    first_start, first_cap = None, None
    for k, gw in enumerate(gws):
        w = opt.decay ** k
        start = pulp.LpVariable.dicts(f"start_{gw}", ids, cat="Binary")
        cap = pulp.LpVariable.dicts(f"cap_{gw}", ids, cat="Binary")

        for p in ids:
            prob += start[p] <= squad[p]
            prob += cap[p] <= start[p]

        prob += pulp.lpSum(start.values()) == 11
        prob += pulp.lpSum(cap.values()) == 1
        for pid_type in SQUAD_QUOTA:
            sel = [start[p] for p in ids if pos[p] == pid_type]
            prob += pulp.lpSum(sel) >= XI_MIN[pid_type]
            prob += pulp.lpSum(sel) <= XI_MAX[pid_type]

        for p in ids:
            pts = proj[p].get(gw, 0.0)
            obj.append(w * pts * (start[p] + cap[p]))
            obj.append(w * opt.bench_weight * pts * (squad[p] - start[p]))

        if k == 0:
            first_start, first_cap = start, cap

    obj += [EPS * xmins[p] * squad[p] for p in ids]
    prob += pulp.lpSum(obj)

    prob.solve(pulp.PULP_CBC_CMD(msg=1 if opt.verbose else 0))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"No feasible squad found: {pulp.LpStatus[prob.status]}")

    sol = _read_solution(players, proj, gws, squad, first_start, first_cap)
    # The decayed multi-gameweek total the solver actually maximised. `proj_next`
    # is only the first gameweek, so it is not the quantity that is guaranteed to
    # improve when constraints are relaxed -- this is.
    sol["objective"] = round(float(pulp.value(prob.objective) or 0.0), 4)
    return sol


# --------------------------------------------------------------------------- multi-period transfer planner


def plan_transfers(
    players: pd.DataFrame,
    proj: dict,
    gws: list[int],
    current: list[int],
    selling: dict[int, float],
    opt: SolveOptions,
) -> dict:
    """Multi-period transfer trajectory optimizer with dynamic FT banking and hit penalty modeling."""
    prob = pulp.LpProblem("fpl_multi_period_xfer", pulp.LpMaximize)
    ids = list(players["id"])
    pos = dict(zip(players["id"], players["pos_id"]))
    club = dict(zip(players["id"], players["team_id"]))
    price = dict(zip(players["id"], players["price"]))
    xmins = dict(zip(players["id"], players["xmins"]))
    cur = set(current)
    H = len(gws)

    # Decision variables across time periods k in 0..H-1
    squad = {}
    buy = {}
    sell = {}
    start = {}
    cap = {}
    hits = {}
    ft = {}
    n_in = {}
    bank = {}

    obj = []

    for k, gw in enumerate(gws):
        w = opt.decay ** k
        squad[k] = pulp.LpVariable.dicts(f"squad_{gw}", ids, cat="Binary")
        buy[k] = pulp.LpVariable.dicts(f"buy_{gw}", ids, cat="Binary")
        sell[k] = pulp.LpVariable.dicts(f"sell_{gw}", ids, cat="Binary")
        start[k] = pulp.LpVariable.dicts(f"start_{gw}", ids, cat="Binary")
        cap[k] = pulp.LpVariable.dicts(f"cap_{gw}", ids, cat="Binary")
        hits[k] = pulp.LpVariable(f"hits_{gw}", lowBound=0, cat="Integer")
        ft[k] = pulp.LpVariable(f"ft_{gw}", lowBound=0 if k == 0 else 1, upBound=5, cat="Integer")

        # Squad state transitions
        if k == 0:
            prob += ft[0] == max(0, opt.free_transfers)
            for p in ids:
                if p in cur:
                    prob += squad[k][p] == 1 - sell[k][p]
                    prob += buy[k][p] == 0
                else:
                    prob += squad[k][p] == buy[k][p]
                    prob += sell[k][p] == 0
            # Budget in period 0
            funds_in = pulp.lpSum(selling.get(p, price[p]) * sell[k][p] for p in ids if p in cur)
            funds_out = pulp.lpSum(price[p] * buy[k][p] for p in ids if p not in cur)
            bank[0] = opt.bank + funds_in - funds_out
            prob += bank[0] >= 0
        else:
            for p in ids:
                prob += squad[k][p] == squad[k - 1][p] + buy[k][p] - sell[k][p]
                prob += buy[k][p] <= 1 - squad[k - 1][p]
                prob += sell[k][p] <= squad[k - 1][p]
            # Budget in period k
            funds_in = pulp.lpSum(price[p] * sell[k][p] for p in ids)
            funds_out = pulp.lpSum(price[p] * buy[k][p] for p in ids)
            bank[k] = bank[k - 1] + funds_in - funds_out
            prob += bank[k] >= 0

            # FT banking accumulation (2026/27 rule: up to 5 FTs can be saved)
            prob += ft[k] <= ft[k - 1] - n_in[k - 1] + hits[k - 1] + 1

        # Transfer count & limits
        n_in[k] = pulp.lpSum(buy[k].values())
        n_out = pulp.lpSum(sell[k].values())
        prob += n_in[k] == n_out
        prob += n_in[k] <= (15 if (k == 0 and opt.wildcard) else opt.max_transfers)

        # Hits calculation
        if k == 0 and opt.wildcard:
            prob += hits[k] == 0
        else:
            prob += hits[k] >= n_in[k] - ft[k]

        # 15-man squad constraints
        prob += pulp.lpSum(squad[k].values()) == 15
        for t, n in SQUAD_QUOTA.items():
            prob += pulp.lpSum(squad[k][p] for p in ids if pos[p] == t) == n
        for c in set(club.values()):
            prob += pulp.lpSum(squad[k][p] for p in ids if club[p] == c) <= opt.max_per_club

        # Include/Exclude constraints for GW0
        if k == 0:
            for p in opt.include:
                if p in squad[k]:
                    prob += squad[k][p] == 1
            for p in opt.exclude:
                if p in squad[k]:
                    prob += squad[k][p] == 0

        # Lineup & captaincy
        for p in ids:
            prob += start[k][p] <= squad[k][p]
            prob += cap[k][p] <= start[k][p]

        prob += pulp.lpSum(start[k].values()) == 11
        prob += pulp.lpSum(cap[k].values()) == 1
        for pid_type in SQUAD_QUOTA:
            sel = [start[k][p] for p in ids if pos[p] == pid_type]
            prob += pulp.lpSum(sel) >= XI_MIN[pid_type]
            prob += pulp.lpSum(sel) <= XI_MAX[pid_type]

        # Expected points objective contributions
        for p in ids:
            pts = proj[p].get(gw, 0.0)
            obj.append(w * pts * (start[k][p] + cap[k][p]))
            obj.append(w * opt.bench_weight * pts * (squad[k][p] - start[k][p]))
            obj.append(EPS * xmins[p] * squad[k][p])

        # Penalty for hits
        obj.append(-1.0 * w * opt.hit_cost * hits[k])

    prob += pulp.lpSum(obj)
    prob.solve(pulp.PULP_CBC_CMD(msg=1 if opt.verbose else 0))

    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"No feasible transfer trajectory: {pulp.LpStatus[prob.status]}")

    # Read GW0 immediate solution
    sol = _read_solution(players, proj, gws, squad[0], start[0], cap[0])
    new_squad_0 = {p["id"] for p in sol["squad"]}
    sol["out"] = [_row(players, proj, gws, p) for p in sorted(cur - new_squad_0)]
    sol["in"] = [_row(players, proj, gws, p) for p in sorted(new_squad_0 - cur)]
    sol["transfers"] = len(sol["in"])
    sol["hit"] = round(float(hits[0].value() or 0) * opt.hit_cost, 1)

    # Record future multi-gameweek planned transfers
    future_plans = []
    prev_squad = new_squad_0
    for k in range(1, H):
        gw = gws[k]
        gw_squad = {p for p in ids if squad[k][p].value() > 0.5}
        gw_out = [_row(players, proj, gws, p) for p in sorted(prev_squad - gw_squad)]
        gw_in = [_row(players, proj, gws, p) for p in sorted(gw_squad - prev_squad)]
        gw_hit = round(float(hits[k].value() or 0) * opt.hit_cost, 1)
        future_plans.append({
            "gw": gw,
            "transfers": len(gw_in),
            "in": gw_in,
            "out": gw_out,
            "hit": gw_hit,
        })
        prev_squad = gw_squad

    sol["future_transfers"] = future_plans
    sol["objective"] = round(float(pulp.value(prob.objective) or 0.0), 4)
    return sol


# --------------------------------------------------------------------------- solution formatting


def _row(players: pd.DataFrame, proj: dict, gws: list[int], pid: int) -> dict:
    r = players[players["id"] == pid].iloc[0]
    return {
        "id": int(pid),
        "name": r["name"],
        "team": r["team"],
        "pos": r["pos"],
        "price": float(r["price"]),
        "proj_next": float(proj[pid].get(gws[0], 0.0)),
        "proj_horizon": float(sum(proj[pid].get(g, 0.0) for g in gws)),
        "xmins": float(r["xmins"]),
        "p_start": float(r.get("p_start", 0.0)),
        "news": r.get("news", ""),
        # Availability and transfer context, so the reports can explain a pick
        # rather than just assert it.
        "news_category": r.get("news_category", "fit"),
        "news_summary": r.get("news_summary", ""),
        "web_headline": r.get("web_headline", ""),
        "is_new_signing": bool(r.get("is_new_signing", False)),
        "days_at_club": (None if pd.isna(r.get("days_at_club")) else int(r.get("days_at_club"))),
        "avail_next": float(r.get("avail_next", 1.0)),
    }


def _read_solution(players: pd.DataFrame, proj: dict, gws: list[int], squad, start, cap) -> dict:
    chosen = [p for p in players["id"] if squad[p].value() > 0.5]
    xi = [p for p in chosen if start[p].value() > 0.5]
    captain = next((p for p in chosen if cap[p].value() > 0.5), None)
    order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    rows = [_row(players, proj, gws, p) for p in chosen]
    rows.sort(key=lambda r: (order[r["pos"]], -r["proj_next"]))
    bench = [r for r in rows if r["id"] not in xi]
    bench.sort(key=lambda r: (r["pos"] == "GKP", -r["proj_next"]))
    xi_rows = [r for r in rows if r["id"] in xi]
    vice = max((r for r in xi_rows if r["id"] != captain), key=lambda r: r["proj_next"], default=None)
    return {
        "squad": rows,
        "xi": xi_rows,
        "bench": bench,
        "captain": captain,
        "vice": vice["id"] if vice else None,
        "cost": round(sum(r["price"] for r in rows), 1),
        "proj_next": round(sum(r["proj_next"] for r in xi_rows) + (proj[captain].get(gws[0], 0.0) if captain else 0.0), 2),
    }

