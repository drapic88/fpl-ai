"""Plan the coming gameweek from a saved squad file, with no FPL entry id.

    python plan_week.py                     # plan with the saved bank / free transfers
    python plan_week.py --compare           # score every option from roll to 3 moves
    python plan_week.py --max-transfers 2   # allow a -4 hit to be considered
    python plan_week.py --free-transfers 2  # override what the file says
    python plan_week.py --wildcard          # what a clean slate would look like

Reads my_squad.json, so there is no need to share a team id or screenshots.
Nothing is ever submitted to FPL -- this only ever prints a plan.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

from fplai import optimize
from fplai.cli import load_data

SQUAD_FILE = Path(__file__).with_name("my_squad.json")

try:  # the pound sign and accented names need utf-8 on a Windows console
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def norm(s: str) -> str:
    """Fold accents and punctuation so 'Guehi' matches 'Guehi' and 'O'Reilly' matches."""
    s = unicodedata.normalize("NFKD", str(s))
    stripped = "".join(c for c in s if not unicodedata.combining(c))
    return stripped.lower().replace("'", "").replace(".", "").strip()


def resolve_squad(df, squad):
    """Map (name, club) pairs from the squad file onto FPL player ids."""
    ids, prices, missing = [], {}, []
    for p in squad:
        n = norm(p["name"])
        hit = df[(df["_n"] == n) & (df["team"] == p["club"])]
        if hit.empty:
            hit = df[(df["_n"].str.contains(n, regex=False)) & (df["team"] == p["club"])]
        if len(hit) != 1:
            missing.append("%s (%s) -> %d matches" % (p["name"], p["club"], len(hit)))
            continue
        r = hit.iloc[0]
        ids.append(int(r["id"]))
        prices[int(r["id"])] = float(r["price"])
    return ids, prices, missing


def render(sol, gw, horizon, bank_before, label=None):
    if label:
        print("\n  ===== %s =====" % label)
    if sol["transfers"] == 0:
        print("\n  No transfer is worth it -- roll it.")
    else:
        print("\n  %d transfer(s), hit -%d" % (sol["transfers"], int(sol["hit"])))
        for o, i in zip(sol["out"], sol["in"]):
            note = o.get("news_summary", "")
            note = "  [%s]" % note if note and note != "available" else ""
            print("    OUT %-14s £%-5.1f ->  IN %-14s £%-5.1f  (+%.2f over %d GWs)%s" % (
                o["name"], o["price"], i["name"], i["price"],
                i["proj_horizon"] - o["proj_horizon"], horizon, note))

    bank_after = bank_before + sum(o["price"] for o in sol["out"]) - sum(i["price"] for i in sol["in"])
    print("\n  squad £%.1fm   bank after £%.1fm   GW%d gross %.2f   NET %.2f pts" % (
        sol["cost"], bank_after, gw, sol["proj_next"], sol["proj_next"] - sol["hit"]))

    cap, vice = sol["captain"], sol["vice"]
    print("\n  %-5s%-4s%-16s%-5s%-7s%6s%8s%6s%7s" % (
        "", "POS", "PLAYER", "TEAM", "PRICE", "GW" + str(gw), "NEXT" + str(horizon), "XMIN", "START"))
    print("  " + "-" * 66)
    for r in sol["xi"]:
        mark = "(C)" if r["id"] == cap else ("(V)" if r["id"] == vice else "")
        flag = "!" if r.get("news_category") not in ("fit", "", None) else ""
        print("  %-5s%-4s%-16s%-5s£%-6.1f%6.2f%8.2f%6.0f%6.0f%% %s" % (
            mark, r["pos"], r["name"], r["team"], r["price"], r["proj_next"],
            r["proj_horizon"], r["xmins"], r.get("p_start", 0) * 100, flag))
    print("  BENCH (autosub order)")
    for n, r in enumerate(sol["bench"], 1):
        tag = "GK" if r["pos"] == "GKP" else str(n)
        print("  %-5s%-4s%-16s%-5s£%-6.1f%6.2f%8.2f%6.0f%6.0f%%" % (
            tag, r["pos"], r["name"], r["team"], r["price"], r["proj_next"],
            r["proj_horizon"], r["xmins"], r.get("p_start", 0) * 100))

    risky = [r for r in sol["xi"] if r.get("news_category") not in ("fit", "", None)]
    if risky:
        print("\n  Watch before the deadline:")
        for r in risky:
            extra = r.get("web_headline") or r.get("news") or ""
            print("    ! %-14s %s%s" % (r["name"], r.get("news_summary", ""),
                                        ("  -- " + extra[:50]) if extra else ""))

    if sol.get("future_transfers"):
        print("\n  Planned next weeks:")
        for f in sol["future_transfers"]:
            if f["transfers"] == 0:
                print("    GW%d: roll" % f["gw"])
            else:
                hit = " (hit -%d)" % int(f["hit"]) if f["hit"] > 0 else " (free)"
                moves = ", ".join("%s -> %s" % (o["name"], i["name"])
                                  for o, i in zip(f["out"], f["in"]))
                print("    GW%d: %s%s" % (f["gw"], moves, hit))


def main():
    ap = argparse.ArgumentParser(description="Plan the coming gameweek from my_squad.json")
    ap.add_argument("--bank", type=float, help="override the saved bank")
    ap.add_argument("--free-transfers", type=int, help="override the saved free transfers")
    ap.add_argument("--max-transfers", type=int, default=1, help="cap on moves; raise to allow hits")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--wildcard", action="store_true")
    ap.add_argument("--compare", action="store_true", help="score roll / 1 / 2 / 3 moves side by side")
    ap.add_argument("--shallow", action="store_true", help="skip the per-player history fetch")
    cli = ap.parse_args()

    data = json.loads(SQUAD_FILE.read_text(encoding="utf-8"))
    state = data["state"]
    bank = cli.bank if cli.bank is not None else state["bank"]
    ft = (cli.free_transfers if cli.free_transfers is not None
          else state.get("free_transfers", 1))

    args = argparse.Namespace(
        gw=None, horizon=cli.horizon, decay=0.86, ep_blend=0.35, bench_weight=0.12,
        include=[], exclude=[], overrides=None, shallow=cli.shallow, json=None, verbose=False,
        no_news=False, news_max_age=10.0, rotation_weight=0.75, new_signing_days=90,
        no_external=False, external_season=None,
    )
    bs, fx, df, proj, gw = load_data(args)
    df["_n"] = df["name"].map(norm)
    gws = list(range(gw, gw + cli.horizon))

    current, prices, missing = resolve_squad(df, data["squad"])
    if missing:
        print("\n  Could not resolve these players -- fix my_squad.json:")
        for m in missing:
            print("   ", m)
        sys.exit(1)

    owned = df[df["id"].isin(current)]
    print("\n  Squad from %s (saved %s): %d players, £%.1fm + £%.1fm bank, %d free transfer(s)" % (
        SQUAD_FILE.name, data.get("updated", "?"), len(current), owned["price"].sum(), bank, ft))

    flagged = owned[~owned["news_category"].isin(["fit", ""])]
    if len(flagged):
        print("\n  Availability in your squad:")
        for r in flagged.itertuples():
            print("    ! %-16s%-5s%s%s" % (r.name, r.team, r.news_summary,
                                           ("  -- " + r.news[:46]) if r.news else ""))
    else:
        print("  All 15 available -- no injuries, doubts or departures.")

    def solve(max_transfers, wildcard=False):
        opt = optimize.SolveOptions(
            bench_weight=0.12, decay=0.86, free_transfers=max(0, ft),
            max_transfers=15 if wildcard else max_transfers, bank=bank, wildcard=wildcard)
        return optimize.plan_transfers(df, proj, gws, current, prices, opt)

    if cli.compare:
        print("\n  GW%d options at %d FT, £%.1fm bank" % (gw, ft, bank))
        print("  %-16s %4s %5s %10s %9s %10s" % ("OPTION", "MV", "HIT", "GW gross", "GW NET", "objective"))
        print("  " + "-" * 60)
        for mx, lbl in ((0, "roll"), (1, "1 move"), (2, "2 moves"), (3, "3 moves")):
            s = solve(mx)
            print("  %-16s %4d %5.0f %10.2f %9.2f %10.2f" % (
                lbl, s["transfers"], s["hit"], s["proj_next"],
                s["proj_next"] - s["hit"], s["objective"]))
            for o, i in zip(s["out"], s["in"]):
                print("       OUT %-13s -> IN %-13s (+%.2f/%dgw)" % (
                    o["name"], i["name"], i["proj_horizon"] - o["proj_horizon"], cli.horizon))
        print()

    sol = solve(cli.max_transfers, cli.wildcard)
    render(sol, gw, cli.horizon, bank,
           label="WILDCARD" if cli.wildcard else "PLAN: up to %d move(s)" % cli.max_transfers)
    print("\n  Nothing submitted -- this is a plan, the moves are yours to make.")


if __name__ == "__main__":
    main()
