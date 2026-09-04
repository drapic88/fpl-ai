"""Command line interface.

    python -m fplai squad                  # build a squad from scratch (GW1 or a wildcard)
    python -m fplai weekly --entry 1234567 # transfers + lineup for the coming gameweek
    python -m fplai players --pos MID      # ranked shortlist
    python -m fplai news                   # who is injured, suspended, or in the headlines
    python -m fplai signings               # who has just moved, and will they play
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

import pandas as pd

from . import api, external, model, news as news_mod, optimize, transfers


# --------------------------------------------------------------------------- helpers


def next_gameweek(bootstrap: dict) -> int:
    for e in bootstrap["events"]:
        if e.get("is_next"):
            return e["id"]
    for e in bootstrap["events"]:
        if not e.get("finished"):
            return e["id"]
    return 38


def resolve(df: pd.DataFrame, names: list[str]) -> list[int]:
    out = []
    for n in names:
        if n.isdigit():
            out.append(int(n))
            continue
        hit = df[df["name"].str.lower() == n.lower()]
        if hit.empty:
            hit = df[df["name"].str.lower().str.contains(n.lower(), regex=False)]
        if hit.empty:
            print(f"  ! unknown player: {n}")
        elif len(hit) > 1:
            print(f"  ! ambiguous '{n}': {', '.join(hit['name'] + ' (' + hit['team'] + ')')}")
        else:
            out.append(int(hit.iloc[0]["id"]))
    return out


def build_configs(args) -> tuple[model.ModelConfig, external.ExternalConfig]:
    """Translate command line flags into the model's sub-configs."""
    news_cfg = news_mod.NewsConfig(
        use_web=not args.no_news,
        web_max_age_days=args.news_max_age,
    )
    squad_cfg = transfers.SquadConfig(
        rotation_weight=args.rotation_weight,
        new_signing_days=args.new_signing_days,
    )
    cfg = model.ModelConfig(
        horizon=args.horizon,
        decay=args.decay,
        ep_blend=args.ep_blend,
        news=news_cfg,
        squad=squad_cfg,
    )
    ext_cfg = external.ExternalConfig(
        enabled=not args.no_external,
        season=args.external_season,
    )
    return cfg, ext_cfg


def load_data(args) -> tuple[dict, list[dict], pd.DataFrame, dict, int]:
    print("Fetching FPL data...")
    bs = api.bootstrap()
    fx = api.fixtures()
    gw = args.gw or next_gameweek(bs)
    gws = list(range(gw, gw + args.horizon))
    today = dt.date.today()

    cfg, ext_cfg = build_configs(args)

    print("Reading team news and availability...")
    avail = news_mod.build_availability(bs, gws, cfg.news, today=today, verbose=True)
    moves = transfers.classify_moves(bs["elements"], today, cfg.squad)
    n_new = sum(1 for m in moves.values() if m["is_new_signing"])
    n_out = sum(1 for a in avail.values() if a.category == news_mod.GONE)
    n_flag = sum(1 for a in avail.values() if a.flagged and a.category != news_mod.GONE)
    print(f"  {n_flag} player(s) carrying a fitness or availability flag, "
          f"{n_out} no longer selectable, {n_new} signed within {cfg.squad.new_signing_days} days")

    ext = {}
    if ext_cfg.enabled:
        print("Fetching foreign-league rates for players with no PL record...")
        ext = external.load(bs["elements"], ext_cfg, today=today, verbose=True)

    hist = {}
    if not args.shallow:
        print("Fetching player histories (cached for 24h, slow the first time)...")
        hist = model.fetch_history([e["id"] for e in bs["elements"]])

    df, proj = model.build(bs, fx, gw, cfg, hist, availability=avail, moves=moves,
                           external=ext, today=today)

    if args.overrides:
        cfg.overrides = model.load_overrides(args.overrides, df)
        df, proj = model.build(bs, fx, gw, cfg, hist, availability=avail, moves=moves,
                               external=ext, today=today)

    return bs, fx, df, proj, gw


def risk_flag(r) -> str:
    """A one-character marker for why a player might not deliver his projection."""
    cat = getattr(r, "news_category", "fit") if not isinstance(r, dict) else r.get("news_category", "fit")
    new = getattr(r, "is_new_signing", False) if not isinstance(r, dict) else r.get("is_new_signing", False)
    if cat == news_mod.GONE:
        return "x"
    if cat in (news_mod.INJURED, news_mod.SUSPENDED, news_mod.PERSONAL):
        return "!"
    if cat == news_mod.UNSETTLED:
        return "~"
    if cat == news_mod.DOUBT:
        return "?"
    if new:
        return "+"
    return " "


def fmt_player(r: dict, flag: str = "") -> str:
    note = r.get("news_summary") or ""
    if note == "available":
        note = ""
    if not note and r.get("is_new_signing"):
        note = f"new signing, {r.get('days_at_club')}d"
    note = f"  {risk_flag(r)} {note[:38]}" if note else ""
    return (
        f"  {flag:<2}{r['pos']:<4}{r['name']:<16}{r['team']:<5}"
        f"£{r['price']:<5.1f}{r['proj_next']:>6.2f}{r['proj_horizon']:>8.2f}"
        f"{r.get('xmins', 0):>6.0f}{note}"
    )


def show(sol: dict, gw: int, horizon: int):
    cap, vice = sol["captain"], sol["vice"]
    print(f"\n{'':<6}{'POS':<4}{'PLAYER':<16}{'TEAM':<5}{'PRICE':<6}"
          f"{'GW' + str(gw):>6}{'NEXT' + str(horizon):>8}{'XMIN':>6}")
    print("  " + "-" * 74)
    print("  STARTING XI")
    for r in sol["xi"]:
        flag = "C" if r["id"] == cap else ("V" if r["id"] == vice else "")
        print(fmt_player(r, flag))
    print("  BENCH")
    for r in sol["bench"]:
        print(fmt_player(r))
    print("  " + "-" * 74)
    print(f"  Squad value £{sol['cost']}m   Projected GW{gw} (with captain): {sol['proj_next']} pts")
    risky = [r for r in sol["xi"] if risk_flag(r).strip()]
    if risky:
        print("\n  Watch before the deadline:")
        for r in risky:
            extra = r.get("web_headline") or r.get("news") or ""
            print(f"    {risk_flag(r)} {r['name']:<14} {r.get('news_summary', '')}"
                  f"{('  -- ' + extra[:52]) if extra else ''}")


# --------------------------------------------------------------------------- commands


def cmd_squad(args):
    _, _, df, proj, gw = load_data(args)
    gws = list(range(gw, gw + args.horizon))
    opt = optimize.SolveOptions(
        budget=args.budget,
        bench_weight=args.bench_weight,
        decay=args.decay,
        include=tuple(resolve(df, args.include)),
        exclude=tuple(resolve(df, args.exclude)),
        verbose=args.verbose,
    )
    print(f"\nOptimising a 15-man squad for GW{gw}-{gw + args.horizon - 1} on a £{args.budget}m budget...")
    sol = optimize.pick_squad(df, proj, gws, opt)
    show(sol, gw, args.horizon)
    if args.json:
        open(args.json, "w").write(json.dumps(sol, indent=2))
        print(f"\n  saved -> {args.json}")


def cmd_weekly(args):
    _, _, df, proj, gw = load_data(args)
    gws = list(range(gw, gw + args.horizon))

    bank, free, current, selling = args.bank, args.free_transfers, [], {}
    if args.cookie:
        team = api.my_team(args.entry, open(args.cookie).read().strip() if args.cookie.endswith(".txt") else args.cookie)
        current = [p["element"] for p in team["picks"]]
        selling = {p["element"]: p["selling_price"] / 10.0 for p in team["picks"]}
        bank = team["transfers"]["bank"] / 10.0
        free = team["transfers"]["limit"] - team["transfers"]["made"]
    else:
        picks = api.entry_picks(args.entry, gw - 1)
        current = [p["element"] for p in picks["picks"]]
        print("  (no cookie given: using last gameweek's picks and current prices as selling prices)")

    if not current:
        sys.exit("Could not read your current squad. Pass --cookie for live data.")

    # Anyone in the squad who has left the league or is out long term is worth calling
    # out explicitly -- the optimiser will move them, but you want to know why.
    owned = df[df["id"].isin(current)]
    gone = owned[owned["news_category"] == news_mod.GONE]
    hurt = owned[owned["news_category"].isin([news_mod.INJURED, news_mod.SUSPENDED, news_mod.PERSONAL])]
    if len(gone) or len(hurt):
        print("\n  Your squad's availability:")
        for r in pd.concat([gone, hurt]).itertuples():
            print(f"    {risk_flag(r)} {r.name:<16}{r.team:<5} {r.news_summary}"
                  f"{('  -- ' + r.news[:48]) if r.news else ''}")

    opt = optimize.SolveOptions(
        bench_weight=args.bench_weight,
        decay=args.decay,
        free_transfers=max(0, free),
        max_transfers=15 if args.wildcard else args.max_transfers,
        bank=bank,
        wildcard=args.wildcard,
        include=tuple(resolve(df, args.include)),
        exclude=tuple(resolve(df, args.exclude)),
        verbose=args.verbose,
    )
    print(f"\nPlanning GW{gw}: bank £{bank}m, {free} free transfer(s){', WILDCARD' if args.wildcard else ''}")
    sol = optimize.plan_transfers(df, proj, gws, current, selling, opt)

    if sol["transfers"] == 0:
        print("\n  No transfer is worth it this week — roll it.")
    else:
        print(f"\n  {sol['transfers']} transfer(s), points hit: -{int(sol['hit'])}")
        for o, i in zip(sol["out"], sol["in"]):
            reason = o.get("news_summary", "")
            reason = f"  [{reason}]" if reason and reason != "available" else ""
            print(f"    OUT {o['name']:<14} £{o['price']:<5.1f} ->  IN {i['name']:<14} £{i['price']:<5.1f}"
                  f"  (+{i['proj_horizon'] - o['proj_horizon']:.2f} over {args.horizon} GWs){reason}")

    if sol.get("future_transfers"):
        print("\n  Future Trajectory Preview:")
        for fut in sol["future_transfers"]:
            if fut["transfers"] == 0:
                print(f"    GW{fut['gw']}: Roll transfer")
            else:
                hit_str = f" (hit -{int(fut['hit'])})" if fut["hit"] > 0 else ""
                moves = ", ".join(f"OUT {o['name']} -> IN {i['name']}" for o, i in zip(fut["out"], fut["in"]))
                print(f"    GW{fut['gw']}: {moves}{hit_str}")

    show(sol, gw, args.horizon)
    if args.json:
        open(args.json, "w").write(json.dumps(sol, indent=2))


def cmd_players(args):
    _, _, df, proj, gw = load_data(args)
    d = df.copy()
    if args.pos:
        d = d[d["pos"] == args.pos.upper()]
    if args.max_price:
        d = d[d["price"] <= args.max_price]
    d = d[d["xmins"] > args.min_minutes]
    d = d.sort_values(args.sort, ascending=False).head(args.top)
    print(f"\n{'':<2}{'POS':<4}{'PLAYER':<16}{'TEAM':<5}{'PRICE':<7}{'GW' + str(gw):>6}"
          f"{'NEXT' + str(args.horizon):>8}{'VALUE':>7}{'OWN%':>7}{'XMIN':>6}{'START':>7}")
    print("-" * 78)
    for r in d.itertuples():
        print(f"{risk_flag(r):<2}{r.pos:<4}{r.name:<16}{r.team:<5}£{r.price:<6.1f}{r.proj_next:>6.2f}"
              f"{r.proj_horizon:>8.2f}{r.value:>7.2f}{r.selected_by:>7.1f}{r.xmins:>6.0f}{r.p_start:>7.0%}")


def cmd_news(args):
    """Everyone the news says something about, worst first."""
    _, _, df, _, gw = load_data(args)
    # A web-only signal (an unsettled player FPL has not flagged) still belongs here.
    d = df[(df["news_category"] != news_mod.FIT) | (df["web_category"] != "")].copy()
    if args.entry:
        picks = api.entry_picks(args.entry, gw - 1)
        owned = {p["element"] for p in picks["picks"]}
        d = d[d["id"].isin(owned)]
    if args.min_owned:
        d = d[d["selected_by"] >= args.min_owned]
    if d.empty:
        print("\n  Nothing flagged.")
        return

    # Ownership first, severity second. A flag on a widely-owned player is the one
    # that costs people points; a departure nobody owned is trivia.
    severity = {news_mod.GONE: 0, news_mod.INJURED: 1, news_mod.SUSPENDED: 2,
                news_mod.PERSONAL: 3, news_mod.UNSETTLED: 4, news_mod.DOUBT: 5}
    d["_sev"] = d["news_category"].map(severity).fillna(9)
    d = d.sort_values(["selected_by", "_sev"], ascending=[False, True]).head(args.top)

    print(f"\n{'':<2}{'POS':<4}{'PLAYER':<16}{'TEAM':<5}{'OWN%':>6}  {'AVAILABILITY':<32}{'NEWS'}")
    print("-" * 112)
    for r in d.itertuples():
        detail = r.news or r.web_headline or ""
        print(f"{risk_flag(r):<2}{r.pos:<4}{r.name:<16}{r.team:<5}{r.selected_by:>6.1f}  "
              f"{r.news_summary[:31]:<32}{detail[:44]}")
        if r.web_headline:
            print(f"{'':<41}web: {r.web_headline[:64]}")

    print(f"\n  {len(d)} shown. Availability is applied per gameweek, so a player with a")
    print("  return date still scores in the gameweeks after it.")


def cmd_signings(args):
    """Players who have just changed club, and what the model thinks they will play."""
    _, _, df, _, gw = load_data(args)
    d = df[df["is_new_signing"]].copy()
    if args.window:
        d = d[d["window"] == args.window]
    if d.empty:
        print("\n  No recent signings found.")
        return
    d = d.sort_values("proj_horizon", ascending=False).head(args.top)

    print(f"\n{'':<2}{'POS':<4}{'PLAYER':<16}{'TEAM':<5}{'PRICE':<7}{'JOINED':>8}"
          f"{'WINDOW':>9}{'START':>7}{'XMIN':>6}{'SETTLE':>8}{'NEXT' + str(args.horizon):>8}  SOURCE")
    print("-" * 104)
    for r in d.itertuples():
        src = f"{r.external_league} {int(r.external_minutes)}min" if r.external_league else "price prior"
        days = "?" if pd.isna(r.days_at_club) else f"{int(r.days_at_club)}d"
        print(f"{risk_flag(r):<2}{r.pos:<4}{r.name:<16}{r.team:<5}£{r.price:<6.1f}"
              f"{days:>8}{r.window:>9}{r.p_start:>7.0%}{r.xmins:>6.0f}"
              f"{r.settle:>8.2f}{r.proj_horizon:>8.2f}  {src}")

    print("\n  START is the chance of starting after competition for the shirt is settled;")
    print("  SETTLE is the minutes discount for having only just arrived.")


def cmd_clear(args):
    print(f"Removed {api.clear_cache()} cached files.")


# --------------------------------------------------------------------------- entry point


def main(argv=None):
    # Player names carry accents and the odd dotless i, and the pound sign is all
    # over the output. A default Windows console encoding kills the run on the
    # first one of those it meets, so ask for UTF-8 and degrade rather than crash.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    p = argparse.ArgumentParser(prog="fplai", description="AI assistant for Fantasy Premier League")
    p.add_argument("--gw", type=int, help="target gameweek (default: the next one)")
    p.add_argument("--horizon", type=int, default=5, help="gameweeks to plan ahead (default 5)")
    p.add_argument("--decay", type=float, default=0.86, help="weight decay per future gameweek")
    p.add_argument("--ep-blend", type=float, default=0.35, help="weight on FPL's own expected points")
    p.add_argument("--bench-weight", type=float, default=0.12)
    p.add_argument("--include", nargs="*", default=[], help="players the squad must contain")
    p.add_argument("--exclude", nargs="*", default=[], help="players to ban")
    p.add_argument("--overrides", help="CSV of your own projections (player,points[,gw])")
    p.add_argument("--shallow", action="store_true", help="skip per-player history fetch (much faster, less accurate)")
    p.add_argument("--json", help="write the result to a JSON file")
    p.add_argument("--verbose", action="store_true")

    news_group = p.add_argument_group("team news")
    news_group.add_argument("--no-news", action="store_true",
                            help="skip the public news feeds; FPL's own injury flags are still used")
    news_group.add_argument("--news-max-age", type=float, default=10.0,
                            help="ignore headlines older than this many days (default 10)")

    xfer_group = p.add_argument_group("transfers and rotation")
    xfer_group.add_argument("--rotation-weight", type=float, default=0.75,
                            help="0 trusts each player's own history; 1 enforces a realistic "
                                 "number of starters per club and position (default 0.75)")
    xfer_group.add_argument("--new-signing-days", type=int, default=90,
                            help="how recently a player must have joined to be treated as a new signing")
    xfer_group.add_argument("--no-external", action="store_true",
                            help="skip foreign-league stats for players with no Premier League record")
    xfer_group.add_argument("--external-season", type=int,
                            help="season start year to read abroad (default: last completed season)")

    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("squad", help="build a squad from scratch")
    s.add_argument("--budget", type=float, default=100.0)
    s.set_defaults(func=cmd_squad)

    w = sub.add_parser("weekly", help="transfers and lineup for the coming gameweek")
    w.add_argument("--entry", type=int, required=True, help="your FPL team id")
    w.add_argument("--cookie", help="cookie string or path to a .txt file containing it")
    w.add_argument("--bank", type=float, default=0.0)
    w.add_argument("--free-transfers", type=int, default=1)
    w.add_argument("--max-transfers", type=int, default=2)
    w.add_argument("--wildcard", action="store_true")
    w.set_defaults(func=cmd_weekly)

    l = sub.add_parser("players", help="ranked player shortlist")
    l.add_argument("--pos", help="GKP / DEF / MID / FWD")
    l.add_argument("--max-price", type=float)
    l.add_argument("--min-minutes", type=float, default=30)
    l.add_argument("--top", type=int, default=25)
    l.add_argument("--sort", default="proj_horizon", choices=["proj_horizon", "proj_next", "value", "selected_by"])
    l.set_defaults(func=cmd_players)

    n = sub.add_parser("news", help="injuries, suspensions, absences and departures")
    n.add_argument("--entry", type=int, help="only show players in this team's squad")
    n.add_argument("--min-owned", type=float, default=0.0, help="only players owned by at least this %%")
    n.add_argument("--top", type=int, default=40)
    n.set_defaults(func=cmd_news)

    g = sub.add_parser("signings", help="recent arrivals and whether they will play")
    g.add_argument("--window", choices=[transfers.SUMMER, transfers.JANUARY],
                   help="restrict to one transfer window")
    g.add_argument("--top", type=int, default=30)
    g.set_defaults(func=cmd_signings)

    sub.add_parser("clear-cache").set_defaults(func=cmd_clear)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
