"""Gameweek snapshots: keep what FPL overwrites, so it can be tested later.

FPL's API only ever shows the present. A player's `ep_next`, injury news,
chance of playing and penalty order are rewritten as the season moves on, so
once a deadline passes there is no way to ask what was known *before* it. The
backtest found that out the hard way: the historical CSVs' `xP` is recorded
after the gameweek, which is why `ep_blend` and the news model cannot be tested
on past seasons at all.

So two kinds of snapshot are kept, one directory per gameweek:

    snapshots/gw12/pre-20261127T1500Z/   before the deadline -- never after it
        meta.json            when, how far from the deadline, model settings, git revision
        bootstrap.json.gz    FPL's own view: ep_next, news, chance of playing,
                             penalty and set-piece order, prices, ownership
        fixtures.json.gz
        projections.csv.gz   this model's projections, production and previous settings
                             (the `cand_` columns)
        squad.json           my_squad.json as it stood: the squad, XI and armbands
        plan.json            what the planner recommended
    snapshots/gw12/post/                 after the gameweek has finished
        live.json.gz         every player's points and stats
        elite_eo.json        effective ownership among the tracked elite managers

A pre snapshot is refused once the deadline has passed: a snapshot that might
contain the gameweek's own information is worse than none. Nothing is ever
overwritten; a second pre snapshot of the same gameweek gets its own directory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

from . import api, elite, model, optimize

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = Path(os.environ.get("FPL_SNAPSHOT_DIR", ROOT / "snapshots"))
ELITE_FILE = "elite_entries.json"
PRE_WINDOW_HOURS = 30.0      # take a pre snapshot only this close to the deadline
MIN_GAP_HOURS = 2.0          # and not again within this long of the last one

# The settings production runs with (plan_week.py), and the ones it replaced on
# the backtest's evidence, so the switch can be checked on real gameweeks.
PRODUCTION = dict(ep_blend=0.35)
CANDIDATE = dict(ep_blend=0.35, shrink_inseason=4.0, shrink_matches=6.0,
                 start_prior_matches=1.5, xg_weight=0.0, defcon=False)
PROJECTION_COLS = [
    "id", "name", "team", "pos", "price", "status", "news", "news_category", "avail",
    "selected_by", "ep_next", "p_start", "xmins", "proj_next", "proj_horizon", "p_haul",
    "xg_next", "xa_next",
]


# --------------------------------------------------------------------------- helpers


def _write_json(path: Path, data, compress: bool = False) -> None:
    raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
    if compress:
        path = path.with_suffix(path.suffix + ".gz")
        path.write_bytes(gzip.compress(raw))
    else:
        path.write_bytes(raw)


def read_json(path: Path):
    """Read a snapshot file, compressed or not."""
    if path.suffix == ".gz":
        return json.loads(gzip.decompress(path.read_bytes()))
    return json.loads(path.read_text(encoding="utf-8"))


def _git_rev() -> str:
    """Short commit of the code that made the projections, `-dirty` if fplai/ has local edits."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    try:
        rev = git("rev-parse", "--short", "HEAD") or "unknown"
        return rev + ("-dirty" if git("status", "--porcelain", "--", "fplai") else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def next_event(bootstrap: dict) -> tuple[dict | None, dt.datetime | None]:
    ev = next((e for e in bootstrap["events"] if e.get("is_next")), None)
    if ev is None:
        ev = next((e for e in bootstrap["events"] if not e.get("finished")), None)
    if ev is None:
        return None, None
    return ev, dt.datetime.fromisoformat(ev["deadline_time"].replace("Z", "+00:00"))


def gw_dir(gw: int, root: Path | None = None) -> Path:
    return (root or SNAPSHOT_DIR) / f"gw{gw:02d}"


def pre_snapshots(gw: int, root: Path | None = None) -> list[Path]:
    d = gw_dir(gw, root)
    return sorted(p for p in d.glob("pre-*") if p.is_dir()) if d.exists() else []


# --------------------------------------------------------------------------- pre-deadline


def pre_gate(bootstrap: dict, now: dt.datetime, root: Path | None = None,
             window: float = PRE_WINDOW_HOURS, min_gap: float = MIN_GAP_HOURS) -> tuple[int | None, str]:
    """(gameweek to snapshot, reason) -- gameweek is None when no snapshot is due."""
    ev, deadline = next_event(bootstrap)
    if ev is None:
        return None, "no gameweek left in the season"
    hours = (deadline - now).total_seconds() / 3600.0
    if hours <= 0:
        return None, f"GW{ev['id']} deadline has passed; a snapshot now could leak the gameweek"
    if hours > window:
        return None, f"GW{ev['id']} deadline is {hours:.1f}h away, outside the {window:.0f}h window"
    last = pre_snapshots(ev["id"], root)
    if last:
        taken = dt.datetime.strptime(last[-1].name[4:], "%Y%m%dT%H%MZ").replace(tzinfo=dt.timezone.utc)
        if (now - taken).total_seconds() / 3600.0 < min_gap:
            return None, f"GW{ev['id']} already snapshotted at {last[-1].name[4:]}"
    return int(ev["id"]), f"GW{ev['id']} deadline in {hours:.1f}h"


def _projections(bootstrap: dict, fixtures: list[dict], gw: int, history: dict,
                 settings: dict, horizon: int = 5) -> tuple[pd.DataFrame, dict]:
    """The same inputs plan_week.py feeds the model: news, moves and foreign-league rates."""
    from . import external, news as news_mod, transfers

    cfg = model.ModelConfig(horizon=horizon, **settings)
    today = dt.date.today()
    gws = list(range(gw, gw + horizon))
    avail = news_mod.build_availability(bootstrap, gws, cfg.news, today=today, verbose=False)
    moves = transfers.classify_moves(bootstrap["elements"], today, cfg.squad)
    ext = external.load(bootstrap["elements"], external.ExternalConfig(), today=today, verbose=False)
    return model.build(bootstrap, fixtures, gw, cfg, history, availability=avail, moves=moves,
                       external=ext, today=today)


def _plan(df: pd.DataFrame, proj: dict, gw: int, squad_file: Path, horizon: int = 5) -> dict:
    """What the planner recommends for the saved squad, as plan_week.py would print it."""
    if not squad_file.exists():
        return {"error": f"{squad_file.name} not found"}
    import plan_week  # the repo root is on sys.path whenever this runs from it

    data = json.loads(squad_file.read_text(encoding="utf-8"))
    df = df.copy()
    df["_n"] = df["name"].map(plan_week.norm)
    current, prices, missing = plan_week.resolve_squad(df, data["squad"])
    if missing:
        return {"error": "unresolved: " + "; ".join(missing)}
    state = data["state"]
    opt = optimize.SolveOptions(bench_weight=0.12, decay=0.86, bank=state["bank"],
                                free_transfers=max(0, int(state.get("free_transfers", 1))),
                                max_transfers=1)
    sol = optimize.plan_transfers(df, proj, list(range(gw, gw + horizon)), current, prices, opt)
    return {
        "transfers": [{"out": o["name"], "in": i["name"]} for o, i in zip(sol["out"], sol["in"])],
        "hit": sol["hit"], "proj_next": sol["proj_next"], "objective": sol["objective"],
        "xi": [r["name"] for r in sol["xi"]], "bench": [r["name"] for r in sol["bench"]],
        "captain": int(sol["captain"]), "vice": int(sol["vice"]),
    }


def take_pre(root: Path | None = None, squad_file: Path | None = None,
             now: dt.datetime | None = None, verbose: bool = True) -> Path | None:
    """Snapshot everything known before the next deadline. Returns the directory, or None."""
    root = root or SNAPSHOT_DIR
    now = now or dt.datetime.now(dt.timezone.utc)
    bootstrap = api.bootstrap(ttl=0)          # fresh: the point is what FPL says right now
    gw, why = pre_gate(bootstrap, now, root)
    if gw is None:
        if verbose:
            print(f"pre: skipped -- {why}")
        return None
    _, deadline = next_event(bootstrap)
    if deadline is None or now >= deadline:
        raise RuntimeError("refusing a pre-deadline snapshot after the deadline")

    fixtures = api.fixtures(ttl=0)
    history = model.fetch_history([e["id"] for e in bootstrap["elements"]], verbose=False)
    frames, plan = {}, {}
    for label, settings in (("production", PRODUCTION), ("candidate", CANDIDATE)):
        df, proj = _projections(bootstrap, fixtures, gw, history, settings)
        frames[label] = df
        if label == "production":
            plan = _plan(df, proj, gw, squad_file or ROOT / "my_squad.json")

    out = gw_dir(gw, root) / f"pre-{now.strftime('%Y%m%dT%H%MZ')}"
    out.mkdir(parents=True, exist_ok=False)
    _write_json(out / "bootstrap.json", bootstrap, compress=True)
    _write_json(out / "fixtures.json", fixtures, compress=True)
    prod = frames["production"][[c for c in PROJECTION_COLS if c in frames["production"]]]
    cand = frames["candidate"].set_index("id")[["proj_next", "proj_horizon", "p_haul", "p_start"]]
    prod = prod.join(cand.add_prefix("cand_"), on="id")
    prod.to_csv(out / "projections.csv.gz", index=False, compression="gzip")
    sf = squad_file or ROOT / "my_squad.json"
    if sf.exists():
        (out / "squad.json").write_text(sf.read_text(encoding="utf-8"), encoding="utf-8")
    _write_json(out / "plan.json", plan)
    _write_json(out / "meta.json", {
        "gw": gw, "taken_at": now.isoformat(), "deadline": deadline.isoformat(),
        "hours_to_deadline": round((deadline - now).total_seconds() / 3600.0, 2),
        "production": PRODUCTION, "candidate": CANDIDATE, "git": _git_rev(),
    })
    if verbose:
        print(f"pre: GW{gw} saved to {out}")
    return out


# --------------------------------------------------------------------------- after the gameweek


def finished_gws(bootstrap: dict) -> list[int]:
    """Gameweeks FPL has finished *and* checked, so the points will not move again."""
    return [int(e["id"]) for e in bootstrap["events"] if e.get("finished") and e.get("data_checked")]


def elite_entries(root: Path | None = None) -> list[int]:
    path = (root or SNAPSHOT_DIR) / ELITE_FILE
    if not path.exists():
        return []
    return [int(r["entry"]) for r in read_json(path)]


def take_post(root: Path | None = None, verbose: bool = True) -> list[int]:
    """Save points and elite ownership for every finished gameweek not yet saved."""
    root = root or SNAPSHOT_DIR
    bootstrap = api.bootstrap()
    entries = elite_entries(root)
    done = []
    for gw in finished_gws(bootstrap):
        out = gw_dir(gw, root) / "post"
        if (out / "live.json.gz").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "live.json", api.event_live(gw, ttl=0), compress=True)
        if entries:
            squads = []
            for e in entries:
                try:
                    squads.append(elite._polite(api.entry_picks, e, gw, ttl=elite.FINISHED_TTL)["picks"])
                except Exception:
                    continue          # a manager who joined later, or a transient failure
            eo = elite.effective_ownership(squads)
            _write_json(out / "elite_eo.json", {"managers": len(squads),
                                                "eo": {str(k): v for k, v in eo.items()}})
        done.append(gw)
    if verbose:
        print("post: " + (f"saved GW{', GW'.join(map(str, done))}" if done else "nothing new"))
    return done


# --------------------------------------------------------------------------- entry point


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Save what FPL will overwrite, gameweek by gameweek")
    ap.add_argument("--pre-only", action="store_true")
    ap.add_argument("--post-only", action="store_true")
    ap.add_argument("--list", action="store_true", help="show what has been saved")
    cli = ap.parse_args(argv)

    if cli.list:
        for d in sorted(SNAPSHOT_DIR.glob("gw*")):
            pre = [p.name[4:] for p in pre_snapshots(int(d.name[2:]))]
            post = (d / "post" / "live.json.gz").exists()
            print(f"{d.name}: pre {', '.join(pre) or '-'}   post {'yes' if post else '-'}")
        return 0
    if not cli.post_only:
        take_pre()
    if not cli.pre_only:
        take_post()
    return 0
