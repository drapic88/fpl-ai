"""Watch your own squad for availability news, and report only what has changed.

Built for an unattended daily run. Two things separate it from `python -m fplai
news`:

* **It is cheap.** It touches the bootstrap endpoint and the RSS feeds and
  nothing else -- no fixtures, no ~626 player histories, no optimiser. A daily
  run on a quiet day costs one small request.
* **It has a memory.** Every run writes what it believed about each of your
  players to a state file, and the next run reports the *diff*. A daily task
  that reprints the same fifteen rows is a task you stop reading, so by default
  a run with no new news prints one line and exits non-zero.

    python watch_availability.py              # what changed since the last run
    python watch_availability.py --full       # the whole squad, changed or not
    python watch_availability.py --json out.json
    python watch_availability.py --no-web     # FPL's own flags only

Exit code 0 -> there is something to report.
Exit code 1 -> nothing new; a scheduled caller should stay quiet.
Exit code 2 -> could not run (squad file missing, or still the template).

The report deliberately stops at "here is what the news says". Whether a doubt
is worth a transfer is a judgement call, and the point of the state file is that
a human -- or an agent with a web search -- only has to make that call about the
handful of players whose situation actually moved.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import unicodedata
from pathlib import Path

from fplai import api, news as news_mod

SQUAD_FILE = Path(__file__).with_name("my_squad.json")
STATE_FILE = Path(__file__).with_name(".availability_state.json")

#: Availability is read over a short horizon rather than a single gameweek so a
#: parsed return date can be placed in the schedule: "expected back 10 Oct" has
#: to resolve to a gameweek before we can say he misses the next one.
HORIZON = 3

#: Categories that mean "this needs a human or a web search before the deadline".
#: `fit` with a full chance is the only state that needs no second look.
WATCH_CATEGORIES = (
    news_mod.DOUBT, news_mod.INJURED, news_mod.SUSPENDED,
    news_mod.PERSONAL, news_mod.UNSETTLED, news_mod.GONE, news_mod.UNKNOWN,
)

#: The fields whose change is worth waking someone up for. Deliberately excludes
#: age-in-days fields, which tick over every day on their own and would make
#: every player look like news every morning.
FINGERPRINT = ("category", "chance_next", "return_gw", "news", "web_category", "web_headline")

try:  # accented names and the pound sign need utf-8 on a Windows console
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_squad(path: Path) -> dict:
    """Read the squad file, tolerating the ways a hand-edited JSON file goes wrong.

    Copying a block of JSON out of a browser or a chat window brings non-breaking
    spaces with it, and `json` rejects U+00A0 as whitespace with an error that
    points at a line looking perfectly innocent. The characters carry no meaning
    in the indentation of a squad file, so they are normalised rather than
    diagnosed. A trailing comma is left to fail: that one can change what a file
    means, and guessing is worse than saying so.
    """
    raw = path.read_text(encoding="utf-8")
    cleaned = raw.replace("\u00a0", " ").replace("\ufeff", "")
    if cleaned != raw:
        print("  (note: %s contained non-breaking spaces; read as ordinary "
              "whitespace)" % path.name)
    return json.loads(cleaned)


def norm(s: str) -> str:
    """Fold accents and punctuation so 'Guehi' matches 'Guehi' and "O'Reilly" matches."""
    s = unicodedata.normalize("NFKD", str(s))
    stripped = "".join(c for c in s if not unicodedata.combining(c))
    return stripped.lower().replace("'", "").replace(".", "").strip()


def next_gameweek(bootstrap: dict) -> int:
    """The gameweek we are planning for -- the same rule the CLI uses."""
    for e in bootstrap["events"]:
        if e.get("is_next"):
            return int(e["id"])
    for e in bootstrap["events"]:
        if not e.get("finished"):
            return int(e["id"])
    return 38


def deadline_of(bootstrap: dict, gw: int) -> dt.datetime | None:
    for e in bootstrap["events"]:
        if int(e["id"]) == gw and e.get("deadline_time"):
            return dt.datetime.fromisoformat(e["deadline_time"].replace("Z", "+00:00"))
    return None


# --------------------------------------------------------------------------- squad resolution


def resolve_squad(elements: list[dict], teams: dict[int, dict], squad: list[dict]):
    """Map (name, club) pairs from the squad file onto FPL elements.

    Matching is on the club plus the web name, exactly as `plan_week.py` does it,
    so one squad file serves both scripts and a name that works there works here.
    """
    by_club: dict[str, list[dict]] = {}
    for e in elements:
        short = teams[e["team"]]["short_name"]
        by_club.setdefault(short.upper(), []).append(e)

    found, missing = [], []
    for p in squad:
        club = str(p.get("club", "")).upper()
        want = norm(p.get("name", ""))
        pool = by_club.get(club, [])
        hits = [e for e in pool if norm(e["web_name"]) == want]
        if not hits:
            hits = [e for e in pool if want and want in norm(
                f"{e.get('first_name', '')} {e.get('second_name', '')} {e['web_name']}")]
        if len(hits) != 1:
            missing.append(f"{p.get('name')} ({p.get('club')}) -> {len(hits)} match(es)")
            continue
        found.append((p, hits[0]))
    return found, missing


def fielding_roles(data: dict) -> dict[str, tuple[str, str]]:
    """normalised player name -> (role, armband) from the file's `fielding` block.

    A doubt is not equally expensive everywhere. Your captain limping out of a
    press conference is a different problem from your third bench outfielder
    doing the same, and a report that lists them in the same tone makes you do
    the triage yourself. The block is optional -- a squad file without it just
    gets blank roles.
    """
    fielding = data.get("fielding") or {}
    out: dict[str, tuple[str, str]] = {}
    captain = norm(fielding.get("captain") or "")
    vice = norm(fielding.get("vice") or "")

    def armband(key: str) -> str:
        return "C" if key and key == captain else ("V" if key and key == vice else "")

    for name in fielding.get("xi") or []:
        key = norm(name)
        out[key] = ("XI", armband(key))
    for n, name in enumerate(fielding.get("bench_order") or [], 1):
        key = norm(name)
        out[key] = ("B%d" % n, armband(key))
    gk = fielding.get("bench_gk")
    if gk:
        out[norm(gk)] = ("BGK", "")
    return out


#: Sort key putting the players whose availability actually decides points first:
#: the XI ahead of the bench, and within the bench in autosub order.
ROLE_ORDER = {"XI": 0, "B1": 1, "B2": 2, "B3": 3, "BGK": 4, "": 5}


# --------------------------------------------------------------------------- records


def record(entry: dict, element: dict, sig: news_mod.Availability, gw: int,
           teams: dict[int, dict], role: str = "", armband: str = "") -> dict:
    """One player's availability, flattened for printing, diffing and JSON."""
    return {
        "id": int(element["id"]),
        "name": element["web_name"],
        "club": teams[element["team"]]["short_name"],
        "pos": entry.get("pos", ""),
        "role": role,
        "armband": armband,
        "category": sig.category,
        "summary": sig.summary(),
        "chance_next": sig.chance_next,
        "return_gw": sig.return_gw,
        "out_all_season": sig.return_beyond_schedule,
        # The availability multiplier the model would apply to this player's
        # minutes in the coming gameweek: 1.0 plays, 0.0 cannot.
        "avail_next": round(float(sig.curve.get(gw, 1.0)), 3),
        "news": sig.news,
        "news_age_days": sig.news_age_days,
        "web_category": sig.web_category or "",
        "web_headline": sig.web_headline,
        "web_age_days": sig.web_age_days,
        "needs_check": bool(
            sig.category in WATCH_CATEGORIES
            or sig.web_category
            or (sig.chance_next is not None and sig.chance_next < 1.0)
        ),
    }


def fingerprint(rec: dict) -> dict:
    return {k: rec.get(k) for k in FINGERPRINT}


def describe_change(old: dict | None, new: dict) -> str:
    """Why this player is in today's report, in one line."""
    if old is None:
        return "first seen"
    bits = []
    if old.get("category") != new["category"]:
        bits.append(f"{old.get('category')} -> {new['category']}")
    if old.get("chance_next") != new["chance_next"]:
        def pct(v):
            return "n/a" if v is None else f"{float(v):.0%}"
        bits.append(f"chance {pct(old.get('chance_next'))} -> {pct(new['chance_next'])}")
    if old.get("return_gw") != new["return_gw"]:
        bits.append(f"return GW{old.get('return_gw') or '?'} -> GW{new['return_gw'] or '?'}")
    if (old.get("news") or "") != (new["news"] or ""):
        # A cleared news string is the best news there is -- say so, rather than
        # burying it under the same wording as a fresh injury.
        if not new["news"]:
            bits.append("FPL news cleared")
        else:
            bits.append("FPL news changed" if old.get("news") else "FPL news added")
    if (old.get("web_category") or "") != (new["web_category"] or ""):
        bits.append(f"headline: {new['web_category'] or 'cleared'}")
    elif (old.get("web_headline") or "") != (new["web_headline"] or ""):
        bits.append("new headline")
    return ", ".join(bits) or "changed"


# --------------------------------------------------------------------------- reporting


def print_row(rec: dict, note: str = "") -> None:
    mark = " " if not rec["needs_check"] else ("x" if rec["category"] == news_mod.GONE else "!")
    plays = f"{rec['avail_next']:.0%}"
    where = rec.get("role", "") + (" " + rec["armband"] if rec.get("armband") else "")
    print("  %s %-16s%-5s%-4s%-6s%-24s%6s  %s" % (
        mark, rec["name"], rec["club"], rec["pos"], where, rec["summary"][:23], plays, note))
    detail = rec["news"] or ""
    if detail:
        print("      fpl: %s" % detail[:88])
    if rec["web_headline"]:
        age = "" if rec["web_age_days"] is None else " (%.0fd old)" % rec["web_age_days"]
        print("      web[%s]: %s%s" % (rec["web_category"], rec["web_headline"][:74], age))


def header(gw: int) -> None:
    print("\n  %s %-16s%-5s%-4s%-6s%-24s%6s  %s" % (
        " ", "PLAYER", "CLUB", "POS", "WHERE", "AVAILABILITY", "GW" + str(gw), "WHY IT IS HERE"))
    print("  " + "-" * 100)


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Report availability news about your own squad, only when it changes")
    ap.add_argument("--squad", default=str(SQUAD_FILE), help="squad JSON (default my_squad.json)")
    ap.add_argument("--state", default=str(STATE_FILE),
                    help="where to remember the last run (default .availability_state.json)")
    ap.add_argument("--full", action="store_true",
                    help="print the whole squad, not just what changed")
    ap.add_argument("--flagged", action="store_true",
                    help="print every currently flagged player, changed or not")
    ap.add_argument("--json", help="also write the report to this JSON file")
    ap.add_argument("--no-web", action="store_true",
                    help="skip the RSS feeds; FPL's own injury flags are still parsed")
    ap.add_argument("--news-max-age", type=float, default=10.0,
                    help="ignore headlines older than this many days (default 10)")
    ap.add_argument("--no-save", action="store_true",
                    help="do not update the state file (a dry run)")
    cli = ap.parse_args()

    squad_path = Path(cli.squad)
    if not squad_path.exists():
        print("No squad file at %s." % squad_path)
        print("Create one with:  cp my_squad.example.json my_squad.json")
        print("then replace the placeholder names with your own fifteen players.")
        return 2

    # A squad file that will not parse must exit 2, never 1. Exit 1 means "nothing
    # has changed, stay quiet", and an unattended caller that cannot tell a broken
    # file from a quiet week will cheerfully report all-clear on a broken week.
    try:
        data = load_squad(squad_path)
    except json.JSONDecodeError as exc:
        print("%s is not valid JSON: %s" % (squad_path.name, exc.msg))
        print("  at line %d, column %d." % (exc.lineno, exc.colno))
        print("  Common causes: a trailing comma after the last player, a missing")
        print("  comma between two, or smart quotes in place of \".")
        return 2
    except OSError as exc:
        print("Could not read %s: %s" % (squad_path, exc))
        return 2
    if not isinstance(data, dict) or not data.get("squad"):
        print("%s has no \"squad\" list -- start from my_squad.example.json." % squad_path.name)
        return 2
    squad = data["squad"]
    if any(str(p.get("club", "")).upper() == "XXX" for p in squad):
        print("%s is still the template (clubs are 'XXX')." % squad_path.name)
        print("Fill in your fifteen players -- names as they appear on the FPL site,")
        print("clubs as the 3-letter FPL short codes -- then this check will run.")
        return 2

    bs = api.bootstrap()
    teams = {t["id"]: t for t in bs["teams"]}
    gw = next_gameweek(bs)
    gws = list(range(gw, gw + HORIZON))
    deadline = deadline_of(bs, gw)
    now = dt.datetime.now(dt.timezone.utc)
    hours = (deadline - now).total_seconds() / 3600.0 if deadline else None

    found, missing = resolve_squad(bs["elements"], teams, squad)
    if missing:
        print("Could not resolve these players -- fix %s:" % squad_path.name)
        for m in missing:
            print("   ", m)
        if not found:
            return 2

    cfg = news_mod.NewsConfig(use_web=not cli.no_web, web_max_age_days=cli.news_max_age)
    avail = news_mod.build_availability(bs, gws, cfg, today=now.date())

    roles = fielding_roles(data)
    records = []
    for entry, element in found:
        sig = avail.get(int(element["id"])) or news_mod.Availability(player_id=int(element["id"]))
        role, armband = roles.get(norm(entry.get("name", "")), ("", ""))
        records.append(record(entry, element, sig, gw, teams, role, armband))
    # XI before bench, so the rows that decide this week's points come first.
    records.sort(key=lambda r: (ROLE_ORDER.get(r["role"], 5), r["name"]))

    state_path = Path(cli.state)
    previous = {}
    if state_path.exists():
        try:
            previous = {str(k): v for k, v in
                        json.loads(state_path.read_text(encoding="utf-8")).get("players", {}).items()}
        except (ValueError, OSError):
            previous = {}
    first_run = not previous

    changed = []
    for rec in records:
        old = previous.get(str(rec["id"]))
        if old is None or fingerprint(old) != fingerprint(rec):
            # On a first run every player looks new. Reporting fifteen "first seen"
            # rows is noise, so only the ones actually carrying news are worth it.
            if first_run and not rec["needs_check"]:
                continue
            changed.append(dict(rec, change=describe_change(old, rec)))

    flagged = [r for r in records if r["needs_check"]]

    stamp = "GW%d" % gw
    if deadline:
        stamp += " deadline %s" % deadline.astimezone().strftime("%a %d %b %H:%M %Z")
        stamp += " (%.1fh away)" % hours if hours is not None and hours >= 0 else " (passed)"
    print("\nSquad availability watch -- %s" % stamp)
    print("  %d of %d players resolved from %s" % (len(records), len(squad), squad_path.name))

    if changed:
        print("\n  CHANGED SINCE THE LAST RUN (%d):" % len(changed))
        header(gw)
        for rec in changed:
            print_row(rec, rec["change"])
    elif first_run:
        # There is no "since the last run" on a first run, and saying so anyway
        # reads as an all-clear that was never actually checked.
        print("\n  First run -- baseline saved. Nothing flagged: all %d player(s) "
              "available for GW%d." % (len(records), gw))
    else:
        print("\n  Nothing has changed since the last run.")

    if cli.full or cli.flagged:
        rest = [r for r in (records if cli.full else flagged)
                if r["id"] not in {c["id"] for c in changed}]
        if rest:
            print("\n  %s:" % ("REST OF THE SQUAD, UNCHANGED" if cli.full
                               else "STILL FLAGGED, UNCHANGED"))
            header(gw)
            for rec in rest:
                print_row(rec, "unchanged")

    # A quiet run must be genuinely quiet: the standing list of flagged players is
    # only worth reprinting when the caller asked for the full picture.
    if changed:
        to_check = [r for r in changed if r["needs_check"]]
    elif cli.full or cli.flagged:
        to_check = flagged
    else:
        to_check = []
    if to_check:
        print("\n  Worth a second look before the deadline: %s" %
              ", ".join("%s (%s%s)" % (r["name"], r["role"] or "?",
                                       " " + r["armband"] if r["armband"] else "")
                        for r in to_check))
        print("  FPL's flags and the RSS feeds lag press conferences, so for these")
        print("  players the last word is the club's own team news.")

    payload = {
        "generated": now.isoformat(),
        "gw": gw,
        "deadline": deadline.isoformat() if deadline else None,
        "hours_to_deadline": round(hours, 2) if hours is not None else None,
        "squad_file": str(squad_path),
        "unresolved": missing,
        "changed": changed,
        "flagged": flagged,
        "players": records,
    }
    if cli.json:
        Path(cli.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("\n  json -> %s" % cli.json)

    if not cli.no_save:
        state_path.write_text(json.dumps(
            {"generated": now.isoformat(), "gw": gw,
             "players": {str(r["id"]): r for r in records}},
            indent=2), encoding="utf-8")

    return 0 if changed else 1


if __name__ == "__main__":
    sys.exit(main())
