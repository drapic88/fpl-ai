"""Is an FPL deadline close enough to be worth a squad check?

Exit code 0 -> a deadline is inside the window, go and run plan_week.py.
Exit code 1 -> nothing due, stay quiet.

Cheap on purpose: it reads only the cached bootstrap endpoint, so a daily run
on a quiet day costs one small request instead of ~626 player histories.

    python deadline_gate.py             # default window
    python deadline_gate.py --hours 48  # look further ahead
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

from fplai import api

# Saturday deadlines land early enough that a same-morning check is tight, so the
# window is wide enough to also fire the day before. Friday-evening deadlines
# stay a single same-day check.
WINDOW_HOURS = 30.0

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def next_deadline(bootstrap: dict):
    """The next gameweek and its deadline, or (None, None) once the season ends."""
    events = bootstrap["events"]
    ev = next((e for e in events if e.get("is_next")), None)
    if ev is None:
        ev = next((e for e in events if not e.get("finished")), None)
    if ev is None:
        return None, None
    deadline = dt.datetime.fromisoformat(ev["deadline_time"].replace("Z", "+00:00"))
    return ev, deadline


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate a squad check on the next FPL deadline")
    ap.add_argument("--hours", type=float, default=WINDOW_HOURS,
                    help="report if the deadline is within this many hours (default %.0f)" % WINDOW_HOURS)
    cli = ap.parse_args()

    ev, deadline = next_deadline(api.bootstrap())
    if ev is None:
        print("SKIP: no gameweek left in the season.")
        return 1

    now = dt.datetime.now(dt.timezone.utc)
    hours = (deadline - now).total_seconds() / 3600.0
    local = deadline.astimezone()
    stamp = "GW%d deadline %s (%.1fh away)" % (
        ev["id"], local.strftime("%a %d %b %H:%M %Z"), hours)

    if hours < 0:
        print("SKIP: %s has passed; waiting for FPL to roll over to the next gameweek." % stamp)
        return 1
    if hours > cli.hours:
        print("SKIP: %s -- outside the %.0fh window, nothing to report." % (stamp, cli.hours))
        return 1

    print("CHECK: %s -- inside the %.0fh window." % (stamp, cli.hours))
    return 0


if __name__ == "__main__":
    sys.exit(main())
