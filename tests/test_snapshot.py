"""Offline checks for gameweek snapshots. Run with:  python tests/test_snapshot.py

Every network call is replaced by a fake, so this runs without touching FPL.
"""

import datetime as dt
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from fplai import api, model, snapshot  # noqa: E402
from test_synthetic import fake_bootstrap  # noqa: E402

UTC = dt.timezone.utc
DEADLINE = dt.datetime(2026, 11, 27, 18, 30, tzinfo=UTC)


def bootstrap_with_deadline() -> dict:
    bs = fake_bootstrap()
    for e in bs["events"]:
        e["is_next"] = e["id"] == 12
        e["finished"] = e["id"] < 12
        e["data_checked"] = e["id"] < 11          # GW11 finished but not yet checked
        e["deadline_time"] = (DEADLINE + dt.timedelta(days=7 * (e["id"] - 12))).strftime("%Y-%m-%dT%H:%M:%SZ")
    return bs


BS = bootstrap_with_deadline()
CALLS = {"live": 0, "picks": 0}


def fake_projections(bootstrap, fixtures, gw, history, settings, horizon=5):
    cfg = model.ModelConfig(horizon=horizon, **settings)
    return model.build(bootstrap, [], gw, cfg, {}, today=dt.date(2026, 11, 26))


def fake_live(gw, ttl=0):
    CALLS["live"] += 1
    return {"elements": [{"id": 1, "stats": {"total_points": gw}}]}


def fake_picks(entry, gw, ttl=0):
    CALLS["picks"] += 1
    return {"picks": [{"element": 1, "multiplier": 2}, {"element": 2, "multiplier": 1}]}


def install_fakes() -> None:
    api.bootstrap = lambda ttl=1800: BS
    api.fixtures = lambda ttl=3600: []
    api.event_live = fake_live
    api.entry_picks = fake_picks
    model.fetch_history = lambda ids, workers=6, verbose=True: {}
    snapshot._projections = fake_projections
    snapshot.elite.time.sleep = lambda s: None


def check_gate(root: Path) -> None:
    gw, why = snapshot.pre_gate(BS, DEADLINE + dt.timedelta(minutes=1), root)
    assert gw is None and "passed" in why, why
    gw, why = snapshot.pre_gate(BS, DEADLINE - dt.timedelta(hours=40), root)
    assert gw is None and "outside" in why, why
    gw, why = snapshot.pre_gate(BS, DEADLINE - dt.timedelta(hours=3), root)
    assert gw == 12, why
    print("gate ok: only inside the window, never after the deadline")


def check_pre(root: Path, squad: Path) -> None:
    t1 = DEADLINE - dt.timedelta(hours=20)
    out = snapshot.take_pre(root, squad, now=t1, verbose=False)
    assert out is not None and out.name == "pre-20261126T2230Z", out
    for f in ("meta.json", "bootstrap.json.gz", "fixtures.json.gz", "projections.csv.gz",
              "squad.json", "plan.json"):
        assert (out / f).exists(), f
    meta = snapshot.read_json(out / "meta.json")
    assert meta["gw"] == 12 and meta["hours_to_deadline"] == 20.0
    proj = pd.read_csv(out / "projections.csv.gz")
    assert {"proj_next", "ep_next", "news", "cand_proj_next"} <= set(proj.columns)
    assert snapshot.read_json(out / "bootstrap.json.gz")["events"][11]["id"] == 12

    again = snapshot.take_pre(root, squad, now=t1 + dt.timedelta(hours=1), verbose=False)
    assert again is None, "a second snapshot within the gap is skipped"
    later = snapshot.take_pre(root, squad, now=DEADLINE - dt.timedelta(hours=3), verbose=False)
    assert later is not None and later != out and out.exists(), "nothing is overwritten"
    after = snapshot.take_pre(root, squad, now=DEADLINE + dt.timedelta(hours=1), verbose=False)
    assert after is None, "never after the deadline"
    assert len(snapshot.pre_snapshots(12, root)) == 2
    print("pre ok: files written, repeats skipped, nothing overwritten, nothing after the deadline")


def check_post(root: Path) -> None:
    (root / snapshot.ELITE_FILE).write_text(json.dumps([{"entry": 7}, {"entry": 8}]))
    done = snapshot.take_post(root, verbose=False)
    assert done == list(range(1, 11)), done   # GW11 is finished but not data-checked yet
    eo = snapshot.read_json(snapshot.gw_dir(10, root) / "post" / "elite_eo.json")
    assert eo["managers"] == 2 and eo["eo"]["1"] == 200.0 and eo["eo"]["2"] == 100.0
    live = snapshot.read_json(snapshot.gw_dir(3, root) / "post" / "live.json.gz")
    assert live["elements"][0]["stats"]["total_points"] == 3
    calls = dict(CALLS)
    assert snapshot.take_post(root, verbose=False) == [] and CALLS == calls, "no refetch"
    print("post ok: only checked gameweeks, elite EO saved, nothing fetched twice")


def main():
    install_fakes()
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "snapshots"
        squad = Path(d) / "my_squad.json"     # names the fake league does not have
        squad.write_text(json.dumps({"state": {"bank": 0.5, "free_transfers": 1},
                                     "squad": [{"name": "Nobody", "club": "XXX"}]}))
        check_gate(root)
        check_pre(root, squad)
        plan = snapshot.read_json(snapshot.pre_snapshots(12, root)[0] / "plan.json")
        assert "error" in plan, plan
        check_post(root)
    print("\nALL SNAPSHOT CHECKS PASSED")


if __name__ == "__main__":
    main()
