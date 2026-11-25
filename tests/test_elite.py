"""Offline checks for the elite-manager analysis. Run with:  python tests/test_elite.py"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fplai import elite  # noqa: E402


def squad(captain: int, triple: bool = False) -> list[dict]:
    picks = []
    for el in range(1, 16):
        mult = 0 if el > 11 else (3 if triple else 2) if el == captain else 1
        picks.append({"element": el, "multiplier": mult, "is_captain": el == captain})
    return picks


def check_effective_ownership() -> None:
    eo = elite.effective_ownership([squad(1), squad(2), squad(1, triple=True), squad(3)])
    assert math.isclose(eo[1], 100 * (2 + 1 + 3 + 1) / 4)
    assert math.isclose(eo[4], 100.0)
    assert 12 not in eo or eo[12] == 0.0, "the bench earns nothing"
    assert elite.effective_ownership([]) == {}
    print(f"effective ownership ok: captain-heavy player at {eo[1]:.0f}%")


def check_consistency() -> None:
    assert elite.consistency_score([{"rank": 10}, {"rank": 20}], min_seasons=3) is None
    s = elite.consistency_score([{"rank": 1000}, {"rank": 10}, {"rank": 100}], min_seasons=3)
    assert math.isclose(s, 2.0)
    worse = elite.consistency_score([{"rank": 100000}, {"rank": 10}, {"rank": 50000}], min_seasons=3)
    assert worse > s
    print("consistency ok: median log rank, lower is better")


def check_tables() -> None:
    decisions = {
        11: {"picks": {1: {"picks": squad(1)}, 2: {"picks": squad(2), "active_chip": "wildcard"}},
             "history": {"current": [
                 {"event": 1, "points": 70, "event_transfers": 0, "event_transfers_cost": 0, "points_on_bench": 3},
                 {"event": 2, "points": 60, "event_transfers": 9, "event_transfers_cost": 0, "points_on_bench": 1}]}},
        22: {"picks": {1: {"picks": squad(1)}, 2: {"picks": squad(1)}},
             "history": {"current": [
                 {"event": 1, "points": 65, "event_transfers": 0, "event_transfers_cost": 0, "points_on_bench": 0},
                 {"event": 2, "points": 50, "event_transfers": 2, "event_transfers_cost": 4, "points_on_bench": 6}]}},
    }
    points = {1: {1: 12, 2: 3}, 2: {1: 2, 2: 8}}
    mgw = elite.manager_gameweeks(decisions, points, {11: "current", 22: "consistent"})
    assert len(mgw) == 4
    assert mgw.loc[(mgw.entry == 11) & (mgw.gw == 2), "captain_points"].item() == 8
    s = elite.summarise(mgw)
    assert s["managers"] == 2 and math.isclose(s["hits_per_manager"], 0.5)
    assert s["chips"]["wildcard"]["count"] == 1
    by_gw = elite.eo_by_gw(decisions)
    assert math.isclose(by_gw[1][1], 200.0) and math.isclose(by_gw[2][2], 150.0)
    print("manager tables ok: points, hits, chips and captaincy line up")


def main():
    check_effective_ownership()
    check_consistency()
    check_tables()
    print("\nALL ELITE CHECKS PASSED")


if __name__ == "__main__":
    main()
