"""Offline unit tests for the scheduled squad watcher.

No network. The watcher's own value is in two places -- resolving a squad file
onto FPL elements, and deciding whether today's news is actually different from
yesterday's -- so those are what is pinned here. An unattended daily task that
either loses a player or cries wolf every morning is worse than no task at all.

Run with:  python tests/test_watch.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import watch_availability as watch  # noqa: E402
from fplai import news  # noqa: E402


def _run(argv):
    """Call the watcher's entry point with argv, returning its exit code.

    Only used for the failure paths, which are reached before any network call.
    """
    saved = sys.argv
    sys.argv = ["watch_availability.py"] + argv
    try:
        return watch.main()
    finally:
        sys.argv = saved

TEAMS = {
    1: {"id": 1, "name": "Crystal Palace", "short_name": "CRY"},
    2: {"id": 2, "name": "Manchester City", "short_name": "MCI"},
}


def element(pid, web_name, team, first="Test", second="Player"):
    return {
        "id": pid, "web_name": web_name, "team": team,
        "first_name": first, "second_name": second, "element_type": 3,
    }


ELEMENTS = [
    element(1, "Mateta", 1, "Jean-Philippe", "Mateta"),
    element(2, "Sarr", 1, "Ismaila", "Sarr"),
    element(3, "Guéhi", 2, "Marc", "Guéhi"),
    element(4, "O'Reilly", 2, "Nico", "O'Reilly"),
    # Same surname, different club: the club must be what separates them.
    element(5, "Sarr", 2, "Malick", "Sarr"),
    # Letters NFKD will not touch, spelt the way FPL spells them.
    element(6, "Gro\u00df", 1, "Pascal", "Gro\u00df"),
    element(7, "\u00d8degaard", 1, "Martin", "\u00d8degaard"),
]


def test_resolution_matches_accents_and_apostrophes():
    squad = [
        {"name": "Guehi", "club": "MCI", "pos": "DEF"},
        {"name": "OReilly", "club": "MCI", "pos": "DEF"},
    ]
    found, missing = watch.resolve_squad(ELEMENTS, TEAMS, squad)
    assert not missing, missing
    assert [e["id"] for _, e in found] == [3, 4]
    print("accented and apostrophised names resolve ok")


def test_resolution_folds_letters_nfkd_leaves_alone():
    """NFKD only decomposes accents, so 'Gro\u00df' and '\u00d8degaard' come through it
    unchanged and a squad file typed on an English keyboard misses them. Both
    scripts resolve the same file, so a miss here silently drops a player from
    the watcher as well as costing the planner a transfer."""
    squad = [
        {"name": "Gross", "club": "CRY", "pos": "MID"},
        {"name": "Odegaard", "club": "CRY", "pos": "MID"},
    ]
    found, missing = watch.resolve_squad(ELEMENTS, TEAMS, squad)
    assert not missing, missing
    assert [e["id"] for _, e in found] == [6, 7]

    # The FPL spelling has to keep working -- the fold is a second way in, not a
    # replacement for the first.
    found, missing = watch.resolve_squad(
        ELEMENTS, TEAMS, [{"name": "Gro\u00df", "club": "CRY", "pos": "MID"}])
    assert not missing and found[0][1]["id"] == 6

    assert watch.norm("Gro\u00df") == watch.norm("Gross") == "gross"
    assert watch.norm("\u00d8degaard") == watch.norm("Odegaard") == "odegaard"
    # Capitals fold too: the sharp S and the slashed O both have upper-case forms.
    assert watch.norm("GRO\u1e9e") == "gross"
    assert watch.norm("\u00d8DEGAARD") == "odegaard"
    print("letters NFKD leaves alone resolve ok")


def test_every_matcher_normalises_identically():
    """Four call sites compare a typed or scraped name against FPL's spelling:
    the planner and watcher resolve the squad file, the news layer matches
    headlines, the external layer matches Understat rows. They share one fold
    table, and the day they disagree is the day a player resolves in one and
    vanishes in another."""
    import plan_week
    from fplai import external, news

    for name in ["Gro\u00df", "Gross", "\u00d8degaard", "Odegaard", "Gu\u00e9hi", "Guehi",
                 "O'Reilly", "Jo\u00e3o Pedro", "  Sels  ", "Havertz"]:
        assert plan_week.norm(name) == watch.norm(name), name
        # The fplai pair also strip spacing and punctuation, so compare the fold
        # itself rather than the whole pipeline.
        assert news.normalise(name) == external.normalise(name), name
    assert news.normalise("Gro\u00df") == watch.norm("Gross") == "gross"
    assert external.normalise("\u00d8degaard") == plan_week.norm("Odegaard") == "odegaard"
    print("all four matchers normalise identically ok")


def test_resolution_is_club_local():
    """Two Sarrs in the league must not collapse into one another."""
    found, missing = watch.resolve_squad(
        ELEMENTS, TEAMS, [{"name": "Sarr", "club": "CRY", "pos": "MID"}])
    assert not missing and found[0][1]["id"] == 2
    found, missing = watch.resolve_squad(
        ELEMENTS, TEAMS, [{"name": "Sarr", "club": "MCI", "pos": "DEF"}])
    assert not missing and found[0][1]["id"] == 5
    print("club-local resolution ok")


def test_unresolvable_player_is_reported_not_dropped():
    """A typo has to surface. Silently watching fourteen players is the failure
    mode that costs a manager a blank."""
    found, missing = watch.resolve_squad(
        ELEMENTS, TEAMS, [{"name": "Nonexistent", "club": "CRY", "pos": "MID"},
                          {"name": "Mateta", "club": "CRY", "pos": "FWD"}])
    assert len(found) == 1 and len(missing) == 1
    assert "Nonexistent" in missing[0]
    print("unresolved players are reported ok")


def test_broken_squad_file_never_looks_like_a_quiet_week(tmp):
    """The exit code that matters most.

    Exit 1 means "nothing has changed, stay quiet". A squad file that will not
    parse must never produce it, or an unattended caller reports all-clear on a
    week it never actually checked. Python's own exit code for an uncaught
    exception is 1, which is exactly the collision being guarded against here.
    """
    bad = tmp / "broken.json"
    bad.write_text('{"squad": [{"name": "Sels", "club": "NFO"},]}', encoding="utf-8")
    assert watch.main.__module__  # sanity: module imported
    code = _run(["--squad", str(bad), "--state", str(tmp / "s.json")])
    assert code == 2, code

    empty = tmp / "empty.json"
    empty.write_text('{"updated": "2026-09-01"}', encoding="utf-8")
    assert _run(["--squad", str(empty), "--state", str(tmp / "s.json")]) == 2

    assert _run(["--squad", str(tmp / "absent.json"), "--state", str(tmp / "s.json")]) == 2
    print("a broken squad file exits 2, not 1, ok")


def test_non_breaking_spaces_are_tolerated(tmp):
    """JSON copied out of a browser or a chat window carries U+00A0 indentation.

    It means nothing in a squad file, and the error `json` raises for it points at
    a line that looks perfectly correct, so it is normalised rather than reported.
    """
    f = tmp / "nbsp.json"
    f.write_text('{\n\u00a0\u00a0"squad": [{"name": "Sels", "club": "NFO", "pos": "GKP"}]\n}',
                 encoding="utf-8")
    data = watch.load_squad(f)
    assert data["squad"][0]["name"] == "Sels"
    print("non-breaking spaces in the squad file are tolerated ok")


def test_fielding_roles_mark_the_xi_and_the_armband():
    """A doubt is not equally expensive everywhere, so the report has to know
    whether a player is in the XI, on the bench, or wearing the armband."""
    data = {"fielding": {
        "xi": ["Sels", "Guehi", "B.Fernandes", "Joao Pedro"],
        "bench_order": ["Wilson", "Cash", "Welbeck"],
        "bench_gk": "Dubravka",
        "captain": "B.Fernandes",
        "vice": "Joao Pedro",
    }}
    roles = watch.fielding_roles(data)
    assert roles[watch.norm("B.Fernandes")] == ("XI", "C")
    # The vice is named without its accent in `fielding` but with one in `squad`;
    # both have to land on the same player.
    assert roles[watch.norm("João Pedro")] == ("XI", "V")
    assert roles[watch.norm("Sels")] == ("XI", "")
    assert roles[watch.norm("Wilson")] == ("B1", "")
    assert roles[watch.norm("Welbeck")] == ("B3", "")
    assert roles[watch.norm("Dubravka")] == ("BGK", "")
    # A squad file with no `fielding` block is still perfectly usable.
    assert watch.fielding_roles({}) == {}
    print("fielding roles and armbands ok")


def test_xi_sorts_ahead_of_the_bench():
    order = [watch.ROLE_ORDER[r] for r in ("XI", "B1", "B2", "B3", "BGK")]
    assert order == sorted(order) and watch.ROLE_ORDER[""] > watch.ROLE_ORDER["BGK"]
    print("XI sorts ahead of the bench ok")


def _record(**over):
    base = {
        "id": 1, "name": "Mateta", "club": "CRY", "pos": "FWD",
        "category": news.FIT, "summary": "available", "chance_next": None,
        "return_gw": None, "out_all_season": False, "avail_next": 1.0,
        "news": "", "news_age_days": None, "web_category": "",
        "web_headline": "", "web_age_days": None, "needs_check": False,
    }
    base.update(over)
    return base


def test_ageing_news_is_not_a_change():
    """The one that decides whether the task is readable: an unchanged injury
    gets a day older every morning, and that must not read as news."""
    yesterday = _record(category=news.INJURED, news="Hamstring injury - Expected back 11 Oct",
                        news_age_days=3.0, web_age_days=2.0)
    today = dict(yesterday, news_age_days=4.0, web_age_days=3.0)
    assert watch.fingerprint(yesterday) == watch.fingerprint(today)
    print("ageing news alone is not a change ok")


def test_real_transitions_are_changes():
    fit = _record()
    cases = [
        (_record(category=news.DOUBT, chance_next=0.75), "fit -> doubt"),
        (_record(news="Knock - 75% chance of playing"), "FPL news added"),
        (_record(web_category=news.INJURED, web_headline="Scan awaited"), "headline: injured"),
        (_record(return_gw=6), "return GW? -> GW6"),
    ]
    for new, expected in cases:
        assert watch.fingerprint(fit) != watch.fingerprint(new)
        got = watch.describe_change(fit, new)
        assert expected in got, f"{expected!r} not in {got!r}"
    print("real transitions are reported ok")


def test_clearing_a_flag_is_good_news_and_says_so():
    """A player coming back changes lineups as much as one breaking down."""
    hurt = _record(category=news.INJURED, news="Ankle injury - Expected back 14 Sep",
                   chance_next=0.0, avail_next=0.0, needs_check=True)
    back = _record()
    change = watch.describe_change(hurt, back)
    assert "injured -> fit" in change and "FPL news cleared" in change
    print("cleared flags are reported as such ok")


def test_needs_check_covers_the_gaps_fpl_leaves():
    gws = [3, 4, 5]
    cases = {
        news.DOUBT: dict(chance_next=0.75),
        news.INJURED: {},
        news.SUSPENDED: {},
        news.UNSETTLED: {},   # FPL has no status letter for this at all
        news.GONE: {},
    }
    for category, extra in cases.items():
        sig = news.Availability(player_id=1, category=category, **extra)
        sig.curve = news.availability_curve(sig, gws, news.NewsConfig(use_web=False))
        rec = watch.record({"pos": "MID"}, ELEMENTS[0], sig, gws[0], TEAMS)
        assert rec["needs_check"], category

    # A fit player FPL says nothing about needs no second look...
    fit = news.Availability(player_id=1, category=news.FIT)
    fit.curve = {gw: 1.0 for gw in gws}
    assert not watch.record({"pos": "MID"}, ELEMENTS[0], fit, gws[0], TEAMS)["needs_check"]

    # ...but one carrying only a web rumour does, since that is the whole point
    # of reading the feeds ahead of FPL's flags.
    rumour = news.Availability(player_id=1, category=news.FIT, web_category=news.UNSETTLED,
                              web_headline="Left out again, says manager")
    rumour.curve = {gw: 1.0 for gw in gws}
    assert watch.record({"pos": "MID"}, ELEMENTS[0], rumour, gws[0], TEAMS)["needs_check"]
    print("needs_check covers rumours and omissions ok")


def test_availability_percentage_is_the_coming_gameweek():
    """The reported percentage must be this week's number, not the horizon's --
    an injury with an October return date still plays in November."""
    sig = news.Availability(player_id=1, category=news.INJURED, return_gw=6)
    sig.curve = news.availability_curve(sig, [3, 4, 5, 6, 7], news.NewsConfig(use_web=False))
    assert watch.record({"pos": "FWD"}, ELEMENTS[0], sig, 3, TEAMS)["avail_next"] == 0.0
    assert watch.record({"pos": "FWD"}, ELEMENTS[0], sig, 6, TEAMS)["avail_next"] > 0.0
    print("reported availability is per gameweek ok")


def main():
    with tempfile.TemporaryDirectory() as d:
        test_broken_squad_file_never_looks_like_a_quiet_week(Path(d))
        test_non_breaking_spaces_are_tolerated(Path(d))
    test_fielding_roles_mark_the_xi_and_the_armband()
    test_xi_sorts_ahead_of_the_bench()
    test_resolution_matches_accents_and_apostrophes()
    test_resolution_folds_letters_nfkd_leaves_alone()
    test_every_matcher_normalises_identically()
    test_resolution_is_club_local()
    test_unresolvable_player_is_reported_not_dropped()
    test_ageing_news_is_not_a_change()
    test_real_transitions_are_changes()
    test_clearing_a_flag_is_good_news_and_says_so()
    test_needs_check_covers_the_gaps_fpl_leaves()
    test_availability_percentage_is_the_coming_gameweek()
    print("\nALL WATCHER CHECKS PASSED")


if __name__ == "__main__":
    main()
