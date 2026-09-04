"""Offline unit tests for the news, transfer and foreign-stats layers.

No network. Everything here is fed hand-written fixtures that mirror the exact
shapes the live APIs return, so the parsers can be changed with confidence.

Run with:  python tests/test_availability.py
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from fplai import external, news, transfers  # noqa: E402

TODAY = dt.date(2026, 8, 24)
# Weekly deadlines from GW1, matching how the real calendar is laid out.
DEADLINES = {gw: dt.date(2026, 8, 21) + dt.timedelta(days=7 * (gw - 1)) for gw in range(1, 39)}
GWS = [2, 3, 4, 5, 6]
CFG = news.NewsConfig(use_web=False)


def element(pid, status="a", news_text="", chance=None, **extra):
    e = {
        "id": pid,
        "status": status,
        "news": news_text,
        "chance_of_playing_next_round": chance,
        "news_added": "2026-08-20",
        "first_name": "Test",
        "second_name": f"Player{pid}",
        "web_name": f"Player{pid}",
        "team": 1,
        "element_type": 3,
        "now_cost": 60,
    }
    e.update(extra)
    return e


def curve_for(el):
    sig = news.parse_fpl_news(el, DEADLINES, TODAY)
    return sig, news.availability_curve(sig, GWS, CFG)


# --------------------------------------------------------------------------- news parsing


def test_fpl_news_parsing():
    sig, _ = curve_for(element(1, "i", "Ankle injury - Expected back 10 Oct"))
    assert sig.category == news.INJURED
    assert sig.return_date == dt.date(2026, 10, 10)
    assert sig.return_gw == 8, sig.return_gw   # GW8's matches start 9 Oct

    sig, _ = curve_for(element(2, "d", "Hamstring injury - 75% chance of playing", 75))
    assert sig.category == news.DOUBT and sig.chance_next == 0.75

    sig, _ = curve_for(element(3, "s", "Suspended until 19 Sep", 0))
    assert sig.category == news.SUSPENDED and sig.return_date == dt.date(2026, 9, 19)

    sig, _ = curve_for(element(4, "u", "Has joined Getafe permanently", 0))
    assert sig.category == news.GONE

    sig, _ = curve_for(element(5, "i", "Personal reasons - Unknown return date", 0))
    assert sig.category == news.PERSONAL

    # A player FPL has de-registered is gone even if the status letter says otherwise.
    sig, _ = curve_for(element(6, "a", "", None, can_select=False))
    assert sig.category == news.GONE

    sig, _ = curve_for(element(7))
    assert sig.category == news.FIT and not sig.flagged

    # Year inference: a December date read in August belongs to this year, and a
    # January one to next year.
    assert news.parse_day_month("3 Dec", TODAY) == dt.date(2026, 12, 3)
    assert news.parse_day_month("3 Jan", TODAY) == dt.date(2027, 1, 3)
    assert news.parse_day_month("nonsense", TODAY) is None
    print("news parsing ok")


def test_availability_curve():
    # The headline behaviour: an injury with a return date must not wipe out the
    # gameweeks after that date.
    _, c = curve_for(element(1, "i", "Knee injury - Expected back 5 Sep", 0))
    assert c[2] == 0.0, c                           # GW2 is played in August
    assert c[3] > 0.0, c                            # GW3's matches run from 4 Sep
    assert c[3] < c[4] < c[5], "a returning player should ramp back up"

    # A suspension is a hard gate, then a clean return to full availability.
    _, c = curve_for(element(2, "s", "Suspended until 19 Sep", 0))
    assert c[2] == c[3] == c[4] == 0.0
    assert c[5] == 1.0, c                           # GW5 deadline 18 Sep, matches 19-20th

    # A departure never comes back.
    _, c = curve_for(element(3, "u", "Has joined Como permanently", 0))
    assert set(c.values()) == {0.0}

    # A doubt applies now and fades.
    _, c = curve_for(element(4, "d", "Knock - 75% chance of playing", 75))
    assert abs(c[2] - 0.75) < 1e-9
    assert c[2] < c[3] < c[4] < 1.0

    # An unknown-return injury is out now but not written off for the whole horizon.
    _, c = curve_for(element(5, "i", "Groin injury - Unknown return date", 0))
    assert c[2] == 0.0 and c[6] > 0.0

    # A return date past the end of the schedule means out throughout.
    short = {1: dt.date(2026, 8, 21), 2: dt.date(2026, 8, 28)}
    sig = news.parse_fpl_news(element(6, "i", "Leg injury - Expected back 28 Nov", 0), short, TODAY)
    assert sig.return_beyond_schedule
    assert set(news.availability_curve(sig, [1, 2], CFG).values()) == {0.0}

    # A fit player is fit.
    _, c = curve_for(element(7))
    assert set(c.values()) == {1.0}
    print("availability curves ok")


def test_web_news_classification():
    """The nearest keyword decides, so one paragraph can describe two situations."""
    text = news.normalise(
        "a testing time for the brighton manager who has said goodbye to danny welbeck "
        "and james milner among others  he was without carlos baleba officially because "
        "of injuries although their proposed moves to other clubs"
    )
    at_welbeck = text.index("danny welbeck")
    at_baleba = text.index("carlos baleba")
    assert news.classify_near(text, at_welbeck) == news.GONE
    assert news.classify_near(text, at_baleba) == news.INJURED

    # A player coming back must not be read as a fresh injury.
    assert news.classify_headline("Smith returns to training after injury lay-off") is None
    assert news.classify_headline("Smith ruled out for six weeks with hamstring injury") == news.INJURED
    assert news.classify_headline("Arsenal beat Chelsea 2-0 at the Emirates") is None
    print("web classification ok")


def test_stale_departure_is_ignored():
    """A story about the transfer that already happened is not an absence."""
    cfg = news.NewsConfig()
    arrived = element(1, team_join_date=(TODAY - dt.timedelta(days=23)).isoformat())
    settled = element(2, team_join_date="2021-07-01")

    # "Brighton said goodbye to Welbeck" -- he is at his new club, and FPL knows.
    assert news.stale_departure(arrived, news.GONE, cfg, TODAY)
    # The same story about a long-serving player is a live exit rumour: keep it.
    assert not news.stale_departure(settled, news.GONE, cfg, TODAY)
    # Only departures are suppressed; an injury report on a new signing still counts.
    assert not news.stale_departure(arrived, news.INJURED, cfg, TODAY)
    assert not news.stale_departure(element(3), news.GONE, cfg, TODAY)
    print("stale departure suppression ok")


def test_web_signal_is_bounded():
    """An unconfirmed report can shade a player down but never rule him out."""
    cfg = news.NewsConfig()
    sig = news.Availability(player_id=1, category=news.FIT, web_category=news.INJURED)
    out = news.apply_web_signal({g: 1.0 for g in GWS}, sig, cfg, GWS)
    assert out[GWS[0]] == cfg.web_penalty[news.INJURED]
    assert out[GWS[0]] >= cfg.web_floor
    # Only the near term is affected; later gameweeks are left alone.
    assert out[GWS[cfg.web_effect_gws]] == 1.0

    # It can never resurrect someone FPL says is unavailable.
    gone = news.Availability(player_id=2, category=news.GONE, web_category=news.DOUBT)
    out = news.apply_web_signal({g: 0.0 for g in GWS}, gone, cfg, GWS)
    assert set(out.values()) == {0.0}
    print("web signal bounding ok")


# --------------------------------------------------------------------------- transfers


def test_transfer_detection():
    cfg = transfers.SquadConfig(new_signing_days=90)
    els = [
        element(1, team_join_date="2026-07-20"),   # this summer
        element(2, team_join_date="2026-01-15"),   # last January, long settled by now
        element(3, team_join_date="2021-07-01"),   # a veteran of the club
        element(4, team_join_date=None),           # FPL has no date
    ]
    moves = transfers.classify_moves(els, TODAY, cfg)
    assert moves[1]["is_new_signing"] and moves[1]["window"] == transfers.SUMMER
    assert not moves[2]["is_new_signing"], "a January move is old news by August"
    assert not moves[3]["is_new_signing"]
    assert moves[3]["days_at_club"] > 1800
    assert moves[4]["days_at_club"] is None and not moves[4]["is_new_signing"]

    # A January signing looked at in February is new, and settles more slowly.
    feb = dt.date(2026, 2, 1)
    jan_moves = transfers.classify_moves([element(2, team_join_date="2026-01-15")], feb, cfg)
    assert jan_moves[2]["is_new_signing"] and jan_moves[2]["window"] == transfers.JANUARY
    assert (transfers.settling_multiplier(jan_moves[2], 0, cfg)
            < transfers.settling_multiplier(moves[1], 0, cfg))
    print("transfer detection ok")


def test_settling_decays_with_appearances():
    cfg = transfers.SquadConfig()
    move = {"is_new_signing": True, "window": transfers.SUMMER}
    none, some, many = (transfers.settling_multiplier(move, n, cfg) for n in (0, 2, 8))
    assert none == cfg.settle_floor
    assert none < some < many <= 1.0
    assert many > 0.97, "after enough appearances the discount should be gone"
    assert transfers.settling_multiplier({"is_new_signing": False}, 0, cfg) == 1.0
    print("settling-in decay ok")


def test_minutes_profile_uses_the_opening_weekend():
    """One gameweek of evidence is still evidence -- and a sub is not a starter."""
    from fplai import model
    past = [{"minutes": 3000, "total_points": 150}]      # a nailed starter last season

    def p_start(logs, **kw):
        return model._minutes_profile(past, logs, 8.0, **kw)[0]

    nailed = p_start([])                                  # nothing played yet: the prior
    left_out = p_start([{"minutes": 0}])                  # not in the matchday squad
    off_bench = p_start([{"minutes": 20}])                # came on
    started = p_start([{"minutes": 90}])                  # started

    assert left_out < off_bench < started, (left_out, off_bench, started)
    assert left_out < nailed, "being left out must move him down from the prior"
    assert started >= nailed - 1e-9, "starting must not move him down"
    # The reaction is proportionate: one omission is a dent, not a demolition.
    assert 0.4 < left_out < 0.75, left_out

    # A new signing's record elsewhere counts for less at his new club.
    assert p_start([], is_new_signing=True) < nailed
    print("minutes profile ok")


def test_matches_since_joining():
    from fplai import model
    log = [
        {"minutes": 90, "kickoff_time": "2026-08-01T15:00:00Z"},   # before joining
        {"minutes": 0, "kickoff_time": "2026-08-22T15:00:00Z"},    # unused sub
        {"minutes": 65, "kickoff_time": "2026-08-23T15:00:00Z"},   # after joining
    ]
    assert model.matches_since_joining(log, "2026-08-10") == 1
    assert model.matches_since_joining(log, None) == 2, "no join date: count every appearance"
    print("club appearance counting ok")


# --------------------------------------------------------------------------- depth chart


def _depth_frame(rows):
    return pd.DataFrame(rows)


def test_depth_chart_splits_the_shirt():
    cfg = transfers.SquadConfig()
    # Three keepers at one club, all of whom look like starters on their own history.
    df = _depth_frame([
        {"team_id": 1, "pos_id": 1, "name": "first", "p_start_raw": 0.90, "avail_next": 1.0},
        {"team_id": 1, "pos_id": 1, "name": "second", "p_start_raw": 0.70, "avail_next": 1.0},
        {"team_id": 1, "pos_id": 1, "name": "third", "p_start_raw": 0.60, "avail_next": 1.0},
    ])
    ps = transfers.depth_chart(df, cfg)
    assert ps.sum() < 1.6, f"only one keeper plays: {list(ps)}"
    assert ps.iloc[0] > ps.iloc[1] > ps.iloc[2], "the pecking order must be preserved"
    assert (ps < df["p_start_raw"]).all(), "competition must reduce, not inflate"


def test_depth_chart_promotes_the_deputy():
    """When the first choice is unavailable, his minutes go to the understudy."""
    cfg = transfers.SquadConfig()
    rows = [
        {"team_id": 1, "pos_id": 1, "name": "first", "p_start_raw": 0.95, "avail_next": 1.0},
        {"team_id": 1, "pos_id": 1, "name": "deputy", "p_start_raw": 0.35, "avail_next": 1.0},
    ]
    healthy = transfers.depth_chart(_depth_frame(rows), cfg)

    rows[0]["avail_next"] = 0.0                      # first choice injured
    injured = transfers.depth_chart(_depth_frame(rows), cfg)

    assert injured.iloc[0] == 0.0
    assert injured.iloc[1] > healthy.iloc[1], "the deputy should start more often"
    assert injured.iloc[1] > 0.6, f"and should be close to a certainty: {injured.iloc[1]}"


def test_depth_chart_cannot_exceed_availability():
    """Filling a club's empty places must not override a player's own doubt.

    Villa's other forwards being injured is not a reason to promote a doubtful
    striker back to a near-certainty: renormalising scales players up, and
    availability has to be a hard ceiling on the result.
    """
    cfg = transfers.SquadConfig()
    df = _depth_frame([
        {"team_id": 1, "pos_id": 4, "name": "doubtful", "p_start_raw": 0.95, "avail_next": 0.55},
        {"team_id": 1, "pos_id": 4, "name": "injured", "p_start_raw": 0.60, "avail_next": 0.0},
        {"team_id": 1, "pos_id": 4, "name": "alsoinjured", "p_start_raw": 0.50, "avail_next": 0.0},
    ])
    ps = transfers.depth_chart(df, cfg)
    assert ps.iloc[0] <= 0.55 + 1e-9, f"availability must cap the share: {ps.iloc[0]}"
    assert ps.iloc[1] == 0.0 and ps.iloc[2] == 0.0


def test_depth_chart_is_club_local():
    """Competition is between team-mates, not across the league."""
    cfg = transfers.SquadConfig()
    df = _depth_frame([
        {"team_id": 1, "pos_id": 1, "name": "a", "p_start_raw": 0.9, "avail_next": 1.0},
        {"team_id": 1, "pos_id": 1, "name": "b", "p_start_raw": 0.9, "avail_next": 1.0},
        {"team_id": 2, "pos_id": 1, "name": "c", "p_start_raw": 0.9, "avail_next": 1.0},
    ])
    ps = transfers.depth_chart(df, cfg)
    assert ps.iloc[2] > ps.iloc[0], "an uncontested keeper keeps his place"
    assert abs(ps.iloc[0] - ps.iloc[1]) < 1e-9, "equal team-mates split it evenly"


def test_rotation_weight_zero_disables():
    cfg = transfers.SquadConfig(rotation_weight=0.0)
    df = _depth_frame([
        {"team_id": 1, "pos_id": 1, "name": "a", "p_start_raw": 0.9, "avail_next": 1.0},
        {"team_id": 1, "pos_id": 1, "name": "b", "p_start_raw": 0.9, "avail_next": 1.0},
    ])
    ps = transfers.depth_chart(df, cfg)
    assert abs(ps.iloc[0] - 0.9) < 1e-9 and abs(ps.iloc[1] - 0.9) < 1e-9
    print("depth chart ok")


# --------------------------------------------------------------------------- external stats


def test_external_name_matching():
    """Full names only, accents folded, ambiguous names dropped rather than guessed."""
    index = {
        "erling haaland": {"minutes": 3000, "goal_rate": 0.9, "assist_rate": 0.2,
                           "league": "EPL", "games": 34},
        "julian alvarez": {"minutes": 2000, "goal_rate": 0.6, "assist_rate": 0.3,
                           "league": "La_liga", "games": 30},
    }
    els = [
        {"id": 1, "first_name": "Erling", "second_name": "Haaland"},
        {"id": 2, "first_name": "Julián", "second_name": "Álvarez"},   # accents must fold
        {"id": 3, "first_name": "Someone", "second_name": "Unknown"},
        {"id": 4, "first_name": "", "second_name": "Haaland"},          # surname alone: no match
    ]
    matched = external.match_players(els, index)
    assert set(matched) == {1, 2}, matched
    assert matched[2]["league"] == "La_liga"
    print("external matching ok")


def test_external_season_default():
    # Mid-season and pre-season should both reach for the last completed campaign.
    assert external.default_season(dt.date(2026, 8, 24)) == 2025
    assert external.default_season(dt.date(2027, 3, 1)) == 2025
    print("external season default ok")


def main():
    test_fpl_news_parsing()
    test_availability_curve()
    test_web_news_classification()
    test_stale_departure_is_ignored()
    test_web_signal_is_bounded()
    test_transfer_detection()
    test_settling_decays_with_appearances()
    test_minutes_profile_uses_the_opening_weekend()
    test_matches_since_joining()
    test_depth_chart_splits_the_shirt()
    test_depth_chart_promotes_the_deputy()
    test_depth_chart_cannot_exceed_availability()
    test_depth_chart_is_club_local()
    test_rotation_weight_zero_disables()
    test_external_name_matching()
    test_external_season_default()
    print("\nALL AVAILABILITY CHECKS PASSED")


if __name__ == "__main__":
    main()
