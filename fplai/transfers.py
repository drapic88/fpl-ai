"""Transfers in, and the fight for a starting shirt.

Two questions the projection engine could not previously answer:

**"Has this player just moved?"** The bootstrap carries `team_join_date` for
almost every player, which is the exact date they joined their *current*
Premier League club. That makes both windows visible for free: the summer
window before a season-start squad pick, and the January window before the
mid-season deadline. A player who signed three weeks ago has no Premier League
history at this club, and the little history he does have describes a different
team's tactics.

**"Will he actually start?"** Nothing in the old model stopped it buying a
backup goalkeeper. Expected minutes were derived from a player's own history in
isolation, so two keepers at the same club could both project as starters, and
a big-money arrival never displaced the incumbent he was bought to replace.

The fix is a depth chart: within each (club, position) group, raw start
probabilities are sharpened and renormalised so the expected number of starters
matches a realistic team shape. Competition becomes zero-sum, which is what it
actually is. Unavailable players drop out of the contest, which correctly
promotes the deputy behind an injured starter.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Expected starters per position across a typical Premier League matchday.
#: Sums to 11. Clubs vary by formation, but the aggregate is stable.
FORMATION_SLOTS = {1: 1.0, 2: 4.0, 3: 3.9, 4: 2.1}

SUMMER = "summer"
JANUARY = "january"
SETTLED = "settled"


@dataclass
class SquadConfig:
    """Knobs for the transfer and rotation model."""

    # --- how recently must a player have joined to count as a new signing? ---
    new_signing_days: int = 90

    # --- settling in ---------------------------------------------------------
    # A new arrival's expected minutes are multiplied by this at the moment he
    # joins: new league, new system, often mid-way through a pre-season.
    settle_floor: float = 0.78
    # ...recovering to 1.0 as he actually plays. After this many appearances for
    # the club, observed minutes are trusted and the discount is essentially gone.
    settle_matches: float = 3.0
    # A January arrival lands mid-season with no pre-season, so settles slower.
    january_settle_floor: float = 0.70

    # --- depth chart ---------------------------------------------------------
    # How hard to renormalise start probabilities within a (club, position) group.
    # 0 disables the depth chart entirely; 1 enforces the formation shape exactly.
    rotation_weight: float = 0.75
    # Sharpening exponent applied before renormalising. Above 1, the better-placed
    # player in a group takes proportionally more of the available starts, which is
    # how a settled first choice actually behaves (especially in goal).
    rotation_sharpen: float = 1.35
    # Nobody is ever a certainty.
    max_start_prob: float = 0.97


# --------------------------------------------------------------------------- transfer detection


def parse_join_date(element: dict) -> dt.date | None:
    raw = element.get("team_join_date")
    if not raw:
        return None
    try:
        return dt.date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


def transfer_window(joined: dt.date | None) -> str:
    """Which window a move belongs to. January arrivals behave differently."""
    if joined is None:
        return SETTLED
    return JANUARY if joined.month in (1, 2) else SUMMER


def classify_moves(elements: list[dict], today: dt.date, cfg: SquadConfig) -> dict[int, dict]:
    """player id -> {joined, days_at_club, is_new_signing, window}.

    `days_at_club` is None when FPL has no join date, which it leaves blank for a
    handful of players; those are treated as settled rather than guessed at.
    """
    out: dict[int, dict] = {}
    for e in elements:
        joined = parse_join_date(e)
        days = (today - joined).days if joined else None
        out[int(e["id"])] = {
            "joined": joined.isoformat() if joined else None,
            "days_at_club": days,
            "is_new_signing": bool(days is not None and 0 <= days <= cfg.new_signing_days),
            "window": transfer_window(joined) if days is not None and days <= cfg.new_signing_days
            else SETTLED,
        }
    return out


def settling_multiplier(move: dict, matches_for_club: float, cfg: SquadConfig) -> float:
    """Minutes discount for a player who has only just arrived.

    Decays away as he racks up appearances: once we have watched him play for
    this club a few times, his own match log is better evidence than any prior
    about settling in, so the discount should get out of the way.
    """
    if not move.get("is_new_signing"):
        return 1.0
    floor = cfg.january_settle_floor if move.get("window") == JANUARY else cfg.settle_floor
    seen = max(0.0, float(matches_for_club))
    # exp decay: full discount at 0 appearances, ~5% of it left after settle_matches
    remaining = float(np.exp(-seen / max(0.5, cfg.settle_matches / 3.0)))
    return float(np.clip(1.0 - (1.0 - floor) * remaining, floor, 1.0))


# --------------------------------------------------------------------------- depth chart


def depth_chart(
    df: pd.DataFrame,
    cfg: SquadConfig,
    p_start_col: str = "p_start_raw",
    avail_col: str = "avail_next",
) -> pd.Series:
    """Renormalise start probabilities so clubs cannot field 3 starting keepers.

    Within each (club, position) the raw probabilities are sharpened, scaled so
    they sum to the number of slots that position actually offers, then blended
    back toward the raw value by `rotation_weight`. Availability enters the
    contest -- an injured first choice frees his slot for the deputy.

    Returns a Series aligned to `df.index`.
    """
    raw = df[p_start_col].astype(float).fillna(0.0).clip(0.0, 1.0)
    avail = df[avail_col].astype(float).fillna(1.0).clip(0.0, 1.0) if avail_col in df else 1.0
    contest = (raw * avail).clip(0.0, 1.0)

    if cfg.rotation_weight <= 0:
        return contest.clip(0.0, cfg.max_start_prob)

    shares = contest.copy()
    for (_, pos_id), grp in df.groupby(["team_id", "pos_id"], sort=False):
        slots = FORMATION_SLOTS.get(int(pos_id), 1.0)
        score = np.power(contest.loc[grp.index], cfg.rotation_sharpen)
        total = float(score.sum())
        if total <= 1e-9:
            # Every candidate is unavailable; leave the raw values alone rather than
            # dividing by zero to conjure a starter out of an empty position group.
            continue
        shares.loc[grp.index] = score * (slots / total)

    blended = cfg.rotation_weight * shares + (1.0 - cfg.rotation_weight) * contest
    # Renormalising can scale a player *up* to fill a club's empty places -- which is
    # right when a rival is merely out of form, and wrong when the player himself is
    # doubtful. Nobody starts more often than he is available to, so availability is
    # a hard ceiling: without it, a club whose other forwards are all injured would
    # promote its own doubtful striker back to a near-certainty.
    return blended.clip(0.0, cfg.max_start_prob).clip(upper=avail)
