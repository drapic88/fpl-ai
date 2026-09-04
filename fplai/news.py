"""Availability intelligence: what the news actually implies about playing time.

Two sources, deliberately weighted very differently.

1. **The FPL API's own flags** (`status`, `news`, `news_added`,
   `chance_of_playing_next_round`). These are authoritative -- FPL writes them
   from club sources -- and they decide whether a player can score at all. The
   strings are highly regular, e.g.

       "Ankle injury - Expected back 10 Oct"
       "Hamstring injury - 75% chance of playing"
       "Suspended until 19 Sep"
       "Has joined Getafe permanently"
       "Personal reasons - Unknown return date"

   Parsing them yields a *return date*, which is the whole point: an injury
   with a known return date must not zero a player out for gameweeks that fall
   after he is back.

2. **Public football news feeds** (BBC Sport, the Guardian, Sky). These break
   before FPL updates its flags, but they are unstructured and often report
   rumour. So they are only ever allowed to *downgrade* a player, never to
   clear one, and never below `web_floor`. They inform; they do not decide.

The output is a per-gameweek availability multiplier in [0, 1] for every
player, which `model.build` applies to expected minutes.
"""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import requests

from . import api

# --------------------------------------------------------------------------- categories

FIT = "fit"
DOUBT = "doubt"
INJURED = "injured"
SUSPENDED = "suspended"
PERSONAL = "personal"
#: Fit, registered, and still may not play: agitating for a move, frozen out, or
#: left out of the matchday squad. FPL carries no flag for any of this -- a player
#: refusing to play reads as perfectly available right up until he doesn't play.
UNSETTLED = "unsettled"
GONE = "gone"          # transferred out, on loan elsewhere, or not registered
UNKNOWN = "unknown"

STATUS_CATEGORY = {
    "a": FIT,
    "d": DOUBT,
    "i": INJURED,
    "s": SUSPENDED,
    "u": GONE,
    "n": GONE,
}


@dataclass
class NewsConfig:
    """Knobs for turning news into availability. Meant to be argued with, not sacred."""

    # Injury with a known return date: availability in the return GW and the two after.
    # A player back from a long lay-off rarely plays 90 minutes immediately.
    injury_ramp: tuple[float, ...] = (0.55, 0.80, 0.92)
    # Injury with "Unknown return date": a hazard curve over the horizon, indexed from
    # the first planned gameweek. The median FPL injury of this kind runs 3-5 weeks.
    unknown_return_curve: tuple[float, ...] = (0.0, 0.15, 0.40, 0.65, 0.80, 0.88)
    # "Personal reasons" resolve less predictably than injuries; be more pessimistic.
    personal_curve: tuple[float, ...] = (0.25, 0.50, 0.70, 0.82, 0.88)
    # A 75%-chance doubt closes the gap to fit by this fraction each further gameweek.
    doubt_recovery: float = 0.5
    # Suspensions with no parseable end date: assume a standard three-match ban.
    default_suspension_gws: int = 3

    # --- web feed layer -----------------------------------------------------
    use_web: bool = True
    feeds: tuple[str, ...] = (
        "https://feeds.bbci.co.uk/sport/football/premier-league/rss.xml",
        "https://feeds.bbci.co.uk/sport/football/rss.xml",
        "https://www.theguardian.com/football/premierleague/rss",
        "https://feeds.skynews.com/feeds/rss/sports.xml",
    )
    feed_ttl: int = 3 * 3600
    web_max_age_days: float = 10.0     # ignore articles older than this
    web_effect_gws: int = 2            # a news story only speaks to the near term
    web_floor: float = 0.35            # web news alone can never push below this
    #: A "leaving" story about a player who joined his current club within this many
    #: days is describing the move that already happened. See `stale_departure`.
    stale_departure_days: int = 120
    web_penalty: dict = field(default_factory=lambda: {
        INJURED: 0.55,
        DOUBT: 0.80,
        SUSPENDED: 0.50,
        PERSONAL: 0.70,
        UNSETTLED: 0.55,               # dropped or pushing for a move: real minutes risk
        GONE: 0.75,                    # an exit rumour, not a confirmed departure
    })


@dataclass
class Availability:
    """What we believe about one player's availability, and why."""

    player_id: int
    category: str = FIT
    chance_next: float | None = None       # FPL's own 0-1 chance for the coming round
    return_date: dt.date | None = None
    return_gw: int | None = None
    #: True when a return date was parsed but lands past the last gameweek we know
    #: about -- i.e. the player is out for the whole of the loaded schedule.
    return_beyond_schedule: bool = False
    news: str = ""
    news_age_days: float | None = None
    web_category: str | None = None
    web_headline: str = ""
    web_age_days: float | None = None
    curve: dict[int, float] = field(default_factory=dict)

    @property
    def flagged(self) -> bool:
        return self.category != FIT or self.web_category is not None

    def summary(self) -> str:
        """One-line human explanation of the availability call."""
        if self.category == FIT and not self.web_category:
            return "available"
        bits = [self.category]
        if self.return_gw:
            bits.append(f"back GW{self.return_gw}")
        elif self.return_beyond_schedule and self.return_date:
            bits.append(f"out until {self.return_date:%d %b}")
        elif self.return_date:
            bits.append(f"back {self.return_date:%d %b}")
        elif self.chance_next is not None:
            bits.append(f"{self.chance_next:.0%} next")
        if self.web_category:
            bits.append(f"web:{self.web_category}")
        return ", ".join(bits)


# --------------------------------------------------------------------------- date helpers

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_day_month(text: str, today: dt.date) -> dt.date | None:
    """Turn FPL's '10 Oct' / '28 Nov' into a real date.

    FPL omits the year, so pick the reading that lands in a sensible window
    around today: a little in the past (news can lag) and up to a year ahead.
    """
    m = re.search(r"(\d{1,2})\s+([A-Za-z]{3,})", text)
    if not m:
        return None
    day = int(m.group(1))
    month = _MONTHS.get(m.group(2)[:3].lower())
    if month is None or not 1 <= day <= 31:
        return None
    for year in (today.year, today.year + 1, today.year - 1):
        try:
            cand = dt.date(year, month, day)
        except ValueError:
            continue
        if -45 <= (cand - today).days <= 400:
            return cand
    return None


def event_deadlines(events: list[dict]) -> dict[int, dt.date]:
    """gameweek id -> deadline date."""
    out: dict[int, dt.date] = {}
    for e in events:
        raw = e.get("deadline_time")
        if not raw:
            continue
        try:
            out[int(e["id"])] = dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00")).date()
        except ValueError:
            continue
    return out


#: A gameweek's matches are played over the days *after* its deadline, so a player
#: who becomes available a day or two into that window can still feature in it.
#: Two days covers a normal Saturday-Sunday round without reaching into Monday
#: night, where assuming an appearance would be wishful.
RETURN_GRACE_DAYS = 2


def gw_available_from(deadlines: dict[int, dt.date], when: dt.date,
                      grace_days: int = RETURN_GRACE_DAYS) -> int | None:
    """The first gameweek a player available from `when` can actually play in.

    Not simply "the next deadline after the date": a suspension that ends on the
    Saturday of a gameweek whose deadline was the Friday still lets the player
    feature in that gameweek, and treating him as banned for it costs a week of
    real points.
    """
    for gw in sorted(deadlines):
        deadline = deadlines[gw]
        if when <= deadline or (when - deadline).days <= grace_days:
            return gw
    return None


# --------------------------------------------------------------------------- FPL news parsing

_RE_RETURN = re.compile(r"expected back\s+(\d{1,2}\s+[A-Za-z]{3,})", re.I)
_RE_SUSPENDED_UNTIL = re.compile(r"suspended until\s+(\d{1,2}\s+[A-Za-z]{3,})", re.I)
_RE_CHANCE = re.compile(r"(\d{1,3})\s*%\s*chance", re.I)
_RE_JOINED = re.compile(r"has joined\s+", re.I)

_PERSONAL_WORDS = ("personal reason", "compassionate", "bereavement", "family reason")


def parse_fpl_news(element: dict, deadlines: dict[int, dt.date], today: dt.date) -> Availability:
    """Turn one bootstrap element's status/news fields into a structured signal."""
    news = (element.get("news") or "").strip()
    status = str(element.get("status") or "a").lower()
    low = news.lower()

    category = STATUS_CATEGORY.get(status, UNKNOWN)
    # FPL de-registers departed players; `can_select` is the cleanest flag for it.
    if element.get("can_select") is False:
        category = GONE
    if any(w in low for w in _PERSONAL_WORDS):
        category = PERSONAL
    elif _RE_JOINED.search(low):
        category = GONE
    elif "suspend" in low and category != GONE:
        category = SUSPENDED

    chance = element.get("chance_of_playing_next_round")
    chance_next = None
    if chance is not None:
        chance_next = max(0.0, min(1.0, float(chance) / 100.0))
    else:
        m = _RE_CHANCE.search(news)
        if m:
            chance_next = max(0.0, min(1.0, float(m.group(1)) / 100.0))

    return_date = None
    m = _RE_SUSPENDED_UNTIL.search(news) or _RE_RETURN.search(news)
    if m:
        return_date = parse_day_month(m.group(1), today)

    age = None
    added = element.get("news_added")
    if added:
        try:
            added_dt = dt.datetime.fromisoformat(str(added).replace("Z", "+00:00")).date()
            age = float((today - added_dt).days)
        except ValueError:
            age = None

    return_gw = gw_available_from(deadlines, return_date) if return_date else None
    return Availability(
        player_id=int(element["id"]),
        category=category,
        chance_next=chance_next,
        return_date=return_date,
        return_gw=return_gw,
        return_beyond_schedule=bool(return_date and return_gw is None),
        news=news,
        news_age_days=age,
    )


# --------------------------------------------------------------------------- availability curve


def _extend(curve: tuple[float, ...], k: int) -> float:
    """Read index k of a short curve, holding the final value beyond its end."""
    if not curve:
        return 1.0
    return curve[k] if k < len(curve) else curve[-1]


def availability_curve(sig: Availability, gws: list[int], cfg: NewsConfig) -> dict[int, float]:
    """Per-gameweek availability in [0, 1] implied by one player's news.

    This is the fix for the model's biggest availability bug: a flat `avail`
    applied to every gameweek meant an injury with a known October return date
    also wiped out the player's November fixtures.
    """
    out: dict[int, float] = {}

    # A parsed return date we could not place in the schedule is still information:
    # it is later than every gameweek we know about, so the player is out throughout.
    long_term_out = sig.return_beyond_schedule and sig.category in (INJURED, SUSPENDED, PERSONAL)

    for k, gw in enumerate(gws):
        if sig.category == GONE or long_term_out:
            out[gw] = 0.0
        elif sig.category == SUSPENDED:
            back = sig.return_gw
            if back is None:
                back = gws[0] + cfg.default_suspension_gws
            out[gw] = 0.0 if gw < back else 1.0
        elif sig.category == INJURED:
            if sig.return_gw is not None:
                out[gw] = 0.0 if gw < sig.return_gw else _extend(cfg.injury_ramp, gw - sig.return_gw)
            else:
                out[gw] = _extend(cfg.unknown_return_curve, k)
        elif sig.category == PERSONAL:
            if sig.return_gw is not None and gw >= sig.return_gw:
                out[gw] = _extend(cfg.injury_ramp, gw - sig.return_gw)
            else:
                out[gw] = _extend(cfg.personal_curve, k)
        elif sig.category == DOUBT:
            c = sig.chance_next if sig.chance_next is not None else 0.5
            # The doubt decays: by the time GW+3 arrives, a knock is usually gone.
            out[gw] = float(min(1.0, 1.0 - (1.0 - c) * (cfg.doubt_recovery ** k)))
        else:
            out[gw] = 1.0

    # FPL's own published chance is about the *coming* round specifically, and it
    # is better information than any curve of ours. Let it cap the first gameweek.
    if sig.chance_next is not None and gws:
        out[gws[0]] = min(out[gws[0]], sig.chance_next)

    return {gw: float(max(0.0, min(1.0, v))) for gw, v in out.items()}


# --------------------------------------------------------------------------- web feeds

_STRIP = re.compile(r"[^a-z ]+")


def normalise(name: str) -> str:
    """Lowercase, strip accents and punctuation, so 'Rodriguez' matches either spelling."""
    decomposed = unicodedata.normalize("NFKD", str(name))
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _STRIP.sub(" ", plain.lower()).strip()


#: Surnames that are also ordinary English words or extremely common. Matching on
#: these alone produces nonsense hits, which silently bench a good player.
AMBIGUOUS_SURNAMES = {
    "david", "wilson", "anthony", "rogers", "young", "may", "cash", "king", "moore",
    "walker", "wood", "hill", "reed", "bell", "clark", "cook", "ward", "webb", "gray",
    "long", "white", "black", "brown", "green", "day", "west", "banks", "brooks",
    "rice", "sterling", "phillips", "james", "lewis", "martin", "richards", "murphy",
    "smith", "taylor", "jones", "wright", "roberts", "scott", "turner", "cooper",
}

_WEB_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (SUSPENDED, ("suspend", "red card", "banned", " ban ", "three-match", "match ban")),
    (INJURED, (
        "injur", "hamstring", "ankle", "knee ", "calf", "groin", "thigh", "achilles",
        "acl", "fracture", "broken", "surgery", "operation", "sidelined", "ruled out",
        "out for", "months out", "weeks out", "torn", "strain",
    )),
    (PERSONAL, ("personal reasons", "compassionate", "bereavement", "family reasons")),
    (UNSETTLED, (
        "want out", "wants out", "wants to leave", "wants away", "wanting to play",
        "verge of leaving", "pushing for a move", "asked to leave", "hand in a transfer",
        "transfer request", "frozen out", "unsettled", "agitating", "refused to play",
        "left out", "omitted", "dropped from", "not in the squad", "absence at",
        "s absence", "training alone", "exiled", "wantaway",
    )),
    (GONE, ("completes move", "joins ", "signs for", "sealed a move", "on loan to",
            "transfer to", "goodbye to", "farewell", "departs", "left the club",
            "moves to", "sold to")),
    (DOUBT, ("doubt", "scan", "fitness test", "race to be fit", "knock", "assessed", "rested")),
)

#: How much text either side of a player's name counts as being "about" him.
#: Match reports name a dozen players; only the clause around the name is evidence.
PROXIMITY_BEFORE = 70
PROXIMITY_AFTER = 110
#: A keyword further than this from the name is about somebody else in the article.
PROXIMITY_MAX = 85

#: Phrases that mean the *opposite* -- a player coming back. These must not be read
#: as fresh injuries just because the sentence contains the word "injury".
_RETURN_WORDS = ("returns", "return to", "back in training", "fit again", "available again",
                 "back from injury", "set to return", "recovered")


def _nearest(text: str, words: tuple[str, ...], at: int) -> int | None:
    """Distance from `at` to the closest occurrence of any of `words`, or None."""
    best = None
    for w in words:
        start = 0
        while True:
            i = text.find(w, start)
            if i < 0:
                break
            d = abs(i - at)
            if best is None or d < best:
                best = d
            start = i + 1
    return best


def classify_near(text: str, at: int, max_distance: int = PROXIMITY_MAX) -> str | None:
    """Category of the keyword *closest* to position `at` in `text`.

    Taking the nearest keyword rather than the first matching pattern is what
    separates "Brighton said goodbye to Welbeck ... was without Baleba because of
    injuries" into a departure for one player and an injury for the other. Order
    in `_WEB_PATTERNS` should not decide which of them gets which.
    """
    scored = []
    for category, words in _WEB_PATTERNS:
        d = _nearest(text, words, at)
        if d is not None and d <= max_distance:
            scored.append((d, category))
    if not scored:
        return None

    # A "he's back in training" phrase closer than any bad news wins outright.
    good = _nearest(text, _RETURN_WORDS, at)
    best_distance, best_category = min(scored)
    if good is not None and good <= best_distance:
        return None
    return best_category


def classify_headline(text: str) -> str | None:
    """Category for a short standalone headline, where everything is 'nearby'."""
    return classify_near(normalise(text), 0, max_distance=len(text) + 1)


def _parse_pubdate(raw: str) -> dt.datetime | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            parsed = dt.datetime.strptime(raw, fmt)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
    return None


def fetch_feed(url: str, ttl: int) -> list[dict]:
    """Fetch and parse one RSS feed into {title, description, published} items.

    Cached on disk like every other request. A feed that is down, slow or has
    changed shape returns nothing rather than breaking the run.
    """
    key = "feed_" + re.sub(r"[^a-zA-Z0-9]+", "_", url)[:80]
    cached = api.cache_read(key, ttl)
    if cached is not None:
        return cached

    try:
        resp = requests.get(url, headers={"User-Agent": api.UA}, timeout=20)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except (requests.RequestException, ET.ParseError):
        return []

    items = []
    for it in root.findall(".//item"):
        title = (it.findtext("title") or "").strip()
        desc = re.sub(r"<[^>]+>", " ", it.findtext("description") or "").strip()
        pub = _parse_pubdate(it.findtext("pubDate") or "")
        if not title:
            continue
        items.append({
            "title": title,
            "description": desc,
            "published": pub.isoformat() if pub else None,
        })
    api.cache_write(key, items)
    return items


def fetch_web_news(cfg: NewsConfig, verbose: bool = False) -> list[dict]:
    """All configured feeds, fetched in parallel."""
    if not cfg.use_web or not cfg.feeds:
        return []
    items: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(4, len(cfg.feeds))) as pool:
        for got in pool.map(lambda u: fetch_feed(u, cfg.feed_ttl), cfg.feeds):
            items.extend(got)
    if verbose:
        print(f"  news: {len(items)} headlines from {len(cfg.feeds)} feed(s)")
    return items


def _player_keys(element: dict) -> tuple[str, str]:
    """(full name, surname) normalised, for matching against headlines."""
    full = normalise(f"{element.get('first_name', '')} {element.get('second_name', '')}")
    surname = normalise(element.get("second_name") or element.get("web_name") or "")
    surname = surname.split()[-1] if surname else ""
    return full, surname


def match_web_news(
    elements: list[dict],
    items: list[dict],
    teams: dict[int, dict],
    cfg: NewsConfig,
    today: dt.date,
) -> dict[int, dict]:
    """player id -> the most relevant recent headline about them.

    Precision matters far more than recall here: a false hit silently benches a
    good player. So a surname-only match must clear extra bars (long enough, not
    an everyday English word, unique in the league, and corroborated by the
    club's name appearing in the same article).
    """
    if not items:
        return {}

    surname_counts: dict[str, int] = {}
    for e in elements:
        _, sur = _player_keys(e)
        if sur:
            surname_counts[sur] = surname_counts.get(sur, 0) + 1

    # Pre-normalise the corpus once.
    corpus = []
    for it in items:
        pub = None
        if it.get("published"):
            try:
                pub = dt.datetime.fromisoformat(it["published"]).date()
            except ValueError:
                pub = None
        age = float((today - pub).days) if pub else None
        if age is not None and age > cfg.web_max_age_days:
            continue
        text = f"{it['title']} {it.get('description', '')}"
        corpus.append({"text": text, "norm": normalise(text), "age": age, "title": it["title"]})

    if not corpus:
        return {}

    out: dict[int, dict] = {}
    for e in elements:
        full, sur = _player_keys(e)
        team = teams.get(e.get("team"), {})
        team_name = normalise(team.get("name", "") or "")
        team_short = normalise(team.get("short_name", "") or "")
        unique_surname = bool(sur) and surname_counts.get(sur, 0) == 1 and sur not in AMBIGUOUS_SURNAMES

        best = None
        for art in corpus:
            padded = f" {art['norm']} "
            at = -1
            if full and len(full) > 6 and full in art["norm"]:
                at = art["norm"].index(full)
            elif unique_surname and len(sur) >= 5 and f" {sur} " in padded:
                # Surname alone needs the club mentioned in the same article.
                corroborated = bool(team_name and team_name in art["norm"]) or bool(
                    team_short and len(team_short) >= 3 and f" {team_short} " in padded
                )
                if corroborated:
                    at = art["norm"].index(sur)
            if at < 0:
                continue
            # Classify only the text around the player's name, by the keyword
            # nearest to it. A match report that mentions six players and one
            # injury must not put all six on the treatment table.
            lo = max(0, at - PROXIMITY_BEFORE)
            window = art["norm"][lo: at + PROXIMITY_AFTER]
            category = classify_near(window, at - lo)
            if category is None:
                continue
            age = art["age"] if art["age"] is not None else cfg.web_max_age_days
            if best is None or age < best["age_days"]:
                best = {"category": category, "headline": art["title"], "age_days": age}
        if best:
            out[int(e["id"])] = best
    return out


def stale_departure(element: dict, category: str, cfg: NewsConfig, today: dt.date) -> bool:
    """Is this "he's leaving" story just describing the move he has already made?

    "Brighton said goodbye to Danny Welbeck" is true, and useless: he is at
    Chelsea now, registered and selectable, and FPL knows it. Reading it as an
    absence penalises a player for the same transfer that put him in the squad.

    An exit story about a player who has been at his club for years is different
    -- that is a live rumour and worth heeding -- so only recent arrivals are
    suppressed.
    """
    if category != GONE:
        return False
    raw = element.get("team_join_date")
    if not raw:
        return False
    try:
        joined = dt.date.fromisoformat(str(raw)[:10])
    except ValueError:
        return False
    return 0 <= (today - joined).days <= cfg.stale_departure_days


def apply_web_signal(curve: dict[int, float], sig: Availability, cfg: NewsConfig,
                     gws: list[int]) -> dict[int, float]:
    """Fold an unconfirmed web report into an availability curve.

    Bounded on purpose. Web news can shade a player down but never rule him out,
    and only for the next `web_effect_gws` gameweeks -- after that, if it were
    real, FPL's own flags would have caught up.
    """
    if not sig.web_category:
        return curve
    penalty = cfg.web_penalty.get(sig.web_category, 0.85)
    out = dict(curve)
    for gw in gws[: max(0, cfg.web_effect_gws)]:
        base = out.get(gw, 1.0)
        out[gw] = min(base, max(base * penalty, cfg.web_floor))
    return out


# --------------------------------------------------------------------------- top level


def build_availability(
    bootstrap: dict,
    gws: list[int],
    cfg: NewsConfig | None = None,
    today: dt.date | None = None,
    verbose: bool = False,
) -> dict[int, Availability]:
    """Everything above, wired together: player id -> Availability with a filled curve."""
    cfg = cfg or NewsConfig()
    today = today or dt.date.today()
    deadlines = event_deadlines(bootstrap.get("events", []))
    teams = {t["id"]: t for t in bootstrap.get("teams", [])}
    elements = bootstrap.get("elements", [])

    web_hits: dict[int, dict] = {}
    if cfg.use_web:
        try:
            items = fetch_web_news(cfg, verbose=verbose)
            web_hits = match_web_news(elements, items, teams, cfg, today)
            if verbose and web_hits:
                print(f"  news: {len(web_hits)} player(s) mentioned in recent headlines")
        except Exception as exc:  # a news feed must never take the whole run down
            if verbose:
                print(f"  ! web news unavailable ({type(exc).__name__}), continuing on FPL flags")

    out: dict[int, Availability] = {}
    for e in elements:
        sig = parse_fpl_news(e, deadlines, today)
        hit = web_hits.get(sig.player_id)
        if hit and stale_departure(e, hit["category"], cfg, today):
            hit = None
        if hit:
            sig.web_category = hit["category"]
            sig.web_headline = hit["headline"]
            sig.web_age_days = hit["age_days"]
        curve = availability_curve(sig, gws, cfg)
        sig.curve = apply_web_signal(curve, sig, cfg, gws)
        out[sig.player_id] = sig
    return out
