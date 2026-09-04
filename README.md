# fpl-ai

A decision engine for Fantasy Premier League. It pulls live FPL data, projects
every player's points for the gameweeks ahead, and solves for the best legal
squad, the best transfers, and the best XI + captain.

Two modes, matching the two things you do all season:

- `squad` — build a 15-man squad from nothing (season start, or a wildcard)
- `weekly` — after each gameweek, decide transfers, lineup and armband

Plus two read-only reports for the questions that decide those calls:

- `news` — who is injured, suspended, away, or in this morning's headlines
- `signings` — who has just changed club, and whether they will actually play

The optimiser is an exact integer programme, not a greedy picker: it considers
budget, the 3-per-club cap, formation legality, bench value, points hits and
future fixtures **at the same time**, which is exactly where human intuition
tends to leak points.

---

## Setup

### Option A: Local Python

Requires Python 3.10+. No API key or login required for core features.

```bash
cd fpl-ai
python3 -m venv .venv && source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Your squad file

`plan_week.py` reads your squad from `my_squad.json`, which is gitignored so
your team stays local. Start from the template:

```bash
cp my_squad.example.json my_squad.json
```

Then edit it: names must match FPL web names, clubs are the 3-letter FPL short
codes, and `state` holds your bank, free transfers and planning gameweek.

### Option B: Docker

Build the Docker image:

```bash
docker build -t fpl-ai .
```

Or use **Docker Compose** directly (handles cache volume automatically):

```bash
docker compose build
```

---

## First run — build your Gameweek 1 squad

The first run downloads a history file for every player (~700 requests, a
couple of minutes). It's cached for 24 hours, so every later run is instant.
Add `--shallow` to skip it and rely on FPL's own expected points instead —
faster, noticeably blunter.

### Local Python
```bash
python -m fplai squad
```

### Docker
```bash
# Using a persistent volume for cache across runs
docker run --rm -v fpl-cache:/app/.cache fpl-ai squad

# Quick shallow run
docker run --rm fpl-ai squad --shallow

# Using Docker Compose
docker compose run --rm fpl-ai squad
```

### Useful variations

```bash
# Force a pick, look further ahead
python -m fplai squad --include Haaland --horizon 6

# Ban a player, specify budget
python -m fplai squad --exclude Isak --budget 100

# Shortlist, not a full squad
python -m fplai players --pos MID --max-price 8.0 --top 20

# Best points per million
python -m fplai players --sort value
```

*(Any of the above can be run in Docker by replacing `python -m fplai` with `docker run --rm -v fpl-cache:/app/.cache fpl-ai` or `docker compose run --rm fpl-ai`)*

---

## Every week after that

Find your team id: log in to the FPL site, open "Points", and it's the number
in the URL (`.../entry/1234567/event/3`).

```bash
python -m fplai weekly --entry 1234567 --free-transfers 2 --bank 0.5
```

That reads last week's squad from the public API. Prices drift, so selling
prices won't be exact. For exact numbers (selling price, bank, free transfers
all read live), grab your cookie from a logged-in browser — DevTools →
Network → any `api` request → Request Headers → `cookie` — save it to
`cookie.txt` and run:

### Local Python
```bash
python -m fplai weekly --entry 1234567 --cookie cookie.txt
```

### Docker (Mounting local files)
```bash
# Mount current directory to pass cookie.txt / overrides or export JSON
docker run --rm -v fpl-cache:/app/.cache -v "${PWD}:/workspace" -w /workspace fpl-ai weekly --entry 1234567 --cookie cookie.txt
```

Wildcard week:

```bash
python -m fplai weekly --entry 1234567 --wildcard
```

Nothing is ever submitted on your behalf. It prints a plan; you make the moves.

---

## How the projection works

For each player, each gameweek, points are broken down into probabilistic scoring components:

```
points = appearance + goals(xG) + assists(xA) + clean_sheet(Poisson) + concede_penalty + saves + bonus - cards
```

- **Expected minutes (`xMins`) & Starts (`p_start`)** — derived from recent starts (last 4 matches), in-season form, historical playing time, and official FPL availability flags.
- **In-season EWMA & Empirical Bayes** — Exponentially Weighted Moving Average (EWMA) tracks current-season returns with decay, blended with past-season totals and price-based positional priors.
- **Poisson Match Engine** — converts home/away offensive and defensive strengths into expected goals scored ($\lambda$) and conceded ($\mu$), deriving exact clean sheet probabilities $P(CS) = e^{-\mu}$ and multi-goal concession penalties.
- **Positional Scoring Matrices** — goals (DEF: 6, MID: 5, FWD: 4), clean sheets (DEF/GKP: 4, MID: 1), and save point distributions are explicitly mapped per FPL rulebook.
- **Multi-Period Transfer Trajectory** — optimizes moves across the entire horizon with dynamic free transfer accumulation (up to 5 stacked FTs under 2026/27 rules) and hit penalties.
- For the coming gameweek only, FPL's published `ep_next` can be blended in via `--ep-blend`.

Future gameweeks are discounted by `--decay` (0.86 per week by default), so the coming week matters most but you don't walk into a fixture wall.

---

## Availability, transfers and team news

Three gates sit in front of every one of those scoring components, and all three
are evaluated **per gameweek** rather than once for the whole horizon.

### 1. Will he be fit?

FPL's own flags are the authority, and their wording is regular enough to parse
into a return date:

| FPL says | Model reads |
|---|---|
| `Ankle injury - Expected back 10 Oct` | out until the gameweek whose matches start 10 Oct, then a 55% / 80% / 92% ramp back to full |
| `Hamstring injury - 75% chance of playing` | 75% this week, the doubt halving each week after |
| `Suspended until 19 Sep` | hard zero, then straight back to 100% |
| `Personal reasons - Unknown return date` | a slower, more pessimistic recovery curve than an injury |
| `Has joined Getafe permanently` | zero, permanently |
| `Groin injury - Unknown return date` | zero now, recovering over roughly four weeks |

This is the single biggest change to the projections. Previously one injury flag
zeroed a player out for *every* gameweek in the horizon, so the model would never
buy someone due back in a fortnight — and it applied a "75% chance of playing",
which is about the coming round only, to every future gameweek as well.

The flags do not cover everything. A player who is fit, registered and refusing
to play reads as perfectly available right up until he doesn't play — FPL has no
status letter for "left out of the squad" or "pushing for a move". So the feeds
are also scanned for squad omissions and transfer agitation, filed as
`unsettled`:

```
~ Watkins   AVL  own 9.9%   FPL status 'a', no news   start 97% -> 55%
    "Watkins' absence at Brighton a question for him - Emery"
~ Elliott   LIV  own 0.1%   FPL status 'a', no news   start 0%
    "Does Elliott have a Liverpool future after latest omission?"
```

On top of that, public news feeds (BBC Sport, the Guardian, Sky) are read for
stories that break before FPL updates its flags. That layer is deliberately weak:
it can only ever shade a player **down**, never clear one, never below a floor,
and only for the next two gameweeks. Classification uses the keyword nearest the
player's name, so a match report naming six players and one injury doesn't put
all six on the treatment table.

```bash
python -m fplai news                    # everyone flagged, most-owned first
python -m fplai news --entry 1234567    # just your squad
python -m fplai --no-news squad         # FPL flags only, no feeds
```

### 2. Has he just moved?

`team_join_date` gives the exact date a player joined his current club, so both
windows are visible: the summer one before a season-start pick, and January
before the mid-season deadline. A recent arrival gets a settling-in discount on
his minutes — steeper for a January signing, who arrives without a pre-season —
and that discount decays away as he racks up appearances for the club.

For players with no Premier League record at all, per-90 goal and assist rates
come from Understat's top-five-league data, weighted toward npxG and xA rather
than actual goals, cut by a league-strength factor, then handed to the
empirical-Bayes step as deliberately *weak* evidence. A price tag is a poor guess
at what a striker from Serie A will do; his xG is a better one.

```bash
python -m fplai signings                     # every recent arrival
python -m fplai signings --window january    # mid-season deadline planning
python -m fplai --no-external squad          # price priors only
```

### 3. Will he get in the team?

Nothing previously stopped the model buying a backup goalkeeper: expected minutes
came from each player's own history in isolation, so two keepers at one club could
both look like starters, and a big-money arrival never displaced the incumbent he
was bought to replace.

Start probability is a prior — what his record before this season suggests — with
this season's team sheets shrunk into it by sample size. That matters most in the
opening weeks, when there is one match to go on and "was he in the team?" is the
most informative fact available. Appearances are graded rather than binary: a
start counts 1.0, a substitute appearance 0.35, being left out of the squad 0.0,
so a starter who was substituted early is not confused with one who has been
frozen out. A new signing's pre-move record is pulled toward "unknown" first —
a keeper who played every week on loan elsewhere is not thereby the number one at
the club that just bought him.

Within each (club, position) group, start probabilities are now sharpened and
renormalised toward a realistic team shape, which makes competition zero-sum. An
unavailable player drops out of the contest, correctly promoting the deputy behind
an injured first choice — and demoting him again when the starter returns.

```
MCI: Donnarumma 100% -> 64%,   Rulli 93% -> 59%       # a real two-keeper situation
ARS: Raya       100% -> 97%,   Arrizabalaga 3% -> 2%  # no contest, barely touched
```

Availability is a hard ceiling on the result: renormalising scales players up to
fill a club's empty places, which is right when a rival is out of form and wrong
when the player himself is doubtful. Villa's other forwards being injured must not
promote a doubtful striker back to a near-certainty.

`--rotation-weight 0` turns this off and trusts each player's own history; `1.0`
enforces the team shape exactly. The default of 0.75 blends the two, so a club's
start probabilities can still sum to slightly more than the places available where
the evidence genuinely supports it.

---

## Watching your squad on a schedule

Team news breaks on a Friday afternoon, not when you happen to open a terminal.
`watch_availability.py` is built to run unattended and to be worth reading when
it does:

```bash
python watch_availability.py              # what changed since the last run
python watch_availability.py --flagged    # plus everyone still carrying a flag
python watch_availability.py --full       # the whole fifteen, changed or not
python watch_availability.py --json report.json
python watch_availability.py --no-web     # FPL's own flags only
```

Two things separate it from `python -m fplai news`:

- **It is cheap.** It touches the bootstrap endpoint and the RSS feeds and
  nothing else — no fixtures, no ~626 player histories, no optimiser. A daily
  run on a quiet day costs one small request.
- **It has a memory.** Each run records what it believed about each of your
  players in `.availability_state.json` (gitignored), and the next run reports
  the *diff*: `fit -> doubt`, `chance 75% -> 25%`, `return GW? -> GW6`, or best
  of all `FPL news cleared`. A daily report that reprints the same fifteen rows
  is a report you stop reading.

```
  CHANGED SINCE THE LAST RUN (2):

    PLAYER          CLUB POS AVAILABILITY                 GW3  WHY IT IS HERE
  ------------------------------------------------------------------------------
  ! Mateta          CRY  FWD injured, back GW6             0%  FPL news changed
      fpl: Hamstring injury - Expected back 11 Oct
  ! Caicedo         CHE  MID doubt, 75% next              75%  fit -> doubt
      fpl: Unspecified injury - 75% chance of playing
```

The `GW3` column is the availability multiplier the model would apply to that
player's minutes in the coming gameweek: `100%` plays, `0%` cannot. The `WHERE`
column comes from the `fielding` block of your squad file — `XI`, `B1`/`B2`/`B3`
in autosub order, `BGK` for the reserve keeper, with `C` and `V` for the
armbands — and the XI sorts ahead of the bench, because a doubt over your
captain is a different problem from the same doubt over your third bench
outfielder.

Exit codes are the point of the script, because they let a scheduler stay quiet:

| Exit | Meaning |
|---|---|
| 0 | something changed — worth reporting |
| 1 | nothing new since the last run — a caller should say nothing |
| 2 | could not run: no `my_squad.json`, or it is still the template |

Pair it with `deadline_gate.py`, which exits 0 only when the next deadline is
inside 30 hours, and you get a cheap daily scan plus a full check on the day
that matters:

```bash
if python deadline_gate.py; then
  python watch_availability.py --flagged   # deadline is close: the full picture
else
  python watch_availability.py             # quiet day: only what changed
fi
```

### As Claude Code scheduled tasks

Two tasks, because one time of day cannot cover both kinds of deadline.

`fpl-availability-watch` runs daily at 09:00 and then does what a script cannot:
for every player the watcher flags, it
web-searches the latest team news, because FPL's flags and the RSS feeds both
lag the manager's press conference. On a deadline run it sweeps all fifteen
players, XI and bench alike, searching **by club** rather than by player — team
news is published per club, so one search covers everyone you own there and also
catches what a player-name search misses, like a squad omission or a rest ahead
of a midweek European game. It reports a verdict per player — likely starts,
rotation risk, bench risk, genuine doubt, will not play — and notifies only when
a squad player's availability actually moved and a deadline is close. On a quiet
day it searches only what the watcher says has changed, so the daily run stays
cheap.

`fpl-deadline-final-check` runs daily at 16:00 but gates itself on
`deadline_gate.py --hours 5`, so it speaks only on days when the deadline is
that same evening — roughly once a gameweek, and otherwise it prints one line
and stops. It exists because most managers face the media around midday: for a
Friday-evening deadline the 09:00 run happens *before* the press conferences for
most of a typical squad, so it structurally cannot see them. This one sweeps the
clubs again afterwards, three hours before you have to decide, and always
sweeps — a watcher reporting "no change" at 16:00 has said nothing about what
was said at noon. A Saturday-lunchtime deadline needs none of this: the 09:00
run that morning is already after Friday's pressers.

Both share `.availability_state.json`, so the afternoon task's "what changed"
means *changed since this morning* — which is exactly what a final check wants.

Both are read-only. Nothing is ever submitted to FPL, and transfer plans stay
with `plan_week.py`, where you can see the alternatives side by side.

---

## Tuning it

The model is meant to be argued with. Every knob is a flag:

| Flag | Effect |
|---|---|
| `--horizon` | How far ahead to plan. 3 = reactive, 8 = fixture-swing planning |
| `--decay` | Lower = more short-termist |
| `--bench-weight` | 0.12 default. Raise it if you plan a Bench Boost, drop to ~0.05 for a pure "fodder bench" build |
| `--ep-blend` | How much to trust FPL's own numbers |
| `--max-transfers` | Cap on moves; the optimiser already prices `-4` hits itself |
| `--rotation-weight` | 0 = trust each player's own history, 1 = enforce a realistic number of starters per club. Default 0.75 |
| `ModelConfig.start_prior_matches` | Prior strength, in matches, for "is he a starter". Lower reacts harder to one benching. Default 1.5 |
| `ModelConfig.new_signing_start_pull` | How much of a new signing's pre-move starting record still applies. Default 0.6 |
| `--new-signing-days` | How recently a player must have joined to count as a new signing. Default 90 |
| `--no-news` | Skip the public news feeds. FPL's own injury flags are still parsed |
| `--news-max-age` | Ignore headlines older than this many days. Default 10 |
| `--no-external` | Skip foreign-league stats for players with no Premier League record |
| `--external-season` | Season start year to read abroad. Default: the last completed one |

If you trust an outside source more than this model for a given player, hand it
your own numbers:

```csv
player,points,gw
Haaland,9.4,1
Saka,6.1,
```

```bash
python -m fplai squad --overrides my_projections.csv
```

Rows without a `gw` apply to every gameweek in the horizon.

---

## Testing changes

Fabricates a fake league offline and asserts every FPL rule holds: 15 players,
2/5/5/3, max 3 per club, budget, legal formations, a reserve keeper benched
last, and that a smaller budget never beats a larger one. Run it after any edit
to the model.

`test_availability.py` covers the newer layers with hand-written fixtures that
mirror the exact shapes the live APIs return: every FPL news string above, the
per-gameweek curves they imply, the nearest-keyword headline classifier, transfer
window detection, settling-in decay, and the depth chart. No network needed.

`test_watch.py` pins the scheduled watcher: club-local squad resolution (two
players called Sarr must not collapse into one), unresolved names surfacing
rather than being silently dropped, and the change detection — a flag that is
simply a day older is not news, but `fit -> doubt` and `injured -> fit` both
are.

### Local Python
```bash
python tests/test_synthetic.py
python tests/test_availability.py
python tests/test_watch.py
```

### Docker
```bash
docker run --rm --entrypoint python fpl-ai tests/test_synthetic.py
docker run --rm --entrypoint python fpl-ai tests/test_availability.py
docker run --rm --entrypoint python fpl-ai tests/test_watch.py
```

---

## Dates that matter this season (2026/27)

- **Gameweek 1 deadline: Friday 21 August, 18:30 BST.** Arsenal v Coventry opens it.
- Up to **5 free transfers** can be stacked.
- Chips are split into two halves — the first set (Wildcard, Free Hit, Triple
  Captain, Bench Boost) expires at the Gameweek 19 deadline and cannot be
  carried over.
- Gameweeks 5 and 6 are three weeks apart because the September and October
  international breaks are merged this year. Worth pointing `--horizon` past
  that gap when you plan around it.

---

## Known limits

- No price-change modelling (team value growth is ignored).
- Premier League xG is not used yet. FPL now publishes `expected_goals_per_90`
  and `expected_assists_per_90`, which would sharpen the in-season rates the same
  way Understat's numbers already sharpen new signings'. Foreign-league xG *is*
  used, for players with no PL record.
- Effective ownership is not modelled, so it plays to maximise points, not to
  beat a specific mini-league.
- The news feeds are English-language general football RSS. They catch the big
  stories, not every press-conference line, and they are bounded so they can only
  nudge a projection. FPL's own flags remain the authority on who can play.
- Return dates are the club's own estimate. One that slips is a stale input until
  FPL updates it, which is why the `news` report prints the raw string to check
  against.
- Rotation is modelled as competition for a place, not as a manager's rotation
  policy: it does not know that a particular club rests players in cup weeks.
- An `unsettled` player reverts to full strength after `web_effect_gws`. That is
  deliberate — an unconfirmed rumour should not zero a player out for a whole
  horizon — but it means a transfer saga that *does* end in a sale is priced too
  optimistically from the third gameweek on. Use `--overrides` when you are
  confident a move is happening.
