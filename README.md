# T20 match simulator

The fitted model estimates delivery probabilities by phase, wickets lost,
innings, and team. From this project directory, run the complete download,
ingestion, validation, fit, save, and held-out evaluation pipeline with:

```powershell
.\cricenv\Scripts\python.exe .\run_pipeline.py
```

Raw data is downloaded only when it is missing. Pass `--refresh-data` to
download and extract the latest Cricsheet archive. Use `--simulations` and
`--seed` to control the evaluation run:

```powershell
.\cricenv\Scripts\python.exe .\run_pipeline.py --refresh-data --simulations 1000 --seed 42
```

The pipeline writes processed tables to `data/processed/` and the fitted model
to `results/models/node_probs.npz`. To run only the existing tests or
evaluation:

```powershell
.\cricenv\Scripts\python.exe -m pytest .\tests -q
.\cricenv\Scripts\python.exe -m src.evaluation --simulations 1000 --seed 42
```

Evaluation fits only on matches before `2025-01-01` and tests on matches on or
after that date.

## Simulate a fixture

After running the pipeline, create a fixture JSON with each ordered XI:

```json
{
  "team1": "Australia",
  "team2": "India",
  "venue": "Melbourne Cricket Ground",
  "match_date": "2026-10-20",
  "batting_first": "team1",
  "lineups": {
    "team1": {
      "batting_order": ["Player 1", "Player 2", "Player 3", "Player 4", "Player 5", "Player 6", "Player 7", "Player 8", "Player 9", "Player 10", "Player 11"]
    },
    "team2": {
      "batting_order": ["Player 1", "Player 2", "Player 3", "Player 4", "Player 5", "Player 6", "Player 7", "Player 8", "Player 9", "Player 10", "Player 11"]
    }
  }
}
```

Replace the example names with actual player names and save the fixture
outside the ignored `data/` directory. Then run:

```powershell
.\cricenv\Scripts\python.exe .\run_simulator.py --fixture .\fixtures\match.json --seed 42
```

To step through a match delivery by delivery, enter the observed outcome after
each model forecast:

```powershell
.\cricenv\Scripts\python.exe .\run_simulator.py `
  --fixture .\fixtures\match.json --interactive --seed 42
```

In interactive mode, choose which team bats first, then set each batting order
by entering the XI numbers in the order you want (or press Enter to keep the
fixture order). At the start of each over, choose a bowler from the numbered
eligible-bowler list. Names are displayed for selection; you do not need to
type them. The simulator prevents consecutive overs and enforces the four-over
limit. Bowling forecasts use your choices for completed and current overs and
the model's projected schedule for future overs.

Enter `0`–`6` for legal-delivery team runs, `W[:<runs>]` for a wicket,
`WD:<total runs>[:<completed runs>]` or
`NB:<total runs>[:<completed runs>]` for an extra delivery, or `q` to stop.
For wides/no-balls, the optional completed-runs value lets strike movement
reflect runs physically completed rather than just the total credited to the
team. After every entry, the score, wickets, legal balls, strike, and batting
state update. The next prompt shows Monte Carlo next-ball outcomes and a
forecast of the innings total (or chase win probability). Use
`--forecast-simulations` to control the simulations per prompt; lower values
are faster, while higher values reduce Monte Carlo noise.

This is sequential conditioning on the observed match state, not online model
retraining: the fitted probabilities remain fixed, while subsequent
probabilities and forecasts use the updated state and current batter/bowler.

The simulator applies team, venue, and player strengths. By default, all XI
members are bowling candidates; an optional `bowlers` list can restrict
selection to 5–11 named players from that XI. Bowlers with prior history are
allocated overs in proportion to their average overs per recent bowling
appearance, subject to the four-over limit. If fewer than five candidates have
bowling history, the remaining places are filled from the candidate list. Their
historical over-number frequencies guide the exact schedule, and a bowler is
not scheduled for consecutive overs. Batter strike rotates on odd runs and over
ends.

Player and team form guides use their latest ten completed appearances strictly
before `match_date` (or `--as-of`, if supplied). Batting averages (batter runs
per dismissal) and bowling averages (bowler-conceded runs per credited wicket)
are estimated separately for each innings, phase, and wickets-lost state.
Sparse averages shrink toward the corresponding pre-fixture league state
average and provide a modest adjustment to simulated run and wicket rates.
The debut-player prior is the average of individual players' first-match
statistics in the training split, with unobserved state rates shrunk toward the
league baseline. No match on or after the fixture date is used to calculate
form or bowling allocations. The fitted model itself only sees matches before
`2025-01-01`; dates earlier than that cutoff are not valid backtest fixture
dates for this model.

For players with no previous history, the simulator uses the training-split
debut-player average. Player names are resolved against player names present
in deliveries strictly before the fixture date; an ambiguous historical name
produces an error rather than selecting an arbitrary player. Names that have
not appeared in that history are treated as debut players.
To resolve an ambiguous name, add a `player_ids` object to that team's lineup,
mapping the displayed name to its historical player ID. The simulator verifies
that the supplied ID was associated with that name before the fixture date.

For example, `fixtures/england_vs_india_2025-01-22.json` uses the actual
England and India XIs from their January 22, 2025 T20I at Eden Gardens:

```powershell
.\cricenv\Scripts\python.exe .\run_simulator.py `
  --fixture .\fixtures\england_vs_india_2025-01-22.json --seed 42
```

## Simulate a match or continue from a game state

```python
from src.model import load
from src.simulator import GameState, simulate_ball, simulate_match

fitted = load()
match = simulate_match(fitted, rng=42)
print(match.first_innings.final_state.runs, match.second_innings.final_state.runs)

# State is the score immediately before the next ball.
state = GameState(innings_number=2, legal_balls=42, wickets=3, runs=71, target_runs=160)
delivery = simulate_ball(fitted, state, rng=42)
print(delivery.outcome, delivery.state_after)
```

Use one `numpy.random.Generator` for reproducible sequences across multiple
calls to `simulate_ball`. `target_runs` is the score required to win (one more
than the first-innings score).

## Team-level strengths

When team columns are present in the training deliveries, the fitted model also
learns batting-team and bowling-team probabilities. Team rates are smoothed
toward the overall rates for each match state, so teams with fewer observations
do not get extreme probabilities. Wicket, legality, and wide probabilities use
the bowling team's profile; run-scoring probabilities use the batting team's
profile. Teams not present in the fitted model raise a `ValueError`.

Pass the teams batting and bowling in the first innings to use those profiles.
The teams switch roles for the chase:

```python
match = simulate_match(
    fitted,
    batting_first_team="Australia",
    bowling_first_team="India",
    rng=42,
)
```

Omitting both team names preserves the baseline simulation.

## Venue effects

The fit also learns state-conditional delivery probabilities by venue, using
the match venue joined onto historical deliveries. Venue estimates are
smoothed toward league-level rates and combine with the team adjustment.
Venue names are matched case-insensitively after trimming whitespace; a venue
not present in the training data raises a clear error instead of silently
pretending it has a learned effect. The held-out evaluation reports venue
coverage and compares venue-only probabilities against the league baseline
for known grounds.

Use the exact ground name as listed in the match data:

```powershell
.\cricenv\Scripts\python.exe .\run_simulator.py `
  --fixture .\fixtures\match.json --seed 42
```

Ground names in source data are not fully standardized (for example, names
may include city or ground-number suffixes), so similar-looking names may be
treated as different venues.

## v1 outcome simplifications

The decision tree maps legal deliveries to six, four, one, two, three, dot, and
wicket outcomes. Wides and no-balls have separate, state-conditioned
distributions over total delivery runs, including additional wide runs and
batter runs scored from no-balls. Illegal deliveries do not consume a legal
ball. The model does not represent batter/bowler identities, super overs, or
run-scoring wicket deliveries. Wickets are treated as legal, zero-run balls.

The evaluation reports conditional node Brier/log-loss metrics, illegal-run
distribution scores, and compares simulated first-innings scores, second-
innings runs per delivery, and wickets against the held-out matches. These are
baseline checks rather than a claim of fully calibrated match-winning
probabilities.

The report also prints equal-width reliability bins and compares team-aware
node scores with the league baseline on the same held-out deliveries for teams
seen during training. Team coverage is reported separately; these comparisons
are diagnostic, not proof of betting value. Treat score-distribution gaps as
model limitations to investigate, and do not infer profitability without
odds-based, leakage-safe backtests across multiple time periods.

## Ball-by-ball win probability: how to use it

Use the replay to explore how the match state changes the model's view of the
game. It answers questions such as:

- Which deliveries coincided with the largest swings in the first-batting
  team's estimated chance of winning?
- How did a wicket, boundary, or quiet over change that forecast?
- How did the projected first-innings score develop, and how close was it to
  the actual total?
- Where did this match's forecast agree with the result, and where did it
  miss?

This is a **retrospective model-exploration tool**: it replays a completed
match, then asks what the model would forecast from each observed score,
wicket, and ball count. It is useful for finding moments to investigate,
explaining the model's response to match events, and spotting forecast
limitations. A single match is an example, not evidence that the model is
accurate across matches.

### Step 1: Prepare the data

From the repository root, run the pipeline. It downloads the Cricsheet archive
if needed, processes the matches, fits and saves the model, and runs the
held-out evaluation:

```powershell
.\cricenv\Scripts\python.exe .\run_pipeline.py
```

This also creates the processed files under `data/processed/`, which the
replay needs. To download a fresh archive instead of reusing the existing raw
data, add `--refresh-data`.

### Step 2: Replay the latest supported match

```powershell
.\cricenv\Scripts\python.exe -m src.win_probability --simulations 200
```

By default, this selects the latest completed match with delivery data for
both innings on or after `2025-01-01`. The model is trained only on matches
before that date; using later matches for the replay keeps the example
separate from the model's training data.

### Step 3: Choose a particular match (optional)

List recent held-out matches and their Cricsheet IDs:

```powershell
.\cricenv\Scripts\python.exe -c "import pandas as pd; m=pd.read_parquet(r'data\processed\matches.parquet'); m['date']=pd.to_datetime(m['date']); m=m.loc[(m['date'] >= '2025-01-01') & m['winner'].notna() & ~m['has_super_over'].fillna(False), ['match_id','date','team_1','team_2','winner']]; print(m.sort_values('date').tail(10).to_string(index=False))"
```

Pass the desired ID to the replay:

```powershell
.\cricenv\Scripts\python.exe -m src.win_probability --match-id "MATCH_ID" --simulations 500
```

Only completed matches with both innings are supported. Matches decided by a
super over are not supported.

### Step 4: Read the outputs

The replay creates `results/win_probability/` with:

- `win_probability_timeline.png` — the first-batting team's estimated win
  probability after each delivery. The dashed line marks 50%, and the
  vertical divider marks the start of the second innings.
- `first_innings_projection.png` — the actual score so far alongside the
  model's projected final-score median and 10th–90th percentile range.
- `ball_by_ball.csv` — one row per delivery, including the score and wickets,
  before/after win probabilities, probability change, and first-innings score
  projection.

The command also prints the five largest absolute probability swings to help
you find moments to inspect. These swings are changes between adjacent
forecasts; they do not establish that a delivery caused the eventual result.

### Step 5: Adjust the run

`--simulations` controls the number of simulated remaining matches at each
delivery. More simulations reduce Monte Carlo noise but take longer. The
default is 200; start lower for a quick check or raise it for a smoother
visualization. Use `--seed` to reproduce the same simulation sample, or
`--output-dir` to save the CSV and charts somewhere else:

```powershell
.\cricenv\Scripts\python.exe -m src.win_probability `
  --match-id "MATCH_ID" --simulations 1000 --seed 42 `
  --output-dir .\results\my_match_replay
```

### What the forecast includes (and omits)

The win estimate is the fraction of simulated outcomes in which the
first-batting team wins, counting a simulated tie as half a win. It starts from
the observed match state and simulates the remaining innings. Team and venue
effects are used when the fitted model has profiles for them. The replay does
not use batter or bowler identities, recent player form, or bowling plans.

The shaded win-probability band is an approximate 95% **Monte Carlo sampling
interval**; it describes variation from the finite simulation count, not all
sources of model uncertainty. These retrospective charts are for model
exploration, not betting advice, and do not by themselves demonstrate
calibrated or reliable win probabilities.
