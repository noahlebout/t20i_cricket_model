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