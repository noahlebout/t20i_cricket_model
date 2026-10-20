"""Simulate a fixture with teams, venue, ordered XIs, and bowling candidates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from src import model, player_model, simulator


def _resolve_player_ids(
    player_names: list[str],
    deliveries: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
) -> list[str]:
    known_names: dict[str, set[str]] = {}
    eligible_ids = matches.loc[
        pd.to_datetime(matches["date"], errors="raise") < as_of_date, "match_id"
    ]
    history = deliveries.loc[deliveries["match_id"].isin(eligible_ids)]
    for name_column, id_column in (
        ("batter", "batter_id"),
        ("bowler", "bowler_id"),
    ):
        names = history[[name_column, id_column]].dropna().drop_duplicates()
        for name, player_id in names.itertuples(index=False, name=None):
            known_names.setdefault(str(name).strip().casefold(), set()).add(str(player_id))

    resolved: list[str] = []
    for name in player_names:
        matches = known_names.get(name.strip().casefold(), set())
        if len(matches) > 1:
            raise ValueError(
                f"Player name {name!r} matches multiple historical player IDs; use a unique name"
            )
        if matches:
            resolved.append(next(iter(matches)))
        else:
            resolved.append(f"debut:{name.strip().casefold()}")
    if len(set(resolved)) != len(resolved):
        raise ValueError("Each lineup must contain unique players")
    return resolved


def _required_names(value: Any, field: str, count: int) -> list[str]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{field} must be a list of exactly {count} player names")
    if any(not isinstance(name, str) or not name.strip() for name in value):
        raise ValueError(f"{field} must contain non-empty player names")
    return [name.strip() for name in value]


def _bowling_candidates(value: Any, field: str, batting_order: list[str]) -> list[str]:
    if value is None:
        return batting_order
    if (
        not isinstance(value, list)
        or not 5 <= len(value) <= 11
        or any(not isinstance(name, str) or not name.strip() for name in value)
    ):
        raise ValueError(f"{field} must be omitted or contain 5 to 11 player names")
    candidates = [name.strip() for name in value]
    if len({name.casefold() for name in candidates}) != len(candidates):
        raise ValueError(f"{field} must contain unique player names")
    batting_names = {name.casefold() for name in batting_order}
    if any(name.casefold() not in batting_names for name in candidates):
        raise ValueError(f"{field} must be selected from that team's batting_order")
    return candidates


def _display_player_form(
    team_name: str,
    names: list[str],
    ids: list[str],
    ratings: player_model.PlayerRatings,
) -> None:
    print(f"\n{team_name} player form (last 10 appearances before the fixture date):")
    for display_name, player_id in zip(names, ids, strict=True):
        batting = ratings.batting_form.get(player_id)
        if batting is None:
            print(f"  {display_name}: no batting history; using training debut-player average")
        else:
            print(
                f"  {display_name}: batting {batting['runs']} runs off "
                f"{batting['balls_faced']} balls (SR {batting['strike_rate']:.1f})"
            )
        bowling = ratings.bowling_form.get(player_id)
        if bowling is not None:
            print(
                f"    bowling {bowling['wickets']} wickets, "
                f"{bowling['runs_conceded']} runs in {bowling['legal_balls']} legal balls "
                f"(econ {bowling['economy']:.2f})"
            )


def _display_bowling_plan(
    team_name: str,
    candidate_names: list[str],
    candidate_ids: list[str],
    schedule: tuple[str, ...],
    ratings: player_model.PlayerRatings,
) -> None:
    names_by_id = dict(zip(candidate_ids, candidate_names, strict=True))
    scheduled_overs = Counter(schedule)
    print(f"\n{team_name} bowling plan (20-over allocation):")
    for player_id in dict.fromkeys(schedule):
        usage = ratings.bowling_usage.get(player_id, {})
        preferred = ", ".join(str(over + 1) for over in usage.get("preferred_overs", []))
        print(
            f"  {names_by_id[player_id]}: {scheduled_overs[player_id]} overs; "
            f"recent average {usage.get('avg_overs', 0.0):.1f}; "
            f"most frequent over(s): {preferred or 'no history'}"
        )


def _parse_observed_delivery(
    value: str,
) -> tuple[int, bool, bool, str, int | None] | None:
    normalized = value.strip().casefold()
    if normalized in {"q", "quit"}:
        return None
    parts = normalized.split(":")
    prefix = parts[0]
    if prefix in {"w", "wk", "wicket", "wd", "nb"}:
        if len(parts) > 3:
            raise ValueError("use FORMAT:<total runs>[:<completed runs>]")
        is_wicket = prefix in {"w", "wk", "wicket"}
        is_legal = prefix in {"w", "wk", "wicket"}
        if len(parts) == 1:
            if not is_wicket:
                raise ValueError(f"enter total runs after {prefix}:, e.g. {prefix}:2")
            runs = 0
            strike_runs = 0
        else:
            try:
                runs = int(parts[1])
                strike_runs = int(parts[2]) if len(parts) == 3 else None
            except ValueError as exc:
                raise ValueError("runs must be non-negative integers") from exc
        if runs < 0 or (strike_runs is not None and strike_runs < 0):
            raise ValueError("runs cannot be negative")
        if not is_wicket and runs < 1:
            raise ValueError(f"{prefix}: must include at least one total run")
        if prefix in {"w", "wk", "wicket"}:
            outcome = "wicket" if runs == 0 else f"wicket+{runs}"
        else:
            outcome = ("wide" if prefix == "wd" else "no_ball") + f"+{runs}"
        return runs, is_wicket, is_legal, outcome, strike_runs
    try:
        runs = int(normalized)
    except ValueError as exc:
        raise ValueError(
            "enter runs (0-6), W[:<runs>], WD:<total>[:<completed runs>], "
            "NB:<total>[:<completed runs>], or q"
        ) from exc
    if runs < 0:
        raise ValueError("runs cannot be negative")
    labels = {0: "dot", 1: "one", 2: "two", 3: "three", 4: "four", 6: "six"}
    return runs, False, True, labels.get(runs, f"runs-{runs}"), None


def _interactive_innings(
    fitted_model: model.NodeProbs,
    *,
    state: simulator.GameState,
    batting_team: str,
    bowling_team: str,
    venue: str,
    batting_order: list[str],
    batting_names: list[str],
    bowling_schedule: tuple[str, ...],
    bowling_names: dict[str, str],
    ratings: player_model.PlayerRatings,
    rng: Any,
    forecast_simulations: int,
) -> simulator.GameState | None:
    striker_index, non_striker_index, next_batter_index = 0, 1, 2
    while not state.is_complete:
        current_batter = batting_order[striker_index]
        current_bowler = bowling_schedule[state.legal_balls // 6]
        over_label = f"{state.legal_balls // 6}.{state.legal_balls % 6}"
        remaining = (
            f", target {state.target_runs}"
            if state.target_runs is not None
            else ""
        )
        print(
            f"\n{batting_team} {state.runs}/{state.wickets} "
            f"({over_label} overs{remaining}) | "
            f"striker {batting_names[striker_index]} | "
            f"bowler {bowling_names[current_bowler]}"
        )

        forecasts = [
            simulator.simulate_innings(
                fitted_model,
                state,
                rng=rng,
                batting_team=batting_team,
                bowling_team=bowling_team,
                venue=venue,
                batting_order=batting_order,
                bowling_order=bowling_schedule,
                player_ratings=ratings,
                striker_index=striker_index,
                non_striker_index=non_striker_index,
                next_batter_index=next_batter_index,
            )
            for _ in range(forecast_simulations)
        ]
        outcomes = Counter(result.deliveries[0].outcome for result in forecasts)
        outcome_summary = ", ".join(
            f"{outcome} {count / forecast_simulations:.0%}"
            for outcome, count in outcomes.most_common(6)
        )
        totals = np.array(
            [result.final_state.runs for result in forecasts], dtype=float
        )
        if state.target_runs is None:
            print(
                f"Projected innings total: mean {totals.mean():.1f}; "
                f"median {np.median(totals):.0f}; "
                f"10th-90th percentile {np.quantile(totals, 0.1):.0f}-"
                f"{np.quantile(totals, 0.9):.0f}"
            )
        else:
            win_probability = float(np.mean(totals >= state.target_runs))
            print(
                f"Projected chase: {win_probability:.1%} win probability; "
                f"final score median {np.median(totals):.0f}"
            )
        print(f"Next-ball outcome estimates: {outcome_summary}")
        sampled = forecasts[0].deliveries[0]
        print(f"One sampled model outcome: {sampled.outcome}")

        while True:
            try:
                observed = _parse_observed_delivery(
                    input(
                        "Actual result [0-6, W[:runs], WD/NB:total[:completed], "
                        "q to quit]: "
                    )
                )
                break
            except ValueError as exc:
                print(f"Invalid result: {exc}")
        if observed is None:
            return None

        runs, is_wicket, is_legal, outcome, strike_runs = observed
        delivery = simulator.record_delivery(
            state,
            runs_scored=runs,
            is_wicket=is_wicket,
            is_legal=is_legal,
            outcome=outcome,
            batter_id=current_batter,
            bowler_id=current_bowler,
            strike_runs=strike_runs,
        )
        state = delivery.state_after
        (
            striker_index,
            non_striker_index,
            next_batter_index,
        ) = simulator.advance_batting_positions(
            striker_index,
            non_striker_index,
            next_batter_index,
            delivery,
        )
        print(f"Recorded {outcome}; score now {state.runs}/{state.wickets}.")
    return state


def _interactive_match(
    fitted_model: model.NodeProbs,
    *,
    first_team: str,
    second_team: str,
    venue: str,
    first_batting_order: list[str],
    second_batting_order: list[str],
    first_bowling_schedule: tuple[str, ...],
    second_bowling_schedule: tuple[str, ...],
    first_batting_names: list[str],
    second_batting_names: list[str],
    first_bowling_names: dict[str, str],
    second_bowling_names: dict[str, str],
    ratings: player_model.PlayerRatings,
    rng: Any,
    forecast_simulations: int,
) -> None:
    print(
        "\nInteractive ball-by-ball mode: enter the observed delivery after "
        "each forecast. The fitted model is not retrained; each recorded ball "
        "updates the score, wickets, strike, and next-ball state used for "
        "subsequent forecasts."
    )
    first = _interactive_innings(
        fitted_model,
        state=simulator.GameState(innings_number=1),
        batting_team=first_team,
        bowling_team=second_team,
        venue=venue,
        batting_order=first_batting_order,
        batting_names=first_batting_names,
        bowling_schedule=first_bowling_schedule,
        bowling_names=first_bowling_names,
        ratings=ratings,
        rng=rng,
        forecast_simulations=forecast_simulations,
    )
    if first is None:
        print("Interactive session stopped.")
        return
    print(f"\nInnings complete: {first_team} {first.runs}/{first.wickets}.")
    second = _interactive_innings(
        fitted_model,
        state=simulator.GameState(
            innings_number=2,
            target_runs=first.runs + 1,
        ),
        batting_team=second_team,
        bowling_team=first_team,
        venue=venue,
        batting_order=second_batting_order,
        batting_names=second_batting_names,
        bowling_schedule=second_bowling_schedule,
        bowling_names=second_bowling_names,
        ratings=ratings,
        rng=rng,
        forecast_simulations=forecast_simulations,
    )
    if second is None:
        print("Interactive session stopped.")
        return
    winner = (
        first_team
        if first.runs > second.runs
        else second_team
        if second.runs > first.runs
        else "Tie"
    )
    print(
        f"\nMatch complete: {first_team} {first.runs}/{first.wickets}, "
        f"{second_team} {second.runs}/{second.wickets}. Winner: {winner}."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        required=True,
        type=Path,
        help="JSON fixture with team names, venue, and an ordered XI per team",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    parser.add_argument(
        "--as-of",
        default=None,
        help="only use player/team history strictly before this date (default: today)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="record actual results ball by ball and refresh the remaining-innings forecast",
    )
    parser.add_argument(
        "--forecast-simulations",
        type=int,
        default=100,
        help="Monte Carlo runs for each interactive forecast (default: 100)",
    )
    args = parser.parse_args()
    if args.forecast_simulations < 1:
        parser.error("--forecast-simulations must be positive")

    try:
        fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
        team1 = fixture["team1"]
        team2 = fixture["team2"]
        venue = fixture["venue"]
        batting_first = fixture["batting_first"]
        lineups = fixture["lineups"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        parser.error(f"invalid fixture file: {exc}")

    if any(not isinstance(value, str) or not value.strip() for value in (team1, team2, venue)):
        parser.error("team1, team2, and venue must be non-empty strings")
    if team1 == team2:
        parser.error("team1 and team2 must be different")
    if batting_first not in {"team1", "team2"}:
        parser.error("batting_first must be 'team1' or 'team2'")
    if not isinstance(lineups, dict):
        parser.error("lineups must contain team1 and team2 lineups")
    if any(not isinstance(lineups.get(team), dict) for team in ("team1", "team2")):
        parser.error("lineups must contain team1 and team2 objects")

    try:
        team1_batters = _required_names(
            lineups["team1"]["batting_order"], "team1 batting_order", 11
        )
        team1_bowlers = _bowling_candidates(
            lineups["team1"].get("bowlers"), "team1 bowlers", team1_batters
        )
        team2_batters = _required_names(
            lineups["team2"]["batting_order"], "team2 batting_order", 11
        )
        team2_bowlers = _bowling_candidates(
            lineups["team2"].get("bowlers"), "team2 bowlers", team2_batters
        )
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(f"invalid fixture lineups: {exc}")

    fitted_model = model.load()
    known_venues = {name.casefold() for name in fitted_model.venue_names}
    if venue.strip().casefold() not in known_venues:
        parser.error(f"venue {venue!r} has no fitted venue profile")

    match_date = pd.Timestamp(args.as_of or fixture.get("match_date") or date.today())
    if pd.isna(match_date):
        parser.error("as-of/match_date must be a valid date")
    training_cutoff = pd.Timestamp(fitted_model.metadata.get("TRAIN_CUTOFF_DATE"))
    if match_date < training_cutoff:
        parser.error(
            f"fixture date must be on or after the model training cutoff "
            f"({training_cutoff.date()})"
        )
    processed = model.PROCESSED
    matches = pd.read_parquet(processed / "matches.parquet")
    deliveries = pd.read_parquet(processed / "deliveries.parquet")

    try:
        team1_batter_ids = _resolve_player_ids(
            team1_batters, deliveries, matches, as_of_date=match_date
        )
        team1_bowler_ids = _resolve_player_ids(
            team1_bowlers, deliveries, matches, as_of_date=match_date
        )
        team2_batter_ids = _resolve_player_ids(
            team2_batters, deliveries, matches, as_of_date=match_date
        )
        team2_bowler_ids = _resolve_player_ids(
            team2_bowlers, deliveries, matches, as_of_date=match_date
        )
        ratings = player_model.build_player_ratings(
            deliveries,
            matches,
            fitted_model,
            as_of_date=match_date,
        )
        team1_bowling_schedule = player_model.build_bowling_schedule(
            team1_bowler_ids, ratings
        )
        team2_bowling_schedule = player_model.build_bowling_schedule(
            team2_bowler_ids, ratings
        )
        team_form = player_model.team_form_guide(
            deliveries,
            matches,
            as_of_date=match_date,
        )
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(f"could not build pre-match player form: {exc}")

    if batting_first == "team1":
        first_team, second_team = team1, team2
        first_batting_order, first_bowling_order = (
            team1_batter_ids,
            team2_bowling_schedule,
        )
        second_batting_order, second_bowling_order = (
            team2_batter_ids,
            team1_bowling_schedule,
        )
        first_batters, second_batters = team1_batters, team2_batters
    else:
        first_team, second_team = team2, team1
        first_batting_order, first_bowling_order = (
            team2_batter_ids,
            team1_bowling_schedule,
        )
        second_batting_order, second_bowling_order = (
            team1_batter_ids,
            team2_bowling_schedule,
        )
        first_batters, second_batters = team2_batters, team1_batters

    if args.interactive:
        _interactive_match(
            fitted_model,
            first_team=first_team,
            second_team=second_team,
            venue=venue,
            first_batting_order=first_batting_order,
            second_batting_order=second_batting_order,
            first_bowling_schedule=(
                team2_bowling_schedule if batting_first == "team1" else team1_bowling_schedule
            ),
            second_bowling_schedule=(
                team1_bowling_schedule if batting_first == "team1" else team2_bowling_schedule
            ),
            first_batting_names=first_batters,
            second_batting_names=second_batters,
            first_bowling_names=dict(
                zip(
                    team2_bowler_ids if batting_first == "team1" else team1_bowler_ids,
                    team2_bowlers if batting_first == "team1" else team1_bowlers,
                    strict=True,
                )
            ),
            second_bowling_names=dict(
                zip(
                    team1_bowler_ids if batting_first == "team1" else team2_bowler_ids,
                    team1_bowlers if batting_first == "team1" else team2_bowlers,
                    strict=True,
                )
            ),
            ratings=ratings,
            rng=np.random.default_rng(args.seed),
            forecast_simulations=args.forecast_simulations,
        )
        return 0

    result = simulator.simulate_match(
        fitted_model,
        rng=args.seed,
        batting_first_team=first_team,
        bowling_first_team=second_team,
        venue=venue,
        batting_first_order=first_batting_order,
        bowling_first_order=first_bowling_order,
        batting_second_order=second_batting_order,
        bowling_second_order=second_bowling_order,
        player_ratings=ratings,
    )

    print(f"Fixture date: {match_date.date()} (all form data strictly before this date)")
    print(f"Venue: {venue}")
    print(
        f"{first_team}: {result.first_innings.final_state.runs}/"
        f"{result.first_innings.final_state.wickets}"
    )
    print(
        f"{second_team}: {result.second_innings.final_state.runs}/"
        f"{result.second_innings.final_state.wickets}"
    )
    winner = (
        first_team
        if result.winner_innings == 1
        else second_team
        if result.winner_innings == 2
        else "Tie"
    )
    print(f"Winner: {winner}")
    _display_bowling_plan(
        team1, team1_bowlers, team1_bowler_ids, team1_bowling_schedule, ratings
    )
    _display_bowling_plan(
        team2, team2_bowlers, team2_bowler_ids, team2_bowling_schedule, ratings
    )
    print("\nTeam form (last 10 completed matches before fixture date):")
    selected_team_form = team_form.loc[team_form["team"].isin([team1, team2])]
    if selected_team_form.empty:
        print("  No prior match history available for either team.")
    else:
        print(selected_team_form.to_string(index=False, float_format=lambda value: f"{value:.2f}"))

    _display_player_form(first_team, first_batters, first_batting_order, ratings)
    _display_player_form(second_team, second_batters, second_batting_order, ratings)
    if args.seed is not None:
        print(f"\nSeed: {args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
