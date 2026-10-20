"""Simulate a fixture with teams, venue, ordered XIs, and bowling candidates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from functools import cache
from itertools import pairwise
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
    player_id_overrides: dict[str, str] | None = None,
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
        override = (player_id_overrides or {}).get(name)
        if override is not None:
            if override not in matches:
                raise ValueError(
                    f"Player ID override for {name!r} must match one of its "
                    "historical player IDs"
                )
            resolved.append(override)
            continue
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


def _player_id_overrides(
    value: Any,
    field: str,
    player_names: list[str],
) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object mapping player names to historical IDs")
    names_by_key = {name.casefold(): name for name in player_names}
    overrides: dict[str, str] = {}
    for name, player_id in value.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{field} keys must be non-empty player names")
        if not isinstance(player_id, str) or not player_id.strip():
            raise ValueError(f"{field} values must be non-empty player IDs")
        resolved_name = names_by_key.get(name.strip().casefold())
        if resolved_name is None:
            raise ValueError(f"{field} player {name!r} is not in the batting_order")
        overrides[resolved_name] = player_id.strip()
    return overrides


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


def _parse_batting_order(value: str, player_names: list[str]) -> list[str]:
    if not value.strip():
        return player_names.copy()
    try:
        positions = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError("enter comma-separated lineup numbers, or press Enter") from exc
    if len(positions) != len(player_names) or set(positions) != set(
        range(1, len(player_names) + 1)
    ):
        raise ValueError(
            f"enter each lineup number from 1 to {len(player_names)} exactly once"
        )
    return [player_names[position - 1] for position in positions]


def _prompt_batting_order(team_name: str, player_names: list[str]) -> list[str]:
    print(f"\n{team_name} XI:")
    for index, player_name in enumerate(player_names, start=1):
        print(f"  {index}. {player_name}")
    while True:
        try:
            value = input(
                "Batting order (enter XI numbers in order, or press Enter to keep it): "
            )
            return _parse_batting_order(value, player_names)
        except ValueError as exc:
            print(f"Invalid batting order: {exc}")


def _prompt_batting_first(team1: str, team2: str) -> str:
    while True:
        choice = input(f"Who bats first? 1. {team1}  2. {team2} [1/2]: ").strip()
        if choice == "1":
            return "team1"
        if choice == "2":
            return "team2"
        print("Enter 1 or 2.")


def _project_bowling_schedule(
    candidate_ids: list[str],
    preferred_schedule: tuple[str, ...],
    selected_overs: list[str],
) -> tuple[str, ...]:
    total_overs = simulator.MAX_LEGAL_BALLS // 6
    if (
        not candidate_ids
        or len(set(candidate_ids)) != len(candidate_ids)
        or len(selected_overs) > total_overs
        or any(bowler not in candidate_ids for bowler in selected_overs)
    ):
        raise ValueError("invalid bowling candidates or selected overs")
    selected_counts = Counter(selected_overs)
    if (
        max(selected_counts.values(), default=0) > 4
        or any(left == right for left, right in pairwise(selected_overs))
    ):
        raise ValueError("selected overs already break bowling allocation rules")

    @cache
    def complete(
        over_index: int,
        previous: str | None,
        counts: tuple[int, ...],
    ) -> tuple[str, ...] | None:
        if over_index == total_overs:
            return ()
        preferred = (
            preferred_schedule[over_index]
            if over_index < len(preferred_schedule)
            else None
        )
        available_indices = [
            index
            for index, player_id in enumerate(candidate_ids)
            if player_id != previous and counts[index] < 4
        ]
        available_indices.sort(
            key=lambda index: (
                candidate_ids[index] != preferred,
                counts[index],
                index,
            )
        )
        for index in available_indices:
            next_counts = list(counts)
            next_counts[index] += 1
            rest = complete(
                over_index + 1,
                candidate_ids[index],
                tuple(next_counts),
            )
            if rest is not None:
                return (candidate_ids[index], *rest)
        return None

    completion = complete(
        len(selected_overs),
        selected_overs[-1] if selected_overs else None,
        tuple(selected_counts[player_id] for player_id in candidate_ids),
    )
    if completion is None:
        raise ValueError("bowling choices leave no valid 20-over allocation")
    return (*selected_overs, *completion)


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
    bowling_candidate_ids: list[str],
    bowling_names: dict[str, str],
    ratings: player_model.PlayerRatings,
    rng: Any,
    forecast_simulations: int,
) -> simulator.GameState | None:
    striker_index, non_striker_index, next_batter_index = 0, 1, 2
    selected_overs: list[str] = []
    while not state.is_complete:
        current_batter = batting_order[striker_index]
        over_index = state.legal_balls // 6
        if over_index == len(selected_overs):
            previous = selected_overs[-1] if selected_overs else None
            counts = Counter(selected_overs)
            available = [
                player_id
                for player_id in bowling_candidate_ids
                if player_id != previous and counts[player_id] < 4
            ]
            while True:
                print(f"\nChoose a bowler for over {over_index + 1}:")
                for index, player_id in enumerate(available, start=1):
                    print(
                        f"  {index}. {bowling_names[player_id]} "
                        f"({counts[player_id]} overs used)"
                    )
                try:
                    choice = input("Bowler number: ").strip()
                    if not choice.isdigit() or not 1 <= int(choice) <= len(available):
                        raise ValueError("choose one of the listed bowler numbers")
                    selected = available[int(choice) - 1]
                    projected_schedule = _project_bowling_schedule(
                        bowling_candidate_ids,
                        bowling_schedule,
                        selected_overs + [selected],
                    )
                    selected_overs.append(selected)
                    break
                except ValueError as exc:
                    print(f"Invalid bowler choice: {exc}")
        else:
            projected_schedule = _project_bowling_schedule(
                bowling_candidate_ids,
                bowling_schedule,
                selected_overs,
            )
        current_bowler = projected_schedule[over_index]
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
                bowling_order=projected_schedule,
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
    first_bowling_candidate_ids: list[str],
    second_bowling_candidate_ids: list[str],
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
        "subsequent forecasts. Select teams, batters, and bowlers by number "
        "from the displayed lists; no player names need to be typed."
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
        bowling_candidate_ids=first_bowling_candidate_ids,
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
        bowling_candidate_ids=second_bowling_candidate_ids,
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
        team1_player_ids = _player_id_overrides(
            lineups["team1"].get("player_ids"),
            "team1 player_ids",
            team1_batters,
        )
        team2_batters = _required_names(
            lineups["team2"]["batting_order"], "team2 batting_order", 11
        )
        team2_bowlers = _bowling_candidates(
            lineups["team2"].get("bowlers"), "team2 bowlers", team2_batters
        )
        team2_player_ids = _player_id_overrides(
            lineups["team2"].get("player_ids"),
            "team2 player_ids",
            team2_batters,
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
            team1_batters,
            deliveries,
            matches,
            as_of_date=match_date,
            player_id_overrides=team1_player_ids,
        )
        team1_bowler_ids = _resolve_player_ids(
            team1_bowlers,
            deliveries,
            matches,
            as_of_date=match_date,
            player_id_overrides=team1_player_ids,
        )
        team2_batter_ids = _resolve_player_ids(
            team2_batters,
            deliveries,
            matches,
            as_of_date=match_date,
            player_id_overrides=team2_player_ids,
        )
        team2_bowler_ids = _resolve_player_ids(
            team2_bowlers,
            deliveries,
            matches,
            as_of_date=match_date,
            player_id_overrides=team2_player_ids,
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

    if args.interactive:
        batting_first = _prompt_batting_first(team1, team2)
        original_team1_batters = team1_batters.copy()
        original_team2_batters = team2_batters.copy()
        team1_batters = _prompt_batting_order(team1, team1_batters)
        team2_batters = _prompt_batting_order(team2, team2_batters)
        team1_ids_by_name = dict(
            zip(original_team1_batters, team1_batter_ids, strict=True)
        )
        team2_ids_by_name = dict(
            zip(original_team2_batters, team2_batter_ids, strict=True)
        )
        team1_batter_ids = [team1_ids_by_name[name] for name in team1_batters]
        team2_batter_ids = [team2_ids_by_name[name] for name in team2_batters]

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
            first_bowling_candidate_ids=(
                team2_bowler_ids if batting_first == "team1" else team1_bowler_ids
            ),
            second_bowling_candidate_ids=(
                team1_bowler_ids if batting_first == "team1" else team2_bowler_ids
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
