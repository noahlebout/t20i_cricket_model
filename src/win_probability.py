"""Replay a held-out match as ball-by-ball win-probability forecasts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import features, model, simulator


def _model_context(
    fitted_model: model.NodeProbs,
    batting_team: str,
    bowling_team: str,
    venue: Any,
) -> tuple[str | None, str | None, str | None]:
    known_teams = set(fitted_model.team_names)
    has_team_profiles = (
        fitted_model.batting_team_probs is not None
        and fitted_model.bowling_team_probs is not None
    )
    if has_team_profiles and batting_team in known_teams and bowling_team in known_teams:
        model_batting_team, model_bowling_team = batting_team, bowling_team
    else:
        model_batting_team, model_bowling_team = None, None

    known_venues = {name.casefold(): name for name in fitted_model.venue_names}
    model_venue = (
        known_venues.get(venue.strip().casefold())
        if isinstance(venue, str) and fitted_model.venue_probs is not None
        else None
    )
    return model_batting_team, model_bowling_team, model_venue


def _probability_table(
    fitted_model: model.NodeProbs,
    first_team: str,
    second_team: str,
    venue: str | None,
) -> np.ndarray:
    table = np.empty(
        (features.N_INNINGS, features.N_STATES, len(fitted_model.node_names)),
        dtype=float,
    )
    for innings_number in (1, 2):
        batting_team = first_team if innings_number == 1 else second_team
        bowling_team = second_team if innings_number == 1 else first_team
        batting_team, bowling_team, model_venue = _model_context(
            fitted_model,
            batting_team,
            bowling_team,
            venue,
        )
        for state_id in range(features.N_STATES):
            table[innings_number - 1, state_id] = model.get_node_probs(
                fitted_model,
                state_id,
                batting_team=batting_team,
                bowling_team=bowling_team,
                venue=model_venue,
            )
    return table


def _simulate_innings_batch(
    probability_table: np.ndarray,
    illegal_run_probs: np.ndarray | None,
    state: simulator.GameState,
    *,
    n_simulations: int,
    rng: np.random.Generator,
    target_runs: int | np.ndarray | None = None,
) -> np.ndarray:
    legal_balls = np.full(n_simulations, state.legal_balls, dtype=np.int16)
    wickets = np.full(n_simulations, state.wickets, dtype=np.int8)
    runs = np.full(n_simulations, state.runs, dtype=np.int32)
    if isinstance(target_runs, np.ndarray):
        targets: np.ndarray | None = target_runs.astype(np.int32, copy=False)
    else:
        target = state.target_runs if target_runs is None else target_runs
        targets = (
            None
            if target is None
            else np.full(n_simulations, target, dtype=np.int32)
        )
    step = 0
    while True:
        active = legal_balls < simulator.MAX_LEGAL_BALLS
        active &= wickets < simulator.N_WICKETS
        if targets is not None:
            active &= runs < targets
        active_indices = np.flatnonzero(active)
        if not len(active_indices):
            return runs
        step += 1
        if step > 10_000:
            raise RuntimeError("Simulated innings exceeded 10,000 deliveries")

        active_balls = legal_balls[active_indices]
        phase = np.searchsorted(features.PHASE_STARTS, active_balls, side="right") - 1
        wicket_bucket = np.minimum(wickets[active_indices], features.N_WICKET_BUCKETS - 1)
        state_ids = (
            (state.innings_number - 1) * features.N_PHASES * features.N_WICKET_BUCKETS
            + phase * features.N_WICKET_BUCKETS
            + wicket_bucket
        )
        probabilities = probability_table[state.innings_number - 1, state_ids]
        random_values = rng.random((len(active_indices), 9))

        wicket = random_values[:, 0] < probabilities[:, 0]
        legal = wicket | (random_values[:, 1] < probabilities[:, 1])
        scored = (
            ~wicket
            & legal
            & (random_values[:, 3] < probabilities[:, 3])
        )
        boundary = scored & (random_values[:, 4] < probabilities[:, 4])
        six = boundary & (random_values[:, 5] < probabilities[:, 5])
        four = boundary & ~six
        not_boundary = scored & ~boundary
        one = not_boundary & (random_values[:, 6] < probabilities[:, 6])
        remaining = not_boundary & ~one
        two = remaining & (random_values[:, 7] < probabilities[:, 7])
        three = remaining & ~two
        delivery_runs = (
            1 * one
            + 2 * two
            + 3 * three
            + 4 * four
            + 6 * six
        ).astype(np.int32)

        illegal = ~wicket & ~legal
        if illegal.any():
            illegal_rows = np.flatnonzero(illegal)
            if illegal_run_probs is None:
                delivery_runs[illegal_rows] = 1
            else:
                wide = random_values[illegal_rows, 2] < probabilities[illegal_rows, 2]
                type_indices = np.where(wide, 0, 1)
                distributions = illegal_run_probs[type_indices, state_ids[illegal_rows]]
                cumulative = np.cumsum(distributions, axis=1)
                sampled = (
                    random_values[illegal_rows, 8, None] >= cumulative
                ).sum(axis=1)
                delivery_runs[illegal_rows] = np.minimum(
                    sampled + 1,
                    distributions.shape[1],
                )

        legal_balls[active_indices] += legal.astype(np.int16)
        wickets[active_indices] += wicket.astype(np.int8)
        runs[active_indices] += delivery_runs


def _forecast_state(
    fitted_model: model.NodeProbs,
    state: simulator.GameState,
    *,
    first_team: str,
    second_team: str,
    venue: str | None,
    first_runs: int | None,
    n_simulations: int,
    seed: int,
    probability_table: np.ndarray | None = None,
) -> tuple[float, float, float, float, float]:
    if probability_table is None:
        probability_table = _probability_table(
            fitted_model,
            first_team,
            second_team,
            venue,
        )
    rng = np.random.default_rng(seed)

    if state.innings_number == 1:
        projected_scores = _simulate_innings_batch(
            probability_table,
            fitted_model.illegal_run_probs,
            state,
            n_simulations=n_simulations,
            rng=rng,
        )
        chase = _simulate_innings_batch(
            probability_table,
            fitted_model.illegal_run_probs,
            simulator.GameState(innings_number=2),
            n_simulations=n_simulations,
            rng=rng,
            target_runs=projected_scores + 1,
        )
        outcomes = (chase < projected_scores).astype(float) + 0.5 * (
            chase == projected_scores
        )
    else:
        if first_runs is None:
            raise ValueError("first_runs is required when forecasting the second innings")
        projected_scores = _simulate_innings_batch(
            probability_table,
            fitted_model.illegal_run_probs,
            state,
            n_simulations=n_simulations,
            rng=rng,
        )
        outcomes = (projected_scores < first_runs).astype(float) + 0.5 * (
            projected_scores == first_runs
        )

    low, median, high = np.quantile(projected_scores, [0.1, 0.5, 0.9])
    probability = float(outcomes.mean())
    standard_error = (
        float(outcomes.std(ddof=1) / np.sqrt(n_simulations))
        if n_simulations > 1
        else 0.0
    )
    return (
        probability,
        standard_error,
        float(low),
        float(median),
        float(high),
    )


def analyze_match(
    fitted_model: model.NodeProbs,
    deliveries: pd.DataFrame,
    matches: pd.DataFrame,
    match_id: str,
    *,
    n_simulations: int = 200,
    seed: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Forecast the match after each observed delivery using only its current state."""
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    required_match_columns = {
        "match_id",
        "date",
        "team_1",
        "team_2",
        "first_batting_team",
        "winner",
    }
    missing = required_match_columns.difference(matches.columns)
    if missing:
        raise ValueError(f"matches is missing required columns: {sorted(missing)}")
    required_delivery_columns = {
        "match_id",
        "innings_number",
        "legal_balls_before",
        "wickets_before",
        "runs_before",
        "is_legal",
        "is_wicket",
        "total_runs",
    }
    missing = required_delivery_columns.difference(deliveries.columns)
    if missing:
        raise ValueError(f"deliveries is missing required columns: {sorted(missing)}")

    selected = matches.loc[matches["match_id"].astype(str) == str(match_id)]
    if len(selected) != 1:
        raise ValueError(f"Expected one match with ID {match_id!r}, found {len(selected)}")
    match = selected.iloc[0]
    if pd.isna(match["winner"]):
        raise ValueError("The selected match has no winner")
    match_date = pd.to_datetime(match["date"], errors="raise")
    if match_date < pd.Timestamp(model.TRAIN_CUTOFF_DATE):
        raise ValueError(
            f"The selected match must be held out on or after {model.TRAIN_CUTOFF_DATE}"
        )
    has_super_over = match.get("has_super_over", False)
    if pd.notna(has_super_over) and bool(has_super_over):
        raise ValueError("Matches decided by a super over are not supported")

    match_deliveries = deliveries.loc[
        deliveries["match_id"].astype(str) == str(match_id)
    ].copy()
    if "is_super_over" in match_deliveries:
        match_deliveries = match_deliveries.loc[
            ~match_deliveries["is_super_over"].astype(bool)
        ]
    sort_columns = [
        column
        for column in ("innings_number", "over", "ball_in_over")
        if column in match_deliveries
    ]
    if sort_columns:
        match_deliveries = match_deliveries.sort_values(sort_columns, kind="stable")
    if match_deliveries.empty:
        raise ValueError(f"No deliveries found for match {match_id!r}")

    first_team = str(match["first_batting_team"])
    team_1, team_2 = str(match["team_1"]), str(match["team_2"])
    if first_team == team_1:
        second_team = team_2
    elif first_team == team_2:
        second_team = team_1
    else:
        raise ValueError("first_batting_team is not one of the match's two teams")
    winner = str(match["winner"])
    if winner not in {first_team, second_team}:
        raise ValueError("The match winner is not one of the two teams")

    innings_rows = [
        match_deliveries.loc[match_deliveries["innings_number"] == number]
        for number in (1, 2)
    ]
    if any(rows.empty for rows in innings_rows):
        raise ValueError("The selected match must contain deliveries from both innings")
    actual_first_runs = int(innings_rows[0]["total_runs"].sum())
    venue_value = match.get("venue")
    venue = venue_value if isinstance(venue_value, str) else None
    probability_table = _probability_table(
        fitted_model,
        first_team,
        second_team,
        venue,
    )
    records: list[dict[str, Any]] = []
    previous_probability = 0.0
    previous_standard_error = 0.0

    initial_forecast = _forecast_state(
        fitted_model,
        simulator.GameState(innings_number=1),
        first_team=first_team,
        second_team=second_team,
        venue=venue,
        first_runs=None,
        n_simulations=n_simulations,
        seed=seed,
        probability_table=probability_table,
    )
    previous_probability, previous_standard_error = initial_forecast[:2]
    delivery_number = 0

    for innings_number, rows in enumerate(innings_rows, start=1):
        first_runs = actual_first_runs if innings_number == 2 else None
        target_runs = actual_first_runs + 1 if innings_number == 2 else None
        rows = rows.reset_index(drop=True)
        for row_index, delivery in rows.iterrows():
            delivery_number += 1
            state_before = simulator.GameState(
                innings_number=innings_number,
                legal_balls=int(delivery["legal_balls_before"]),
                wickets=int(delivery["wickets_before"]),
                runs=int(delivery["runs_before"]),
                target_runs=target_runs,
            )
            if row_index + 1 < len(rows):
                next_delivery = rows.iloc[row_index + 1]
                state_after = simulator.GameState(
                    innings_number=innings_number,
                    legal_balls=int(next_delivery["legal_balls_before"]),
                    wickets=int(next_delivery["wickets_before"]),
                    runs=int(next_delivery["runs_before"]),
                    target_runs=target_runs,
                )
            else:
                state_after = simulator.GameState(
                    innings_number=innings_number,
                    legal_balls=state_before.legal_balls + int(bool(delivery["is_legal"])),
                    wickets=state_before.wickets + int(bool(delivery["is_wicket"])),
                    runs=state_before.runs + int(delivery["total_runs"]),
                    target_runs=target_runs,
                )
            (
                probability,
                standard_error,
                score_low,
                score_median,
                score_high,
            ) = _forecast_state(
                fitted_model,
                state_after,
                first_team=first_team,
                second_team=second_team,
                venue=venue,
                first_runs=first_runs,
                n_simulations=n_simulations,
                seed=seed,
                probability_table=probability_table,
            )
            over_text = (
                f"{state_after.legal_balls // 6}."
                f"{state_after.legal_balls % 6}"
            )
            records.append(
                {
                    "match_id": str(match_id),
                    "date": match_date.date().isoformat(),
                    "innings": innings_number,
                    "delivery_number": delivery_number,
                    "over": over_text,
                    "legal_balls": state_after.legal_balls,
                    "runs": state_after.runs,
                    "wickets": state_after.wickets,
                    "delivery_runs": int(delivery["total_runs"]),
                    "is_wicket": bool(delivery["is_wicket"]),
                    "first_batting_team": first_team,
                    "actual_winner": winner,
                    "actual_first_team_win": float(winner == first_team),
                    "win_probability_before": previous_probability,
                    "win_probability_after": probability,
                    "win_probability_change": probability - previous_probability,
                    "win_probability_before_standard_error": previous_standard_error,
                    "win_probability_after_standard_error": standard_error,
                    "projected_first_innings_runs_low": (
                        score_low if innings_number == 1 else np.nan
                    ),
                    "projected_first_innings_runs_median": (
                        score_median if innings_number == 1 else np.nan
                    ),
                    "projected_first_innings_runs_high": (
                        score_high if innings_number == 1 else np.nan
                    ),
                }
            )
            previous_probability = probability
            previous_standard_error = standard_error

    result = pd.DataFrame.from_records(records)
    metadata = {
        "match_id": str(match_id),
        "date": match_date.date().isoformat(),
        "first_batting_team": first_team,
        "second_batting_team": second_team,
        "winner": winner,
        "venue": venue,
        "actual_first_innings_runs": actual_first_runs,
        "n_deliveries": len(result),
        "simulations_per_forecast": n_simulations,
        "seed": seed,
    }
    return metadata, result


def plot_match(metadata: dict[str, Any], results: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    innings_boundary = int((results["innings"] == 1).sum())

    fig, axis = plt.subplots(figsize=(12, 5.5))
    axis.plot(
        results["delivery_number"],
        results["win_probability_after"] * 100,
        color="#147d64",
        linewidth=1.8,
        label="Win probability",
    )
    probability = results["win_probability_after"].to_numpy()
    standard_error = results["win_probability_after_standard_error"].to_numpy()
    axis.fill_between(
        results["delivery_number"],
        np.clip(probability - 1.96 * standard_error, 0, 1) * 100,
        np.clip(probability + 1.96 * standard_error, 0, 1) * 100,
        color="#147d64",
        alpha=0.16,
        label="Approx. 95% Monte Carlo interval",
    )
    axis.axhline(50, color="#666666", linewidth=1, linestyle="--")
    axis.axvline(innings_boundary + 0.5, color="#999999", linewidth=1, linestyle=":")
    axis.set(
        title=(
            f"{metadata['first_batting_team']} vs {metadata['second_batting_team']}: "
            f"win probability after each ball"
        ),
        xlabel="Delivery number (illegal balls included)",
        ylabel=f"{metadata['first_batting_team']} win probability (%)",
        ylim=(0, 100),
    )
    axis.legend(frameon=False, loc="upper left")
    axis.text(
        innings_boundary / 2,
        4,
        "First innings",
        ha="center",
        color="#555555",
    )
    axis.text(
        (innings_boundary + len(results)) / 2,
        4,
        "Second innings",
        ha="center",
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "win_probability_timeline.png", dpi=160)
    plt.close(fig)

    first_innings = results.loc[results["innings"] == 1]
    fig, axis = plt.subplots(figsize=(12, 5.5))
    x = first_innings["legal_balls"].to_numpy()
    actual_runs = first_innings["runs"].to_numpy()
    low = first_innings["projected_first_innings_runs_low"].to_numpy()
    median = first_innings["projected_first_innings_runs_median"].to_numpy()
    high = first_innings["projected_first_innings_runs_high"].to_numpy()
    axis.fill_between(x, low, high, color="#83c9b6", alpha=0.35, label="Model 10th–90th percentile")
    axis.plot(x, median, color="#147d64", linewidth=1.8, label="Model median final score")
    axis.plot(x, actual_runs, color="#333333", linewidth=1.5, label="Actual score")
    axis.set(
        title=(
            f"First-innings score projection: "
            f"{metadata['first_batting_team']} vs {metadata['second_batting_team']}"
        ),
        xlabel="Legal balls bowled",
        ylabel="Innings runs",
    )
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "first_innings_projection.png", dpi=160)
    plt.close(fig)


def _latest_supported_match(
    matches: pd.DataFrame,
    deliveries: pd.DataFrame,
) -> str:
    match_dates = pd.to_datetime(matches["date"], errors="raise")
    candidates = matches.loc[
        (match_dates >= pd.Timestamp(model.TRAIN_CUTOFF_DATE))
        & matches["winner"].notna()
    ].copy()
    if "has_super_over" in candidates:
        candidates = candidates.loc[
            ~candidates["has_super_over"].fillna(False).astype(bool)
        ]
    candidates["_sort_date"] = pd.to_datetime(candidates["date"], errors="raise")
    candidates = candidates.sort_values(["_sort_date", "match_id"], ascending=False)
    for match_id in candidates["match_id"].astype(str):
        match_rows = deliveries.loc[deliveries["match_id"].astype(str) == match_id]
        if "is_super_over" in match_rows:
            match_rows = match_rows.loc[~match_rows["is_super_over"].astype(bool)]
        innings = [
            match_rows.loc[match_rows["innings_number"] == number]
            for number in (1, 2)
        ]
        if all(not rows.empty for rows in innings):
            return match_id
    raise ValueError(
        "No held-out completed match with deliveries from both innings was found"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--match-id",
        default=None,
        help="match ID to replay (default: latest completed held-out match)",
    )
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "win_probability",
        help="directory for the per-delivery CSV and plots",
    )
    args = parser.parse_args()
    if args.simulations <= 0:
        parser.error("--simulations must be positive")

    deliveries = pd.read_parquet(model.PROCESSED / "deliveries.parquet")
    matches = pd.read_parquet(model.PROCESSED / "matches.parquet")
    fitted_model = model.fit(deliveries, matches)
    match_id = args.match_id or _latest_supported_match(
        matches,
        deliveries,
    )
    metadata, results = analyze_match(
        fitted_model,
        deliveries,
        matches,
        match_id,
        n_simulations=args.simulations,
        seed=args.seed,
    )
    plot_match(metadata, results, args.output_dir)
    results.to_csv(args.output_dir / "ball_by_ball.csv", index=False)
    biggest_swings = results.reindex(
        results["win_probability_change"].abs()
        .sort_values(ascending=False)
        .head(5)
        .index
    )
    print(
        f"{metadata['date']}: {metadata['first_batting_team']} vs "
        f"{metadata['second_batting_team']} at {metadata['venue'] or 'unknown venue'}; "
        f"winner: {metadata['winner']}"
    )
    print(
        f"Generated {metadata['n_deliveries']:,} ball-by-ball forecasts "
        f"using {metadata['simulations_per_forecast']:,} simulations per state."
    )
    opening_forecast = results.loc[results["innings"] == 1].iloc[0]
    print(
        f"First-innings score: actual {metadata['actual_first_innings_runs']}; "
        f"opening median forecast {opening_forecast['projected_first_innings_runs_median']:.0f} "
        f"(10th-90th percentile "
        f"{opening_forecast['projected_first_innings_runs_low']:.0f}-"
        f"{opening_forecast['projected_first_innings_runs_high']:.0f})."
    )
    print("Largest absolute win-probability changes:")
    for row in biggest_swings.itertuples(index=False):
        event = "wicket" if row.is_wicket else f"{row.delivery_runs} run(s)"
        print(
            f"  Innings {row.innings}, over {row.over}: "
            f"{row.win_probability_change:+.1%} "
            f"({row.win_probability_before:.1%} -> {row.win_probability_after:.1%}; "
            f"{event})"
        )
    print(f"Timeline: {args.output_dir / 'win_probability_timeline.png'}")
    print(f"Score projection: {args.output_dir / 'first_innings_projection.png'}")
    print(f"Per-delivery data: {args.output_dir / 'ball_by_ball.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
