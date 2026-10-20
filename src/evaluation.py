"""Held-out delivery calibration and simulated-match score evaluation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import features, model, simulator


def _binary_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    clipped = np.clip(predicted, 1e-15, 1 - 1e-15)
    return {
        "brier_score": float(np.mean((predicted - actual) ** 2)),
        "log_loss": float(
            -np.mean(actual * np.log(clipped) + (1 - actual) * np.log(1 - clipped))
        ),
    }


def heldout_deliveries(
    deliveries: pd.DataFrame, matches: pd.DataFrame
) -> pd.DataFrame:
    """Return non-super-over deliveries from matches on/after the model cutoff."""
    if "date" not in matches or "match_id" not in matches:
        raise ValueError("matches must contain 'date' and 'match_id' columns")
    match_dates = pd.to_datetime(matches["date"], errors="raise")
    cutoff = pd.Timestamp(model.TRAIN_CUTOFF_DATE)
    heldout_ids = matches.loc[match_dates >= cutoff, "match_id"]
    heldout = deliveries.loc[deliveries["match_id"].isin(heldout_ids)].copy()
    if "venue" in matches and "match_id" in heldout:
        venue_lookup = (
            matches[["match_id", "venue"]]
            .drop_duplicates("match_id")
            .set_index("match_id")["venue"]
        )
        heldout["venue"] = heldout["match_id"].map(venue_lookup)
    if "is_super_over" in heldout:
        heldout = heldout.loc[~heldout["is_super_over"].astype(bool)]
    if "innings_number" not in heldout:
        raise ValueError("deliveries must contain an 'innings_number' column")
    heldout = heldout.loc[heldout["innings_number"].between(1, features.N_INNINGS)]
    if heldout.empty:
        raise ValueError(f"No held-out deliveries found on or after {model.TRAIN_CUTOFF_DATE}")
    return features.add_state_features(heldout)


def node_calibration(
    fitted_model: model.NodeProbs, heldout: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    """Report binary calibration and held-out team-strength comparisons by node."""
    nodes = model.build_tree()
    if fitted_model.node_names != [node.name for node in nodes]:
        raise ValueError("Model node names do not match the current decision tree")
    results: dict[str, dict[str, Any]] = {}
    states = heldout["state_id"].to_numpy(dtype=int)
    batting_team_indices = {
        team: index for index, team in enumerate(fitted_model.team_names)
    }
    venue_indices = {
        venue.casefold(): index for index, venue in enumerate(fitted_model.venue_names)
    }
    roles = {
        **{node: ("batting_team", fitted_model.batting_team_probs) for node in model.BATTING_NODES},
        **{node: ("bowling_team", fitted_model.bowling_team_probs) for node in model.BOWLING_NODES},
    }
    for index, node in enumerate(nodes):
        reached = node.reach(heldout).fillna(False).to_numpy(dtype=bool)
        if not reached.any():
            continue
        actual = node.yes(heldout).fillna(False).to_numpy(dtype=bool)[reached].astype(float)
        predicted = fitted_model.probs[index, states[reached]]
        metrics: dict[str, Any] = {
            "observations": int(reached.sum()),
            "observed_rate": float(actual.mean()),
            "predicted_rate": float(predicted.mean()),
            **_binary_metrics(actual, predicted),
        }
        bin_ids = np.minimum((predicted * 10).astype(int), 9)
        metrics["reliability_bins"] = [
            {
                "count": int((bin_ids == bin_id).sum()),
                "predicted_rate": float(predicted[bin_ids == bin_id].mean()),
                "observed_rate": float(actual[bin_ids == bin_id].mean()),
            }
            for bin_id in range(10)
            if np.any(bin_ids == bin_id)
        ]

        if node.name in roles:
            team_column, team_probabilities = roles[node.name]
            comparison: dict[str, float | int] = {
                "eligible_observations": 0,
                "coverage": 0.0,
                "baseline_brier_score": float("nan"),
                "team_brier_score": float("nan"),
                "baseline_log_loss": float("nan"),
                "team_log_loss": float("nan"),
            }
            if team_probabilities is not None and team_column in heldout:
                team_names = (
                    heldout.loc[reached, team_column]
                    .astype("string")
                    .str.strip()
                )
                team_indices = team_names.map(batting_team_indices).fillna(-1).to_numpy(dtype=int)
                known = team_indices >= 0
                comparison["eligible_observations"] = int(known.sum())
                comparison["coverage"] = float(known.mean())
                if known.any():
                    team_predictions = team_probabilities[
                        team_indices[known], index, states[reached][known]
                    ]
                    baseline_metrics = _binary_metrics(actual[known], predicted[known])
                    team_metrics = _binary_metrics(actual[known], team_predictions)
                    comparison.update(
                        {
                            "baseline_brier_score": baseline_metrics["brier_score"],
                            "team_brier_score": team_metrics["brier_score"],
                            "baseline_log_loss": baseline_metrics["log_loss"],
                            "team_log_loss": team_metrics["log_loss"],
                        }
                    )
            metrics["team_comparison"] = comparison
        if fitted_model.venue_probs is not None and "venue" in heldout:
            venues = heldout.loc[reached, "venue"].astype("string").str.strip()
            venue_team_indices = venues.str.casefold().map(venue_indices).fillna(-1).to_numpy(dtype=int)
            known_venue = venue_team_indices >= 0
            comparison = {
                "eligible_observations": int(known_venue.sum()),
                "coverage": float(known_venue.mean()),
                "baseline_brier_score": float("nan"),
                "venue_brier_score": float("nan"),
                "baseline_log_loss": float("nan"),
                "venue_log_loss": float("nan"),
            }
            if known_venue.any():
                venue_predictions = fitted_model.venue_probs[
                    venue_team_indices[known_venue], index, states[reached][known_venue]
                ]
                baseline_metrics = _binary_metrics(actual[known_venue], predicted[known_venue])
                venue_metrics = _binary_metrics(actual[known_venue], venue_predictions)
                comparison.update(
                    {
                        "baseline_brier_score": baseline_metrics["brier_score"],
                        "venue_brier_score": venue_metrics["brier_score"],
                        "baseline_log_loss": baseline_metrics["log_loss"],
                        "venue_log_loss": venue_metrics["log_loss"],
                    }
                )
            metrics["venue_comparison"] = comparison
        results[node.name] = metrics
    return results


def illegal_run_calibration(
    fitted_model: model.NodeProbs, heldout: pd.DataFrame
) -> dict[str, dict[str, float | int]]:
    """Score predicted total-run distributions for held-out wides and no-balls."""
    illegal = heldout.loc[~heldout["is_legal"].astype(bool)].copy()
    illegal["delivery_type"] = np.where(illegal["wides"] > 0, "wide", "no_ball")
    results: dict[str, dict[str, float | int]] = {}
    for delivery_type, rows in illegal.groupby("delivery_type", sort=True):
        if rows.empty:
            continue
        observed_runs = rows["total_runs"].to_numpy(dtype=int)
        state_ids = rows["state_id"].to_numpy(dtype=int)
        type_index = 0 if delivery_type == "wide" else 1
        if fitted_model.illegal_run_probs is None:
            predicted = np.ones((len(rows), 1))
        else:
            predicted = fitted_model.illegal_run_probs[type_index, state_ids]
        run_values = np.arange(1, predicted.shape[1] + 1)
        actual_indices = observed_runs - 1
        within_support = (actual_indices >= 0) & (actual_indices < predicted.shape[1])
        actual_probabilities = np.zeros(len(rows), dtype=float)
        actual_probabilities[within_support] = predicted[
            np.flatnonzero(within_support), actual_indices[within_support]
        ]
        clipped = np.clip(actual_probabilities, 1e-15, 1)
        squared_probability_sum = np.square(predicted).sum(axis=1)
        brier = squared_probability_sum + 1
        brier[within_support] -= 2 * actual_probabilities[within_support]
        results[str(delivery_type)] = {
            "observations": len(rows),
            "actual_mean_runs": float(observed_runs.mean()),
            "predicted_mean_runs": float((predicted @ run_values).mean()),
            "brier_score": float(brier.mean()),
            "log_loss": float(-np.log(clipped).mean()),
        }
    return results


def _actual_innings(heldout: pd.DataFrame) -> pd.DataFrame:
    innings = (
        heldout.groupby(["match_id", "innings_number"], observed=True)
        .agg(
            runs=("total_runs", "sum"),
            wickets=("is_wicket", "sum"),
            legal_balls=("is_legal", "sum"),
        )
        .reset_index()
    )
    counts = innings.groupby("match_id", observed=True)["innings_number"].nunique()
    complete = counts[counts == features.N_INNINGS].index
    return innings.loc[innings["match_id"].isin(complete)]


def evaluate(
    fitted_model: model.NodeProbs,
    heldout: pd.DataFrame,
    *,
    n_simulations: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate conditional outcomes and simulated score distributions on held-out data."""
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive")
    actual = _actual_innings(heldout)
    if actual.empty:
        raise ValueError("Held-out data must contain matches with both innings")

    actual_first = actual.loc[actual["innings_number"] == 1]
    actual_second = actual.loc[actual["innings_number"] == 2]
    rng = np.random.default_rng(seed)
    simulated: list[dict[str, int | float]] = []
    for _ in range(n_simulations):
        result = simulator.simulate_match(fitted_model, rng)
        simulated.append(
            {
                "first_runs": result.first_innings.final_state.runs,
                "first_wickets": result.first_innings.final_state.wickets,
                "second_runs": result.second_innings.final_state.runs,
                "second_wickets": result.second_innings.final_state.wickets,
                "second_deliveries": len(result.second_innings.deliveries),
                "winner_innings": result.winner_innings if result.winner_innings is not None else 0,
            }
        )
    simulated_frame = pd.DataFrame(simulated)
    simulated_second_deliveries = int(simulated_frame["second_deliveries"].sum())

    return {
        "heldout_matches": len(actual_first),
        "simulated_matches": n_simulations,
        "node_calibration": node_calibration(fitted_model, heldout),
        "illegal_run_calibration": illegal_run_calibration(fitted_model, heldout),
        "scores": {
            "first_innings_actual_mean": float(actual_first["runs"].mean()),
            "first_innings_simulated_mean": float(simulated_frame["first_runs"].mean()),
            "first_innings_actual_median": float(actual_first["runs"].median()),
            "first_innings_simulated_median": float(simulated_frame["first_runs"].median()),
            "second_innings_actual_runs_per_delivery": float(
                heldout.loc[heldout["innings_number"] == 2, "total_runs"].mean()
            ),
            "second_innings_simulated_runs_per_delivery": float(
                simulated_frame["second_runs"].sum() / simulated_second_deliveries
            ),
            "first_innings_actual_wickets_mean": float(actual_first["wickets"].mean()),
            "first_innings_simulated_wickets_mean": float(simulated_frame["first_wickets"].mean()),
            "second_innings_actual_mean": float(actual_second["runs"].mean()),
            "second_innings_simulated_mean": float(simulated_frame["second_runs"].mean()),
            "simulated_team1_win_rate": float((simulated_frame["winner_innings"] == 1).mean()),
            "simulated_team2_win_rate": float((simulated_frame["winner_innings"] == 2).mean()),
            "simulated_tie_rate": float((simulated_frame["winner_innings"] == 0).mean()),
        },
    }


def print_report(report: Mapping[str, Any]) -> None:
    print(
        f"Held-out matches: {report['heldout_matches']:,}; "
        f"simulated matches: {report['simulated_matches']:,}"
    )
    print("\nConditional node calibration:")
    print(f"{'Node':<12} {'N':>9} {'Observed':>10} {'Predicted':>10} {'Brier':>10} {'Log loss':>10}")
    for name, metrics in report["node_calibration"].items():
        print(
            f"{name:<12} {metrics['observations']:>9,} "
            f"{metrics['observed_rate']:>10.3f} {metrics['predicted_rate']:>10.3f} "
            f"{metrics['brier_score']:>10.4f} {metrics['log_loss']:>10.4f}"
        )
    print("\nReliability bins (predicted -> observed [deliveries]):")
    for name, metrics in report["node_calibration"].items():
        bins = metrics["reliability_bins"]
        summary = ", ".join(
            f"{item['predicted_rate']:.2f}->{item['observed_rate']:.2f} [{item['count']:,}]"
            for item in bins
        )
        print(f"  {name:<12} {summary}")
    print("\nIllegal-delivery total-run calibration:")
    for delivery_type, metrics in report["illegal_run_calibration"].items():
        print(
            f"  {delivery_type:<8} n={metrics['observations']:>7,} "
            f"actual mean={metrics['actual_mean_runs']:.3f} "
            f"predicted mean={metrics['predicted_mean_runs']:.3f} "
            f"Brier={metrics['brier_score']:.4f} "
            f"log loss={metrics['log_loss']:.4f}"
        )
    print("\nTeam strength vs baseline (same known-team deliveries):")
    print(f"{'Node':<12} {'Coverage':>10} {'Brier diff':>10} {'Log loss diff':>12}")
    for name, metrics in report["node_calibration"].items():
        comparison = metrics.get("team_comparison")
        if comparison is None:
            continue
        count = comparison["eligible_observations"]
        brier_delta = comparison["team_brier_score"] - comparison["baseline_brier_score"]
        log_loss_delta = comparison["team_log_loss"] - comparison["baseline_log_loss"]
        if count:
            print(
                f"{name:<12} {comparison['coverage']:>9.1%} "
                f"{brier_delta:>10.4f} {log_loss_delta:>12.4f}"
            )
        else:
            print(f"{name:<12} {comparison['coverage']:>9.1%} {'n/a':>10} {'n/a':>12}")
    print("\nVenue effects vs baseline (same known-venue deliveries):")
    print(f"{'Node':<12} {'Coverage':>10} {'Brier diff':>10} {'Log loss diff':>12}")
    for name, metrics in report["node_calibration"].items():
        comparison = metrics.get("venue_comparison")
        if comparison is None:
            continue
        count = comparison["eligible_observations"]
        if count:
            brier_delta = comparison["venue_brier_score"] - comparison["baseline_brier_score"]
            log_loss_delta = comparison["venue_log_loss"] - comparison["baseline_log_loss"]
            print(
                f"{name:<12} {comparison['coverage']:>9.1%} "
                f"{brier_delta:>10.4f} {log_loss_delta:>12.4f}"
            )
        else:
            print(f"{name:<12} {comparison['coverage']:>9.1%} {'n/a':>10} {'n/a':>12}")
    print("\nActual vs simulated:")
    for name, value in report["scores"].items():
        print(f"  {name}: {value:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    deliveries = pd.read_parquet(model.PROCESSED / "deliveries.parquet")
    matches = pd.read_parquet(model.PROCESSED / "matches.parquet")
    fitted_model = model.fit(deliveries, matches)
    test = heldout_deliveries(deliveries, matches)
    print_report(evaluate(fitted_model, test, n_simulations=args.simulations, seed=args.seed))


if __name__ == "__main__":
    main()
