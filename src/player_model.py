"""Leakage-safe player form profiles and pre-match team form guides."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src import features, model

PLAYER_FORM_MATCHES = 10
PLAYER_SMOOTHING_WEIGHT = 30.0
BOWLING_OVER_PRIOR = 0.25
AVERAGE_ADJUSTMENT_STRENGTH = 0.5


@dataclass
class PlayerRatings:
    player_ids: list[str]
    batting_probs: np.ndarray
    bowling_probs: np.ndarray
    debut_batting_probs: np.ndarray
    debut_bowling_probs: np.ndarray
    batting_form: dict[str, dict[str, float | int]]
    bowling_form: dict[str, dict[str, float | int]]
    as_of_date: pd.Timestamp
    batting_state_averages: dict[str, np.ndarray] = field(default_factory=dict)
    bowling_state_averages: dict[str, np.ndarray] = field(default_factory=dict)
    batting_state_dismissals: dict[str, np.ndarray] = field(default_factory=dict)
    bowling_state_wickets: dict[str, np.ndarray] = field(default_factory=dict)
    baseline_batting_averages: np.ndarray = field(default_factory=lambda: np.empty(0))
    baseline_bowling_averages: np.ndarray = field(default_factory=lambda: np.empty(0))
    bowling_usage: dict[str, dict[str, Any]] = field(default_factory=dict)

    def probabilities(self, role: str, player_id: str, state_id: int) -> np.ndarray:
        if role not in {"batting", "bowling"}:
            raise ValueError("role must be 'batting' or 'bowling'")
        if not 0 <= state_id < features.N_STATES:
            raise IndexError(f"state_id must be in [0, {features.N_STATES})")
        try:
            player_index = self.player_ids.index(str(player_id))
        except ValueError:
            prior = self.debut_batting_probs if role == "batting" else self.debut_bowling_probs
            return prior[:, state_id]
        probabilities = self.batting_probs if role == "batting" else self.bowling_probs
        return probabilities[player_index, :, state_id]

    def average_log_adjustment(self, role: str, player_id: str, state_id: int) -> float:
        """Return a shrunk, bounded log-average ratio for one player and state."""
        if role == "batting":
            averages = self.batting_state_averages
            counts = self.batting_state_dismissals
            baseline = self.baseline_batting_averages
        elif role == "bowling":
            averages = self.bowling_state_averages
            counts = self.bowling_state_wickets
            baseline = self.baseline_bowling_averages
        else:
            raise ValueError("role must be 'batting' or 'bowling'")
        if player_id not in averages or state_id >= len(baseline):
            return 0.0
        reference = float(baseline[state_id])
        if reference <= 0:
            return 0.0
        confidence = float(counts[player_id][state_id]) / (
            float(counts[player_id][state_id]) + PLAYER_SMOOTHING_WEIGHT
        )
        log_ratio = np.log(max(float(averages[player_id][state_id]), 1e-9) / reference)
        return float(
            AVERAGE_ADJUSTMENT_STRENGTH
            * confidence
            * np.clip(log_ratio, -np.log(2), np.log(2))
        )


def _role_nodes(nodes: list[model.Node], role: str) -> set[str]:
    return model.BATTING_NODES if role == "batting" else model.BOWLING_NODES


def _player_match_rows(
    deliveries: pd.DataFrame,
    player_column: str,
) -> pd.DataFrame:
    return deliveries.loc[deliveries[player_column].notna()].copy()


def _counts_for_rows(
    rows: pd.DataFrame,
    nodes: list[model.Node],
) -> tuple[np.ndarray, np.ndarray]:
    yes_counts, reach_counts = model.count_nodes(rows, nodes)
    return yes_counts.astype(float), reach_counts.astype(float)


def debut_player_priors(
    training: pd.DataFrame,
    baseline_probs: np.ndarray,
    nodes: list[model.Node],
) -> tuple[np.ndarray, np.ndarray]:
    """Average each player's first-training-match rates, role by role."""
    priors: dict[str, np.ndarray] = {}
    for role, player_column in (
        ("batting", "batter_id"),
        ("bowling", "bowler_id"),
    ):
        role_prior = baseline_probs.copy()
        if player_column not in training or "_match_date" not in training:
            priors[role] = role_prior
            continue

        rows = _player_match_rows(training, player_column)
        if rows.empty:
            priors[role] = role_prior
            continue

        prior_sums = np.zeros_like(baseline_probs, dtype=float)
        prior_counts = np.zeros_like(baseline_probs, dtype=float)
        for _, player_rows in rows.groupby(player_column, sort=False, observed=True):
            first_date = player_rows["_match_date"].min()
            first_ids = player_rows.loc[
                player_rows["_match_date"] == first_date, "match_id"
            ].astype(str)
            first_match_id = first_ids.min()
            first_match = player_rows.loc[
                player_rows["match_id"].astype(str) == first_match_id
            ]
            yes, reach = _counts_for_rows(first_match, nodes)
            smoothed = (yes + PLAYER_SMOOTHING_WEIGHT * baseline_probs) / (
                reach + PLAYER_SMOOTHING_WEIGHT
            )
            role_nodes = _role_nodes(nodes, role)
            for node_index, node in enumerate(nodes):
                if node.name not in role_nodes:
                    continue
                observed = reach[node_index] > 0
                prior_sums[node_index, observed] += smoothed[node_index, observed]
                prior_counts[node_index, observed] += 1

        observed = prior_counts > 0
        role_prior[observed] = prior_sums[observed] / prior_counts[observed]
        priors[role] = role_prior
    return priors["batting"], priors["bowling"]


def build_player_ratings(
    deliveries: pd.DataFrame,
    matches: pd.DataFrame,
    fitted_model: model.NodeProbs,
    *,
    as_of_date: str | pd.Timestamp,
    form_matches: int = PLAYER_FORM_MATCHES,
) -> PlayerRatings:
    """Build last-N player profiles using only matches strictly before as_of_date."""
    if form_matches <= 0:
        raise ValueError("form_matches must be positive")
    if "date" not in matches or "match_id" not in matches:
        raise ValueError("matches must contain 'date' and 'match_id' columns")
    cutoff = pd.Timestamp(as_of_date)
    match_dates = pd.to_datetime(matches["date"], errors="raise")
    dates_by_id = pd.Series(match_dates.to_numpy(), index=matches["match_id"].to_numpy())
    eligible_ids = matches.loc[match_dates < cutoff, "match_id"]
    rows = deliveries.loc[deliveries["match_id"].isin(eligible_ids)].copy()
    if "is_super_over" in rows:
        rows = rows.loc[~rows["is_super_over"].astype(bool)]
    rows = rows.loc[rows["innings_number"].between(1, features.N_INNINGS)].copy()
    rows["_match_date"] = rows["match_id"].map(dates_by_id)
    rows = features.add_state_features(rows)
    baseline_batting_averages = _league_state_average(rows, "batting")
    baseline_bowling_averages = _league_state_average(rows, "bowling")

    nodes = model.build_tree()
    names: set[str] = set()
    role_profiles: dict[str, dict[str, np.ndarray]] = {"batting": {}, "bowling": {}}
    role_forms: dict[str, dict[str, dict[str, float | int]]] = {"batting": {}, "bowling": {}}
    role_averages: dict[str, dict[str, np.ndarray]] = {"batting": {}, "bowling": {}}
    role_average_counts: dict[str, dict[str, np.ndarray]] = {"batting": {}, "bowling": {}}
    bowling_usage: dict[str, dict[str, Any]] = {}
    role_priors = {
        "batting": fitted_model.debut_batting_probs
        if fitted_model.debut_batting_probs is not None
        else fitted_model.probs,
        "bowling": fitted_model.debut_bowling_probs
        if fitted_model.debut_bowling_probs is not None
        else fitted_model.probs,
    }

    for role, player_column in (
        ("batting", "batter_id"),
        ("bowling", "bowler_id"),
    ):
        if player_column not in rows:
            continue
        role_rows = _player_match_rows(rows, player_column)
        for player_id, player_rows in role_rows.groupby(
            player_column, sort=False, observed=True
        ):
            player_id = str(player_id)
            names.add(player_id)
            appearances = (
                player_rows[["match_id", "_match_date"]]
                .drop_duplicates()
                .sort_values(["_match_date", "match_id"])
                .tail(form_matches)
            )
            selected = player_rows.loc[
                player_rows["match_id"].isin(appearances["match_id"])
            ]
            yes, reach = _counts_for_rows(selected, nodes)
            prior = role_priors[role]
            profile = (yes + PLAYER_SMOOTHING_WEIGHT * prior) / (
                reach + PLAYER_SMOOTHING_WEIGHT
            )
            for node_index, node in enumerate(nodes):
                if node.name not in _role_nodes(nodes, role):
                    profile[node_index] = fitted_model.probs[node_index]
            role_profiles[role][player_id] = profile
            role_forms[role][player_id] = _form_summary(selected, role)
            baseline = (
                baseline_batting_averages
                if role == "batting"
                else baseline_bowling_averages
            )
            averages, counts = _state_averages(selected, role, baseline)
            role_averages[role][player_id] = averages
            role_average_counts[role][player_id] = counts
            if role == "bowling":
                bowling_usage[player_id] = _bowling_usage(selected)

    player_ids = sorted(names)
    batting_profiles = np.stack(
        [
            role_profiles["batting"].get(player_id, role_priors["batting"])
            for player_id in player_ids
        ]
    ) if player_ids else np.empty((0, len(nodes), features.N_STATES))
    bowling_profiles = np.stack(
        [
            role_profiles["bowling"].get(player_id, role_priors["bowling"])
            for player_id in player_ids
        ]
    ) if player_ids else np.empty((0, len(nodes), features.N_STATES))
    batting_form = role_forms["batting"]
    bowling_form = role_forms["bowling"]
    return PlayerRatings(
        player_ids=player_ids,
        batting_probs=batting_profiles,
        bowling_probs=bowling_profiles,
        debut_batting_probs=role_priors["batting"],
        debut_bowling_probs=role_priors["bowling"],
        batting_form=batting_form,
        bowling_form=bowling_form,
        as_of_date=cutoff,
        batting_state_averages=role_averages["batting"],
        bowling_state_averages=role_averages["bowling"],
        batting_state_dismissals=role_average_counts["batting"],
        bowling_state_wickets=role_average_counts["bowling"],
        baseline_batting_averages=baseline_batting_averages,
        baseline_bowling_averages=baseline_bowling_averages,
        bowling_usage=bowling_usage,
    )


def _state_averages(
    player_rows: pd.DataFrame,
    role: str,
    baseline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    player_runs, player_dismissals = _state_totals(player_rows, role)
    smoothed = (
        player_runs + PLAYER_SMOOTHING_WEIGHT * baseline
    ) / (player_dismissals + PLAYER_SMOOTHING_WEIGHT)
    return smoothed, player_dismissals


def _league_state_average(rows: pd.DataFrame, role: str) -> np.ndarray:
    league_runs, league_dismissals = _state_totals(rows, role)
    overall_average = (
        float(league_runs.sum() / league_dismissals.sum())
        if league_dismissals.sum()
        else 25.0
    )
    return np.divide(
        league_runs,
        league_dismissals,
        out=np.full(features.N_STATES, overall_average, dtype=float),
        where=league_dismissals > 0,
    )


def _state_totals(rows: pd.DataFrame, role: str) -> tuple[np.ndarray, np.ndarray]:
    if role == "batting":
        state_runs = "batter_runs"
        if "player_out" in rows and "batter" in rows:
            dismissals = (
                rows["is_wicket"].astype(bool)
                & rows["player_out"].notna()
                & rows["player_out"].astype(str).eq(rows["batter"].astype(str))
            )
        else:
            dismissals = rows["is_wicket"].astype(bool)
    else:
        credited_kinds = {
            "bowled",
            "caught",
            "caught and bowled",
            "lbw",
            "stumped",
            "hit wicket",
        }
        dismissals = rows["wicket_kind"].isin(credited_kinds)

    if role == "bowling":
        conceded = (
            rows["total_runs"]
            - rows["byes"]
            - rows["legbyes"]
            - rows.get("penalty", 0)
        ).clip(lower=0)
        runs = conceded.to_numpy(dtype=float)
    else:
        runs = rows[state_runs].to_numpy(dtype=float)
    states = rows["state_id"].to_numpy(dtype=int)
    return (
        np.bincount(states, weights=runs, minlength=features.N_STATES),
        np.bincount(
            states,
            weights=dismissals.to_numpy(dtype=float),
            minlength=features.N_STATES,
        ),
    )


def _bowling_usage(rows: pd.DataFrame) -> dict[str, Any]:
    overs_by_match = rows.groupby("match_id", observed=True)["over"].nunique()
    bowled_overs = rows.drop_duplicates(["match_id", "innings_number", "over"])
    over_counts = np.bincount(
        bowled_overs["over"].astype(int).to_numpy(),
        minlength=20,
    )[:20].astype(float)
    probabilities = (over_counts + BOWLING_OVER_PRIOR) / (
        over_counts.sum() + BOWLING_OVER_PRIOR * 20
    )
    preferred = np.flatnonzero(over_counts == over_counts.max()).astype(int).tolist()
    return {
        "matches": int(overs_by_match.size),
        "avg_overs": float(overs_by_match.mean()) if not overs_by_match.empty else 0.0,
        "over_probabilities": probabilities,
        "preferred_overs": preferred,
    }


def build_bowling_schedule(
    candidate_bowler_ids: list[str] | tuple[str, ...],
    ratings: PlayerRatings,
) -> tuple[str, ...]:
    """Select historical bowling options and assign a leak-safe 20-over schedule."""
    if len(candidate_bowler_ids) < 5:
        raise ValueError("At least five bowling candidates are required")
    if len(set(candidate_bowler_ids)) != len(candidate_bowler_ids):
        raise ValueError("Bowling candidates must be unique")

    candidates = list(candidate_bowler_ids)
    average_overs = {
        player_id: float(
            ratings.bowling_usage.get(player_id, {}).get("avg_overs", 0.0)
        )
        for player_id in candidates
    }
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (-average_overs[item[1]], item[0]),
    )
    known = [player_id for player_id in candidates if average_overs[player_id] > 0]
    if len(known) < 5:
        known_set = set(known)
        known.extend(
            player_id
            for _, player_id in ranked
            if player_id not in known_set
        )
        known = known[:5]
    bowlers = known
    weights = np.array(
        [
            max(average_overs[player_id], 1.0)
            for player_id in bowlers
        ],
        dtype=float,
    )
    quotas = np.zeros(len(bowlers), dtype=int)
    for _ in range(20):
        eligible = np.flatnonzero(quotas < 4)
        chosen = min(
            eligible,
            key=lambda index: (quotas[index] / weights[index], index),
        )
        quotas[chosen] += 1

    preferences = np.stack(
        [
            np.asarray(
                ratings.bowling_usage.get(player_id, {}).get(
                    "over_probabilities", np.full(20, 1 / 20)
                ),
                dtype=float,
            )
            for player_id in bowlers
        ]
    )
    remaining = quotas.copy()
    schedule: list[str] = []
    previous = -1
    for over_number in range(20):
        slots_after = 19 - over_number
        eligible = []
        for index in range(len(bowlers)):
            if remaining[index] <= 0 or index == previous:
                continue
            remaining_after = remaining.copy()
            remaining_after[index] -= 1
            if slots_after and remaining_after.max() > (slots_after + 1) // 2:
                continue
            eligible.append(index)
        if not eligible:
            eligible = [
                index for index in range(len(bowlers)) if remaining[index] > 0
            ]
        chosen = max(
            eligible,
            key=lambda index: (
                preferences[index, over_number] * remaining[index],
                remaining[index] / weights[index],
                -index,
            ),
        )
        schedule.append(bowlers[chosen])
        remaining[chosen] -= 1
        previous = chosen
    return tuple(schedule)


def _form_summary(rows: pd.DataFrame, role: str) -> dict[str, float | int]:
    if role == "batting":
        balls_faced = (~rows["wides"].astype(bool)).sum()
        runs = rows["batter_runs"].sum()
        return {
            "matches": int(rows["match_id"].nunique()),
            "runs": int(runs),
            "balls_faced": int(balls_faced),
            "strike_rate": float(100 * runs / balls_faced) if balls_faced else 0.0,
            "boundaries": int(rows["batter_runs"].isin([4, 6]).sum()),
        }
    legal_balls = rows["is_legal"].astype(bool).sum()
    runs_conceded = (
        rows["total_runs"] - rows["byes"] - rows["legbyes"]
    ).clip(lower=0).sum()
    credited_kinds = {"bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"}
    wickets = rows["wicket_kind"].isin(credited_kinds).sum()
    return {
        "matches": int(rows["match_id"].nunique()),
        "legal_balls": int(legal_balls),
        "runs_conceded": int(runs_conceded),
        "wickets": int(wickets),
        "economy": float(runs_conceded * 6 / legal_balls) if legal_balls else 0.0,
    }


def team_form_guide(
    deliveries: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    as_of_date: str | pd.Timestamp,
    form_matches: int = PLAYER_FORM_MATCHES,
) -> pd.DataFrame:
    """Summarize teams' last-N completed matches strictly before as_of_date."""
    cutoff = pd.Timestamp(as_of_date)
    dated_matches = matches.copy()
    dated_matches["date"] = pd.to_datetime(dated_matches["date"], errors="raise")
    dated_matches = dated_matches.loc[dated_matches["date"] < cutoff]
    history = deliveries.loc[deliveries["match_id"].isin(dated_matches["match_id"])]
    innings = (
        history.groupby(["match_id", "innings_number", "batting_team"], observed=True)
        .agg(
            runs=("total_runs", "sum"),
            wickets=("is_wicket", "sum"),
            legal_balls=("is_legal", "sum"),
        )
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for _, match in dated_matches.iterrows():
        match_innings = innings.loc[innings["match_id"] == match["match_id"]]
        for team in (str(match["team_1"]), str(match["team_2"])):
            batting = match_innings.loc[match_innings["batting_team"].astype(str) == team]
            bowling = match_innings.loc[match_innings["batting_team"].astype(str) != team]
            if batting.empty or bowling.empty:
                continue
            rows.append(
                {
                    "team": team,
                    "match_id": match["match_id"],
                    "date": match["date"],
                    "runs_scored": int(batting.iloc[0]["runs"]),
                    "wickets_lost": int(batting.iloc[0]["wickets"]),
                    "runs_conceded": int(bowling.iloc[0]["runs"]),
                    "wickets_taken": int(bowling.iloc[0]["wickets"]),
                    "won": bool(match["winner"] == team),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "team",
                "matches",
                "wins",
                "win_rate",
                "avg_runs_scored",
                "avg_wickets_lost",
                "avg_runs_conceded",
                "avg_wickets_taken",
            ]
        )
    frame = pd.DataFrame(rows).sort_values(["team", "date", "match_id"])
    recent = frame.groupby("team", sort=False).tail(form_matches)
    guide = (
        recent.groupby("team", sort=True)
        .agg(
            matches=("match_id", "nunique"),
            wins=("won", "sum"),
            avg_runs_scored=("runs_scored", "mean"),
            avg_wickets_lost=("wickets_lost", "mean"),
            avg_runs_conceded=("runs_conceded", "mean"),
            avg_wickets_taken=("wickets_taken", "mean"),
        )
        .reset_index()
    )
    guide["win_rate"] = guide["wins"] / guide["matches"]
    return guide
