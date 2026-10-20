"""Estimate the probability tables that the simulator sameples from.

the simulator will walk down a decision tree on every ball:

wicket?
    no -> legal?
        no -> wide/noball?
        yes -> runs?
            no -> dot
            yes -> boundary?
                yes -> six?
                no -> 1?
                    no -> 2?
                    
Each node is a yes/no quesiton with its own probability per match state.
This module estimates those probabilities from historical deliveries:
    
    for each noode:
        take the training deliveries that reach the node
        P(yes | state) = yes_count / reach_count, smoothed for thin states
    

Reads: data/processd/deliveries.parquet, matches.parquet
Writes: results/models/node_probs.npz (+metadata)

The simulator should nly ever call get_node_probs(); never looks inside so that the lookup tables can later be swapped for regression without touching the simulator
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import features

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "results" / "models"

TRAIN_CUTOFF_DATE = "2025-01-01"
SMOOTHING_WEIGHT = 30
VENUE_SMOOTHING_WEIGHT = 100
BATTING_NODES = {"runs", "boundary", "six", "one", "two"}
BOWLING_NODES = {"wicket", "legal", "wide"}


@dataclass(frozen=True)

class Node:
    name: str
    reach: Callable[[pd.DataFrame],pd.Series]
    yes: Callable[[pd.DataFrame], pd.Series]


def build_tree() -> list[Node]:
    """Return the Nodes in the order the simulator walks them"""

    def child(parent: Node, name: str, branch: bool, yes) -> Node:
        if branch:
            reach = lambda d: parent.reach(d) & parent.yes(d)
        else:
            reach = lambda d:parent.reach(d) & ~parent.yes(d)
        return Node(name=name, reach=reach, yes=yes)

    wicket = Node(
        name= "wicket",
        reach = lambda d: pd.Series(True, index=d.index),
        yes = lambda d: d["is_wicket"]
    )

    legal = child(wicket, "legal", branch=False, yes=lambda d: d["is_legal"])

    wide = child(legal, "wide", branch=False, yes=lambda d: d["wides"] > 0 )

    runs = child(legal, "runs", branch=True, yes=lambda d: d["total_runs"] > 0)

    boundary = child(runs, "boundary", branch = True, yes =lambda d: d["total_runs"] >= 4)

    six = child(boundary, "six", branch = True, yes = lambda d: d["total_runs"] == 6)

    one = child(boundary, "one", branch =False, yes=lambda d: d["total_runs"] == 1)

    two = child(one, "two", branch = False, yes=lambda d: d["total_runs"] == 2)

    return [wicket, legal, wide, runs, boundary, six, one, two]

def training_deliveries(
    deliveries: pd.DataFrame, matches: pd.DataFrame
) -> pd.DataFrame:
    if "date" not in matches or "match_id" not in matches:
        raise ValueError("matches must contain 'date' and 'match_id' columns")
    if "match_id" not in deliveries:
        raise ValueError("deliveries must contain a 'match_id' column")

    cutoff = pd.Timestamp(TRAIN_CUTOFF_DATE)
    match_dates = pd.to_datetime(matches["date"], errors="raise")
    train_ids = matches.loc[match_dates < cutoff, "match_id"]
    return deliveries.loc[deliveries["match_id"].isin(train_ids)].copy()


def count_nodes(df: pd.DataFrame, nodes: list[Node]) -> tuple[np.ndarray, np.ndarray]:
    if "state_id" not in df:
        raise ValueError("deliveries must have a 'state_id' column")
    state_ids = df["state_id"].to_numpy()
    if (
        not np.issubdtype(state_ids.dtype, np.integer)
        or np.any(state_ids < 0)
        or np.any(state_ids >= features.N_STATES)
    ):
        raise ValueError(f"state_id values must be integers in [0, {features.N_STATES})")

    yes_counts = np.zeros((len(nodes), features.N_STATES), dtype=np.int64)
    reach_counts = np.zeros((len(nodes), features.N_STATES), dtype=np.int64)
    for i, node in enumerate(nodes):
        reached = node.reach(df).fillna(False).astype(bool)
        said_yes = reached & node.yes(df).fillna(False).astype(bool)
        reach_counts[i] = np.bincount(state_ids[reached.to_numpy()], minlength=features.N_STATES)
        yes_counts[i] = np.bincount(state_ids[said_yes.to_numpy()], minlength=features.N_STATES)
    return yes_counts, reach_counts


def smooth(yes: np.ndarray, reach: np.ndarray, weight: float = SMOOTHING_WEIGHT) -> np.ndarray:
    """Return probabilities shaped (n_nodes, N_STATES).
 
    Blend each state's raw rate with a broader rate for the same node
    (for example the node's rate across that phase and innings):
 
        p = (yes + weight * broader_rate) / (reach + weight)
 
    The broader rate pools wicket buckets within the same innings and phase.
    If that pool has no observations, use the node's overall rate; a node with
    no observations at all falls back to 0.5.
    """
    yes = np.asarray(yes, dtype=float)
    reach = np.asarray(reach, dtype=float)
    if yes.ndim != 2 or reach.shape != yes.shape or yes.shape[1] != features.N_STATES:
        raise ValueError(f"yes and reach must have matching shape (n_nodes, {features.N_STATES})")
    if not np.isfinite(yes).all() or not np.isfinite(reach).all():
        raise ValueError("yes and reach counts must be finite")
    if np.any(yes < 0) or np.any(reach < 0) or np.any(yes > reach):
        raise ValueError("counts must satisfy 0 <= yes <= reach")
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("smoothing weight must be a finite non-negative number")

    probs = np.empty_like(yes)
    states_per_phase = features.N_WICKET_BUCKETS
    states_per_innings = features.N_PHASES * states_per_phase
    for node_idx in range(yes.shape[0]):
        total_reach = reach[node_idx].sum()
        global_rate = yes[node_idx].sum() / total_reach if total_reach else 0.5
        for innings in range(features.N_INNINGS):
            for phase in range(features.N_PHASES):
                start = innings * states_per_innings + phase * states_per_phase
                end = start + states_per_phase
                phase_reach = reach[node_idx, start:end].sum()
                broader_rate = (
                    yes[node_idx, start:end].sum() / phase_reach
                    if phase_reach
                    else global_rate
                )
                denominator = reach[node_idx, start:end] + weight
                if weight == 0:
                    probs[node_idx, start:end] = np.divide(
                        yes[node_idx, start:end],
                        reach[node_idx, start:end],
                        out=np.full(states_per_phase, broader_rate),
                        where=reach[node_idx, start:end] > 0,
                    )
                else:
                    probs[node_idx, start:end] = (
                        yes[node_idx, start:end] + weight * broader_rate
                    ) / denominator
    return np.clip(probs, 0.0, 1.0)


def _smooth_team_counts(
    yes: np.ndarray,
    reach: np.ndarray,
    baseline_probs: np.ndarray,
    weight: float = SMOOTHING_WEIGHT,
) -> np.ndarray:
    """Shrink team/state rates toward the league probability for that state."""
    return (yes + weight * baseline_probs) / (reach + weight)
 
 
# =============================================================================
# 6. THE MODEL OBJECT AND THE INTERFACE THE SIMULATOR USES
# =============================================================================
@dataclass
class NodeProbs:
    probs: np.ndarray          # shape (n_nodes, N_STATES)
    node_names: list[str]
    metadata: dict             # bucket settings, cutoff date, data snapshot, smoothing weight
    team_names: list[str] = field(default_factory=list)
    batting_team_probs: np.ndarray | None = None
    bowling_team_probs: np.ndarray | None = None
    illegal_run_probs: np.ndarray | None = None
    venue_names: list[str] = field(default_factory=list)
    venue_probs: np.ndarray | None = None
    debut_batting_probs: np.ndarray | None = None
    debut_bowling_probs: np.ndarray | None = None


def _fit_illegal_run_probs(train: pd.DataFrame) -> np.ndarray:
    """Fit P(total runs | wide/no-ball, match state), shrunk to type-wide rates."""
    illegal = train.loc[~train["is_legal"].astype(bool)]
    total_runs = illegal["total_runs"].to_numpy(dtype=float)
    if (
        not np.isfinite(total_runs).all()
        or np.any(total_runs < 1)
        or np.any(total_runs != np.floor(total_runs))
    ):
        raise ValueError("Illegal deliveries must have positive integer total_runs")

    run_values = np.arange(1, int(total_runs.max()) + 1) if len(total_runs) else np.array([1])
    counts = np.zeros((2, features.N_STATES, len(run_values)), dtype=float)
    state_ids = illegal["state_id"].to_numpy(dtype=int)
    run_indices = total_runs.astype(int) - 1
    wide = illegal["wides"].to_numpy(dtype=float) > 0
    type_indices = np.where(wide, 0, 1)
    np.add.at(counts, (type_indices, state_ids, run_indices), 1)

    probabilities = np.empty_like(counts)
    for type_index in range(2):
        global_counts = counts[type_index].sum(axis=0)
        global_probs = (
            global_counts / global_counts.sum()
            if global_counts.sum()
            else np.eye(1, len(run_values), 0, dtype=float)[0]
        )
        for state_id in range(features.N_STATES):
            state_counts = counts[type_index, state_id]
            probabilities[type_index, state_id] = (
                state_counts + SMOOTHING_WEIGHT * global_probs
            ) / (state_counts.sum() + SMOOTHING_WEIGHT)
    return probabilities
 
 
def fit(deliveries: pd.DataFrame, matches: pd.DataFrame) -> NodeProbs:
    """Fit smoothed node probabilities using matches before the training cutoff."""
    train = training_deliveries(deliveries, matches)
    if "is_super_over" in train:
        train = train.loc[~train["is_super_over"].astype(bool)]
    if "innings_number" not in train:
        raise ValueError("deliveries must contain an 'innings_number' column")
    train = train.loc[train["innings_number"].between(1, features.N_INNINGS)].copy()
    if train.empty:
        raise ValueError(f"No training deliveries found before {TRAIN_CUTOFF_DATE}")

    match_dates = pd.to_datetime(matches["date"], errors="raise")
    date_lookup = pd.Series(match_dates.to_numpy(), index=matches["match_id"].to_numpy())
    train["_match_date"] = train["match_id"].map(date_lookup)
    venue_names: list[str] = []
    venue_probs = None
    if "venue" in matches:
        match_venues = matches[["match_id", "venue"]].drop_duplicates("match_id")
        venue_lookup = match_venues.set_index("match_id")["venue"]
        train["venue"] = train["match_id"].map(venue_lookup).astype("string").str.strip()
        venue_names = sorted(
            set(train["venue"].dropna().loc[lambda values: values != ""])
        )

    train = features.add_state_features(train)
    illegal_run_probs = _fit_illegal_run_probs(train)
    nodes = build_tree()
    yes_counts, reach_counts = count_nodes(train, nodes)
    probs = smooth(yes_counts, reach_counts)
    from src.player_model import debut_player_priors

    debut_batting_probs, debut_bowling_probs = debut_player_priors(train, probs, nodes)
    if venue_names:
        venue_probs = np.tile(probs, (len(venue_names), 1, 1))
        venue_indices = {venue: index for index, venue in enumerate(venue_names)}
        venue_rows = train.loc[train["venue"].isin(venue_names)]
        for venue, venue_deliveries in venue_rows.groupby("venue", sort=False):
            venue_yes, venue_reach = count_nodes(venue_deliveries, nodes)
            venue_probs[venue_indices[str(venue)]] = _smooth_team_counts(
                venue_yes,
                venue_reach,
                probs,
                weight=VENUE_SMOOTHING_WEIGHT,
            )

    team_columns = {"batting_team", "bowling_team"}
    present_team_columns = team_columns.intersection(train.columns)
    if present_team_columns and present_team_columns != team_columns:
        raise ValueError("deliveries must contain both 'batting_team' and 'bowling_team' columns")

    team_names: list[str] = []
    batting_team_probs = None
    bowling_team_probs = None
    if present_team_columns:
        batting_teams = train["batting_team"].astype("string").str.strip()
        bowling_teams = train["bowling_team"].astype("string").str.strip()
        team_names = sorted(
            set(batting_teams.dropna().loc[lambda values: values != ""])
            | set(bowling_teams.dropna().loc[lambda values: values != ""])
        )
        batting_team_probs = np.tile(probs, (len(team_names), 1, 1))
        bowling_team_probs = np.tile(probs, (len(team_names), 1, 1))
        node_indices = {
            "batting": [i for i, node in enumerate(nodes) if node.name in BATTING_NODES],
            "bowling": [i for i, node in enumerate(nodes) if node.name in BOWLING_NODES],
        }
        team_indices = {team: index for index, team in enumerate(team_names)}
        for column, team_probs, role in (
            ("batting_team", batting_team_probs, "batting"),
            ("bowling_team", bowling_team_probs, "bowling"),
        ):
            grouped = train.assign(
                _team_name=train[column].astype("string").str.strip()
            )
            grouped = grouped.loc[grouped["_team_name"].notna() & grouped["_team_name"].ne("")]
            for team, team_rows in grouped.groupby("_team_name", sort=False):
                team_yes, team_reach = count_nodes(team_rows, nodes)
                team_index = team_indices[str(team)]
                team_probs[team_index] = _smooth_team_counts(
                    team_yes, team_reach, probs
                )
                unused_nodes = set(range(len(nodes))) - set(node_indices[role])
                team_probs[team_index, list(unused_nodes)] = probs[list(unused_nodes)]

    metadata: dict[str, Any] = {
        "PHASE_STARTS": list(features.PHASE_STARTS),
        "N_PHASES": features.N_PHASES,
        "N_WICKET_BUCKETS": features.N_WICKET_BUCKETS,
        "N_INNINGS": features.N_INNINGS,
        "N_STATES": features.N_STATES,
        "TRAIN_CUTOFF_DATE": TRAIN_CUTOFF_DATE,
        "SMOOTHING_WEIGHT": SMOOTHING_WEIGHT,
        "VENUE_SMOOTHING_WEIGHT": VENUE_SMOOTHING_WEIGHT,
        "N_TEAMS": len(team_names),
        "N_VENUES": len(venue_names),
    }
    return NodeProbs(
        probs=probs,
        node_names=[node.name for node in nodes],
        metadata=metadata,
        team_names=team_names,
        batting_team_probs=batting_team_probs,
        bowling_team_probs=bowling_team_probs,
        illegal_run_probs=illegal_run_probs,
        venue_names=venue_names,
        venue_probs=venue_probs,
        debut_batting_probs=debut_batting_probs,
        debut_bowling_probs=debut_bowling_probs,
    )


def get_node_probs(
    model: NodeProbs,
    state_id: int,
    *,
    batting_team: str | None = None,
    bowling_team: str | None = None,
    venue: str | None = None,
    batting_player_id: str | None = None,
    bowling_player_id: str | None = None,
    player_ratings: Any | None = None,
) -> np.ndarray:
    """The ONLY function the simulator calls: one probability per node for this state."""
    if not 0 <= state_id < model.probs.shape[1]:
        raise IndexError(f"state_id must be in [0, {model.probs.shape[1]})")
    team_probabilities = model.probs[:, state_id].astype(float, copy=True)
    for team, team_probs, relevant_nodes, role in (
        (batting_team, model.batting_team_probs, BATTING_NODES, "batting"),
        (bowling_team, model.bowling_team_probs, BOWLING_NODES, "bowling"),
    ):
        if team is None:
            continue
        if not isinstance(team, str) or not team.strip():
            raise ValueError(f"{role}_team must be a non-empty string or None")
        if team_probs is None:
            raise ValueError("This model was not fitted with team data")
        try:
            team_index = model.team_names.index(team.strip())
        except ValueError as exc:
            raise ValueError(f"Unknown {role} team: {team!r}") from exc
        for node_index, node_name in enumerate(model.node_names):
            if node_name in relevant_nodes:
                team_probabilities[node_index] = team_probs[team_index, node_index, state_id]
    if venue is not None:
        if not isinstance(venue, str) or not venue.strip():
            raise ValueError("venue must be a non-empty string or None")
        if model.venue_probs is None:
            raise ValueError("This model was not fitted with venue data")
        venue_lookup = {name.casefold(): index for index, name in enumerate(model.venue_names)}
        try:
            venue_index = venue_lookup[venue.strip().casefold()]
        except KeyError as exc:
            raise ValueError(f"Unknown venue: {venue!r}") from exc
        baseline = np.clip(model.probs[:, state_id], 1e-9, 1 - 1e-9)
        venue_profile = np.clip(
            model.venue_probs[venue_index, :, state_id], 1e-9, 1 - 1e-9
        )
        current = np.clip(team_probabilities, 1e-9, 1 - 1e-9)
        logit = lambda values: np.log(values / (1 - values))
        venue_probabilities = 1 / (
            1 + np.exp(-(logit(current) + logit(venue_profile) - logit(baseline)))
        )
        team_probabilities = venue_probabilities
    for player_id, role, node_names in (
        (batting_player_id, "batting", BATTING_NODES),
        (bowling_player_id, "bowling", BOWLING_NODES),
    ):
        if player_id is None:
            continue
        if player_ratings is None:
            raise ValueError("Player IDs were supplied without player ratings")
        player_probs = player_ratings.probabilities(role, player_id, state_id)
        for node_index, node_name in enumerate(model.node_names):
            if node_name in node_names:
                baseline = np.clip(model.probs[node_index, state_id], 1e-9, 1 - 1e-9)
                current = np.clip(team_probabilities[node_index], 1e-9, 1 - 1e-9)
                player_probability = np.clip(player_probs[node_index], 1e-9, 1 - 1e-9)
                baseline_logit = np.log(baseline / (1 - baseline))
                current_logit = np.log(current / (1 - current))
                player_logit = np.log(player_probability / (1 - player_probability))
                combined_logit = current_logit + player_logit - baseline_logit
                team_probabilities[node_index] = 1 / (1 + np.exp(-combined_logit))
    if player_ratings is not None:
        average_adjustment = 0.0
        if batting_player_id is not None:
            average_adjustment += player_ratings.average_log_adjustment(
                "batting", batting_player_id, state_id
            )
        if bowling_player_id is not None:
            average_adjustment += player_ratings.average_log_adjustment(
                "bowling", bowling_player_id, state_id
            )
        for node_index, node_name in enumerate(model.node_names):
            if node_name not in {"runs", "wicket"}:
                continue
            direction = 1 if node_name == "runs" else -1
            probability = np.clip(team_probabilities[node_index], 1e-9, 1 - 1e-9)
            current_logit = np.log(probability / (1 - probability))
            team_probabilities[node_index] = 1 / (
                1 + np.exp(-(current_logit + direction * average_adjustment))
            )
    return team_probabilities


def sample_illegal_runs(
    model: NodeProbs,
    state_id: int,
    delivery_type: str,
    rng: np.random.Generator,
) -> int:
    """Sample total delivery runs for a wide or no-ball."""
    if delivery_type not in {"wide", "no_ball"}:
        raise ValueError("delivery_type must be 'wide' or 'no_ball'")
    if not 0 <= state_id < model.probs.shape[1]:
        raise IndexError(f"state_id must be in [0, {model.probs.shape[1]})")
    if model.illegal_run_probs is None:
        return 1
    type_index = 0 if delivery_type == "wide" else 1
    probabilities = model.illegal_run_probs[type_index, state_id]
    return int(rng.choice(np.arange(1, len(probabilities) + 1), p=probabilities))
 
 
# =============================================================================
# 7. SAVE / LOAD (with a safety check on the state definition)
# =============================================================================
def save(model: NodeProbs, path: Path = MODEL_DIR / "node_probs.npz") -> None:
    """Save model arrays and JSON metadata without requiring pickle."""
    path = Path(path)
    if model.probs.shape != (len(model.node_names), features.N_STATES):
        raise ValueError("model probabilities must have shape (len(node_names), N_STATES)")
    if not np.isfinite(model.probs).all() or np.any((model.probs < 0) | (model.probs > 1)):
        raise ValueError("model probabilities must be finite and between 0 and 1")
    if (model.batting_team_probs is None) != (model.bowling_team_probs is None):
        raise ValueError("batting and bowling team probability tables must both be present or absent")
    team_arrays: dict[str, np.ndarray] = {}
    if model.venue_probs is not None:
        expected_shape = (len(model.venue_names), len(model.node_names), features.N_STATES)
        if (
            model.venue_probs.shape != expected_shape
            or not np.isfinite(model.venue_probs).all()
            or np.any((model.venue_probs < 0) | (model.venue_probs > 1))
        ):
            raise ValueError(f"venue_probs must have shape {expected_shape} and valid probabilities")
        team_arrays["venue_names"] = np.asarray(model.venue_names, dtype=str)
        team_arrays["venue_probs"] = model.venue_probs
    if (model.debut_batting_probs is None) != (model.debut_bowling_probs is None):
        raise ValueError("batting and bowling debut priors must both be present or absent")
    if model.debut_batting_probs is not None and model.debut_bowling_probs is not None:
        expected_shape = (len(model.node_names), features.N_STATES)
        for name, probs in (
            ("debut_batting_probs", model.debut_batting_probs),
            ("debut_bowling_probs", model.debut_bowling_probs),
        ):
            if (
                probs.shape != expected_shape
                or not np.isfinite(probs).all()
                or np.any((probs < 0) | (probs > 1))
            ):
                raise ValueError(f"{name} must have shape {expected_shape} and valid probabilities")
            team_arrays[name] = probs
    if model.illegal_run_probs is not None:
        if (
            model.illegal_run_probs.ndim != 3
            or model.illegal_run_probs.shape[:2] != (2, features.N_STATES)
            or model.illegal_run_probs.shape[2] < 1
            or not np.isfinite(model.illegal_run_probs).all()
            or np.any(model.illegal_run_probs < 0)
            or not np.allclose(model.illegal_run_probs.sum(axis=2), 1.0)
        ):
            raise ValueError("illegal_run_probs must contain normalized wide/no-ball distributions by state")
        team_arrays["illegal_run_probs"] = model.illegal_run_probs
    if model.batting_team_probs is not None and model.bowling_team_probs is not None:
        expected_shape = (len(model.team_names), len(model.node_names), features.N_STATES)
        for name, probs in (
            ("batting_team_probs", model.batting_team_probs),
            ("bowling_team_probs", model.bowling_team_probs),
        ):
            if probs.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}")
            if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
                raise ValueError(f"{name} must be finite and between 0 and 1")
            team_arrays[name] = probs
        team_arrays["team_names"] = np.asarray(model.team_names, dtype=str)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        probs=model.probs,
        node_names=np.asarray(model.node_names, dtype=str),
        metadata=np.asarray(json.dumps(model.metadata)),
        **team_arrays,
    )
 
 
def load(path: Path = MODEL_DIR / "node_probs.npz") -> NodeProbs:
    """Load model tables and reject state definitions incompatible with this code."""
    with np.load(Path(path), allow_pickle=False) as saved:
        required = {"probs", "node_names", "metadata"}
        if not required.issubset(saved.files):
            raise ValueError(f"Model file is missing required arrays: {sorted(required - set(saved.files))}")
        probs = np.asarray(saved["probs"], dtype=float)
        node_names = saved["node_names"].astype(str).tolist()
        metadata = json.loads(str(saved["metadata"].item()))
        team_keys = {"team_names", "batting_team_probs", "bowling_team_probs"}
        present_team_keys = team_keys.intersection(saved.files)
        if present_team_keys and present_team_keys != team_keys:
            raise ValueError("Saved model contains incomplete team probability tables")
        team_names = saved["team_names"].astype(str).tolist() if present_team_keys else []
        batting_team_probs = (
            np.asarray(saved["batting_team_probs"], dtype=float)
            if present_team_keys
            else None
        )
        bowling_team_probs = (
            np.asarray(saved["bowling_team_probs"], dtype=float)
            if present_team_keys
            else None
        )
        venue_keys = {"venue_names", "venue_probs"}
        present_venue_keys = venue_keys.intersection(saved.files)
        if present_venue_keys and present_venue_keys != venue_keys:
            raise ValueError("Saved model contains incomplete venue probability tables")
        venue_names = saved["venue_names"].astype(str).tolist() if present_venue_keys else []
        venue_probs = (
            np.asarray(saved["venue_probs"], dtype=float)
            if present_venue_keys
            else None
        )
        debut_keys = {"debut_batting_probs", "debut_bowling_probs"}
        present_debut_keys = debut_keys.intersection(saved.files)
        if present_debut_keys and present_debut_keys != debut_keys:
            raise ValueError("Saved model contains incomplete debut-player priors")
        debut_batting_probs = (
            np.asarray(saved["debut_batting_probs"], dtype=float)
            if present_debut_keys
            else None
        )
        debut_bowling_probs = (
            np.asarray(saved["debut_bowling_probs"], dtype=float)
            if present_debut_keys
            else None
        )
        illegal_run_probs = (
            np.asarray(saved["illegal_run_probs"], dtype=float)
            if "illegal_run_probs" in saved.files
            else None
        )

    current_state_definition = {
        "PHASE_STARTS": list(features.PHASE_STARTS),
        "N_PHASES": features.N_PHASES,
        "N_WICKET_BUCKETS": features.N_WICKET_BUCKETS,
        "N_INNINGS": features.N_INNINGS,
        "N_STATES": features.N_STATES,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in current_state_definition.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Saved model uses an incompatible state definition: {mismatches}")
    if probs.shape != (len(node_names), features.N_STATES):
        raise ValueError("Saved model probability shape does not match its node names and current states")
    if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError("Saved model probabilities must be finite and between 0 and 1")
    expected_team_shape = (len(team_names), len(node_names), features.N_STATES)
    for name, team_probs in (
        ("batting_team_probs", batting_team_probs),
        ("bowling_team_probs", bowling_team_probs),
    ):
        if team_probs is not None and (
            team_probs.shape != expected_team_shape
            or not np.isfinite(team_probs).all()
            or np.any((team_probs < 0) | (team_probs > 1))
        ):
            raise ValueError(f"Saved {name} has an invalid shape or probability value")
    if illegal_run_probs is not None and (
        illegal_run_probs.ndim != 3
        or illegal_run_probs.shape[:2] != (2, features.N_STATES)
        or illegal_run_probs.shape[2] < 1
        or not np.isfinite(illegal_run_probs).all()
        or np.any(illegal_run_probs < 0)
        or not np.allclose(illegal_run_probs.sum(axis=2), 1.0)
    ):
        raise ValueError("Saved illegal_run_probs has invalid shape or probabilities")
    expected_venue_shape = (len(venue_names), len(node_names), features.N_STATES)
    if venue_probs is not None and (
        venue_probs.shape != expected_venue_shape
        or not np.isfinite(venue_probs).all()
        or np.any((venue_probs < 0) | (venue_probs > 1))
    ):
        raise ValueError("Saved venue_probs has an invalid shape or probability value")
    for name, debut_probs in (
        ("debut_batting_probs", debut_batting_probs),
        ("debut_bowling_probs", debut_bowling_probs),
    ):
        if debut_probs is not None and (
            debut_probs.shape != (len(node_names), features.N_STATES)
            or not np.isfinite(debut_probs).all()
            or np.any((debut_probs < 0) | (debut_probs > 1))
        ):
            raise ValueError(f"Saved {name} has an invalid shape or probability value")
    return NodeProbs(
        probs=probs,
        node_names=node_names,
        metadata=metadata,
        team_names=team_names,
        batting_team_probs=batting_team_probs,
        bowling_team_probs=bowling_team_probs,
        illegal_run_probs=illegal_run_probs,
        venue_names=venue_names,
        venue_probs=venue_probs,
        debut_batting_probs=debut_batting_probs,
        debut_bowling_probs=debut_bowling_probs,
    )
 
 
# =============================================================================
# 8. SANITY CHECKS
# =============================================================================
def check_model(model: NodeProbs, train: pd.DataFrame) -> None:
    """Raise AssertionError when probability tables fail basic calibration checks."""
    nodes = build_tree()
    expected_names = [node.name for node in nodes]
    if model.node_names != expected_names:
        raise AssertionError(f"Expected nodes {expected_names}, got {model.node_names}")
    if model.probs.shape != (len(nodes), features.N_STATES):
        raise AssertionError("Probability table has an unexpected shape")
    if not np.isfinite(model.probs).all() or np.any((model.probs < 0) | (model.probs > 1)):
        raise AssertionError("Every node probability must be finite and between 0 and 1")

    train = train.copy()
    if "innings_number" not in train:
        raise AssertionError("Training data must contain an 'innings_number' column")
    if "is_super_over" in train:
        train = train.loc[~train["is_super_over"].astype(bool)]
    train = train.loc[train["innings_number"].between(1, features.N_INNINGS)].copy()
    if "state_id" not in train:
        train = features.add_state_features(train)
    if train.empty:
        raise AssertionError("Cannot check a model against an empty training set")
    if train["state_id"].min() < 0 or train["state_id"].max() >= features.N_STATES:
        raise AssertionError("Training data contains an out-of-range state_id")

    p_wicket, p_legal, p_wide, p_runs, p_boundary, p_six, p_one, p_two = model.probs
    no_wicket = 1 - p_wicket
    no_legal = 1 - p_legal
    no_runs = 1 - p_runs
    no_boundary = 1 - p_boundary
    no_six = 1 - p_six
    no_one = 1 - p_one
    no_two = 1 - p_two
    terminal_probabilities = np.stack(
        [
            p_wicket,
            no_wicket * no_legal * p_wide,
            no_wicket * no_legal * (1 - p_wide),
            no_wicket * p_legal * no_runs,
            no_wicket * p_legal * p_runs * p_boundary * p_six,
            no_wicket * p_legal * p_runs * p_boundary * no_six,
            no_wicket * p_legal * p_runs * no_boundary * p_one,
            no_wicket * p_legal * p_runs * no_boundary * no_one * p_two,
            no_wicket * p_legal * p_runs * no_boundary * no_one * no_two,
        ]
    )
    if not np.allclose(terminal_probabilities.sum(axis=0), 1.0, atol=1e-10):
        raise AssertionError("Outcome path probabilities do not sum to 1 for every state")

    states = train["state_id"].to_numpy(dtype=int)
    state_weights = np.bincount(states, minlength=features.N_STATES).astype(float)
    if state_weights.sum() == 0:
        raise AssertionError("Training data has no valid state observations")
    expected_rates = {
        "wicket": p_wicket,
        "dot": no_wicket * p_legal * no_runs,
        "four": no_wicket * p_legal * p_runs * p_boundary * no_six,
        "six": no_wicket * p_legal * p_runs * p_boundary * p_six,
    }
    observed_rates = {
        "wicket": train["is_wicket"].astype(bool).to_numpy(),
        "dot": ((train["total_runs"] == 0) & ~train["is_wicket"]).to_numpy(),
        "four": ((train["total_runs"] == 4) & ~train["is_wicket"]).to_numpy(),
        "six": ((train["total_runs"] == 6) & ~train["is_wicket"]).to_numpy(),
    }
    for label, state_probabilities in expected_rates.items():
        predicted = np.dot(state_weights, state_probabilities) / state_weights.sum()
        observed = observed_rates[label].mean()
        if abs(predicted - observed) > 0.04:
            raise AssertionError(
                f"Model {label} rate ({predicted:.3f}) differs from training rate "
                f"({observed:.3f}) by more than 0.04"
            )

    # Compare high/low wicket buckets and middle/death phases after pooling innings.
    by_innings = features.N_PHASES * features.N_WICKET_BUCKETS
    wicket_by_bucket = np.zeros(features.N_WICKET_BUCKETS)
    boundary_by_phase = np.zeros(features.N_PHASES)
    for innings in range(features.N_INNINGS):
        base = innings * by_innings
        for bucket in range(features.N_WICKET_BUCKETS):
            ids = base + np.arange(features.N_PHASES) * features.N_WICKET_BUCKETS + bucket
            weights = state_weights[ids]
            if weights.sum():
                wicket_by_bucket[bucket] += np.dot(weights, p_wicket[ids]) / weights.sum()
        for phase in range(features.N_PHASES):
            ids = base + phase * features.N_WICKET_BUCKETS + np.arange(features.N_WICKET_BUCKETS)
            weights = state_weights[ids]
            if weights.sum():
                boundary_by_phase[phase] += np.dot(weights, p_boundary[ids]) / weights.sum()
    if wicket_by_bucket[-1] <= wicket_by_bucket[0]:
        raise AssertionError("Wicket probability should be higher with nine wickets down")
    if boundary_by_phase[-1] <= boundary_by_phase[1]:
        raise AssertionError("Boundary probability should be higher in the death overs")

    score_by_state = (
        no_wicket
        * (
            no_legal * 1.0
            + p_legal
            * p_runs
            * (
                no_boundary * (p_one + 2 * no_one * p_two + 3 * no_one * no_two)
                + p_boundary * (4 * no_six + 6 * p_six)
            )
        )
    )
    if model.illegal_run_probs is None:
        expected_illegal_runs = np.ones(features.N_STATES)
    else:
        illegal_run_values = np.arange(
            1, model.illegal_run_probs.shape[2] + 1, dtype=float
        )
        expected_wide_runs = model.illegal_run_probs[0] @ illegal_run_values
        expected_no_ball_runs = model.illegal_run_probs[1] @ illegal_run_values
        expected_illegal_runs = (
            p_wide * expected_wide_runs + (1 - p_wide) * expected_no_ball_runs
        )
    score_by_state += no_wicket * (1 - p_legal) * expected_illegal_runs
    predicted_delivery_runs = score_by_state[states]
    actual_runs = train["total_runs"].to_numpy(dtype=float)
    innings = train["innings_number"].to_numpy()
    match_ids = train["match_id"].to_numpy() if "match_id" in train else None
    first_innings = innings == 1
    if match_ids is not None and first_innings.any():
        first = pd.DataFrame(
            {
                "match_id": match_ids[first_innings],
                "predicted": predicted_delivery_runs[first_innings],
                "actual": actual_runs[first_innings],
            }
        ).groupby("match_id")[["predicted", "actual"]].sum()
        if not first.empty:
            score_difference = abs(first["predicted"].mean() - first["actual"].mean())
            if score_difference > max(8.0, 0.12 * first["actual"].mean()):
                raise AssertionError(
                    f"Implied first-innings average differs from training by {score_difference:.1f} runs"
                )

    second_innings = innings == 2
    if second_innings.any():
        predicted_rate = predicted_delivery_runs[second_innings].mean()
        observed_rate = actual_runs[second_innings].mean()
        if abs(predicted_rate - observed_rate) > 0.25:
            raise AssertionError(
                f"Implied second-innings runs per ball ({predicted_rate:.3f}) differs "
                f"from training ({observed_rate:.3f}) by more than 0.25"
            )
 
 
# =============================================================================
# 9. ENTRY POINT
# =============================================================================
def main() -> None:
    deliveries = pd.read_parquet(PROCESSED / "deliveries.parquet")
    matches = pd.read_parquet(PROCESSED / "matches.parquet")
 
    model = fit(deliveries, matches)
    check_model(model, training_deliveries(deliveries, matches))
    save(model)
    print(f"Saved model with {len(model.node_names)} nodes x {features.N_STATES} states")
 
 
if __name__ == "__main__":
    main()