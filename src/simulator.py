"""Ball-by-ball and match simulation using the fitted delivery model."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from src import features, model

MAX_LEGAL_BALLS = 120
N_WICKETS = 10


@dataclass(frozen=True)
class GameState:
    """State immediately before the next delivery."""

    innings_number: int = 1
    legal_balls: int = 0
    wickets: int = 0
    runs: int = 0
    target_runs: int | None = None

    def __post_init__(self) -> None:
        for name in ("innings_number", "legal_balls", "wickets", "runs"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
        if self.target_runs is not None and not isinstance(self.target_runs, (int, np.integer)):
            raise TypeError("target_runs must be an integer or None")
        if self.innings_number not in (1, 2):
            raise ValueError("innings_number must be 1 or 2")
        if not 0 <= self.legal_balls <= MAX_LEGAL_BALLS:
            raise ValueError(f"legal_balls must be in [0, {MAX_LEGAL_BALLS}]")
        if not 0 <= self.wickets <= N_WICKETS:
            raise ValueError(f"wickets must be in [0, {N_WICKETS}]")
        if self.runs < 0:
            raise ValueError("runs cannot be negative")
        if self.target_runs is not None and self.target_runs < 1:
            raise ValueError("target_runs must be positive")

    @property
    def is_complete(self) -> bool:
        return (
            self.legal_balls >= MAX_LEGAL_BALLS
            or self.wickets >= N_WICKETS
            or (self.target_runs is not None and self.runs >= self.target_runs)
        )

    @property
    def state_id(self) -> int:
        if self.is_complete:
            raise ValueError("A completed innings has no next-delivery state")
        return features.state_id_of(self.legal_balls, self.wickets, self.innings_number)


@dataclass(frozen=True)
class DeliveryResult:
    state_before: GameState
    state_after: GameState
    runs_scored: int
    is_wicket: bool
    is_legal: bool
    outcome: str
    batter_id: str | None = None
    bowler_id: str | None = None
    strike_runs: int | None = None


@dataclass(frozen=True)
class InningsResult:
    final_state: GameState
    deliveries: tuple[DeliveryResult, ...]


@dataclass(frozen=True)
class MatchResult:
    first_innings: InningsResult
    second_innings: InningsResult
    winner_innings: int | None

    @property
    def is_tie(self) -> bool:
        return self.winner_innings is None


def _as_generator(
    rng: np.random.Generator | int | None,
) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _sample_yes(probability: float, rng: np.random.Generator) -> bool:
    return bool(rng.random() < probability)


def simulate_ball(
    fitted_model: model.NodeProbs,
    state: GameState,
    rng: np.random.Generator | int | None = None,
    *,
    batting_team: str | None = None,
    bowling_team: str | None = None,
    venue: str | None = None,
    batter_id: str | None = None,
    bowler_id: str | None = None,
    player_ratings=None,
) -> DeliveryResult:
    """Simulate one delivery and return the delivery plus the updated game state.

    Wickets are treated as legal, zero-run deliveries. Illegal deliveries use
    the fitted total-run distribution for wides and no-balls.
    """
    if state.is_complete:
        raise ValueError("Cannot simulate a delivery after the innings is complete")

    generator = _as_generator(rng)
    probabilities = model.get_node_probs(
        fitted_model,
        state.state_id,
        batting_team=batting_team,
        bowling_team=bowling_team,
        venue=venue,
        batting_player_id=batter_id,
        bowling_player_id=bowler_id,
        player_ratings=player_ratings,
    )
    if probabilities.shape != (len(fitted_model.node_names),):
        raise ValueError("Model returned an invalid probability vector")

    is_wicket = _sample_yes(probabilities[0], generator)
    if is_wicket:
        runs_scored, is_legal, outcome = 0, True, "wicket"
    elif not _sample_yes(probabilities[1], generator):
        is_legal = False
        is_wide = _sample_yes(probabilities[2], generator)
        outcome_type = "wide" if is_wide else "no_ball"
        runs_scored = model.sample_illegal_runs(
            fitted_model, state.state_id, outcome_type, generator
        )
        outcome = f"{outcome_type}+{runs_scored}"
    elif not _sample_yes(probabilities[3], generator):
        runs_scored, is_legal, outcome = 0, True, "dot"
    elif _sample_yes(probabilities[4], generator):
        is_six = _sample_yes(probabilities[5], generator)
        runs_scored, is_legal = (6 if is_six else 4), True
        outcome = "six" if is_six else "four"
    elif _sample_yes(probabilities[6], generator):
        runs_scored, is_legal, outcome = 1, True, "one"
    elif _sample_yes(probabilities[7], generator):
        runs_scored, is_legal, outcome = 2, True, "two"
    else:
        runs_scored, is_legal, outcome = 3, True, "three"

    next_state = GameState(
        innings_number=state.innings_number,
        legal_balls=state.legal_balls + int(is_legal),
        wickets=state.wickets + int(is_wicket),
        runs=state.runs + runs_scored,
        target_runs=state.target_runs,
    )
    return DeliveryResult(
        state_before=state,
        state_after=next_state,
        runs_scored=runs_scored,
        is_wicket=is_wicket,
        is_legal=is_legal,
        outcome=outcome,
        batter_id=batter_id,
        bowler_id=bowler_id,
    )


def record_delivery(
    state: GameState,
    *,
    runs_scored: int,
    is_wicket: bool = False,
    is_legal: bool = True,
    outcome: str | None = None,
    batter_id: str | None = None,
    bowler_id: str | None = None,
    strike_runs: int | None = None,
) -> DeliveryResult:
    """Advance the innings state using an observed delivery instead of a sample."""
    if state.is_complete:
        raise ValueError("Cannot record a delivery after the innings is complete")
    if not isinstance(runs_scored, (int, np.integer)) or runs_scored < 0:
        raise ValueError("runs_scored must be a non-negative integer")
    if not isinstance(is_wicket, (bool, np.bool_)):
        raise TypeError("is_wicket must be a boolean")
    if not isinstance(is_legal, (bool, np.bool_)):
        raise TypeError("is_legal must be a boolean")
    if strike_runs is not None and (
        not isinstance(strike_runs, (int, np.integer)) or strike_runs < 0
    ):
        raise ValueError("strike_runs must be a non-negative integer or None")
    next_state = GameState(
        innings_number=state.innings_number,
        legal_balls=state.legal_balls + int(is_legal),
        wickets=state.wickets + int(is_wicket),
        runs=state.runs + int(runs_scored),
        target_runs=state.target_runs,
    )
    return DeliveryResult(
        state_before=state,
        state_after=next_state,
        runs_scored=int(runs_scored),
        is_wicket=bool(is_wicket),
        is_legal=bool(is_legal),
        outcome=outcome or f"runs-{runs_scored}",
        batter_id=batter_id,
        bowler_id=bowler_id,
        strike_runs=None if strike_runs is None else int(strike_runs),
    )


def advance_batting_positions(
    striker_index: int,
    non_striker_index: int,
    next_batter_index: int,
    delivery: DeliveryResult,
) -> tuple[int, int, int]:
    """Advance strike and the next batter after one recorded or simulated ball."""
    strike_runs = (
        delivery.runs_scored
        if delivery.strike_runs is None
        else delivery.strike_runs
    )
    if strike_runs % 2:
        striker_index, non_striker_index = non_striker_index, striker_index
    if delivery.is_wicket and next_batter_index < 11:
        striker_index = next_batter_index
        next_batter_index += 1
    if (
        delivery.is_legal
        and delivery.state_after.legal_balls % 6 == 0
        and not delivery.state_after.is_complete
    ):
        striker_index, non_striker_index = non_striker_index, striker_index
    return striker_index, non_striker_index, next_batter_index


def simulate_innings(
    fitted_model: model.NodeProbs,
    state: GameState | None = None,
    *,
    innings_number: int = 1,
    target_runs: int | None = None,
    rng: np.random.Generator | int | None = None,
    batting_team: str | None = None,
    bowling_team: str | None = None,
    venue: str | None = None,
    batting_order: tuple[str, ...] | list[str] | None = None,
    bowling_order: tuple[str, ...] | list[str] | None = None,
    player_ratings=None,
    striker_index: int = 0,
    non_striker_index: int = 1,
    next_batter_index: int = 2,
) -> InningsResult:
    """Simulate from the supplied state to the end of an innings."""
    if state is None:
        state = GameState(innings_number=innings_number, target_runs=target_runs)
    elif target_runs is not None and target_runs != state.target_runs:
        raise ValueError("target_runs conflicts with the supplied game state")
    if (batting_order is None) != (bowling_order is None):
        raise ValueError("Provide both batting_order and bowling_order, or neither")
    if batting_order is not None:
        if len(batting_order) != 11 or len(set(batting_order)) != 11:
            raise ValueError("batting_order must contain 11 unique player IDs")
        if len(bowling_order) == 5:
            if len(set(bowling_order)) != 5:
                raise ValueError("bowling_order must contain 5 unique player IDs")
        elif len(bowling_order) == MAX_LEGAL_BALLS // 6:
            if max(Counter(bowling_order).values()) > 4:
                raise ValueError("A bowler cannot be scheduled for more than four overs")
        else:
            raise ValueError("bowling_order must contain 5 bowlers or a 20-over schedule")
        if player_ratings is None:
            raise ValueError("Player lineups require player ratings")

    if batting_order is not None and (
        not 0 <= striker_index < len(batting_order)
        or not 0 <= non_striker_index < len(batting_order)
        or striker_index == non_striker_index
        or not 2 <= next_batter_index <= len(batting_order)
    ):
        raise ValueError("Invalid batting-order positions")
    generator = _as_generator(rng)
    deliveries: list[DeliveryResult] = []
    while not state.is_complete:
        batter_id = (
            batting_order[striker_index] if batting_order is not None else None
        )
        bowler_id = (
            bowling_order[
                (state.legal_balls // 6) % len(bowling_order)
                if len(bowling_order) == 5
                else state.legal_balls // 6
            ]
            if bowling_order is not None
            else None
        )
        result = simulate_ball(
            fitted_model,
            state,
            generator,
            batting_team=batting_team,
            bowling_team=bowling_team,
            venue=venue,
            batter_id=batter_id,
            bowler_id=bowler_id,
            player_ratings=player_ratings,
        )
        deliveries.append(result)
        state = result.state_after
        if batting_order is not None:
            (
                striker_index,
                non_striker_index,
                next_batter_index,
            ) = advance_batting_positions(
                striker_index,
                non_striker_index,
                next_batter_index,
                result,
            )
    return InningsResult(final_state=state, deliveries=tuple(deliveries))


def simulate_match(
    fitted_model: model.NodeProbs,
    rng: np.random.Generator | int | None = None,
    *,
    batting_first_team: str | None = None,
    bowling_first_team: str | None = None,
    venue: str | None = None,
    batting_first_order: tuple[str, ...] | list[str] | None = None,
    bowling_first_order: tuple[str, ...] | list[str] | None = None,
    batting_second_order: tuple[str, ...] | list[str] | None = None,
    bowling_second_order: tuple[str, ...] | list[str] | None = None,
    player_ratings=None,
) -> MatchResult:
    """Simulate a match with optional team strengths and learned venue effects."""
    if (batting_first_team is None) != (bowling_first_team is None):
        raise ValueError("Provide both batting_first_team and bowling_first_team, or neither")
    lineups = (
        batting_first_order,
        bowling_first_order,
        batting_second_order,
        bowling_second_order,
    )
    if any(lineup is not None for lineup in lineups) and (
        any(lineup is None for lineup in lineups) or player_ratings is None
    ):
        raise ValueError("Player-aware match simulation requires both teams' lineups and ratings")
    generator = _as_generator(rng)
    first = simulate_innings(
        fitted_model,
        rng=generator,
        batting_team=batting_first_team,
        bowling_team=bowling_first_team,
        venue=venue,
        batting_order=batting_first_order,
        bowling_order=bowling_first_order,
        player_ratings=player_ratings,
    )
    target = first.final_state.runs + 1
    second = simulate_innings(
        fitted_model,
        innings_number=2,
        target_runs=target,
        rng=generator,
        batting_team=bowling_first_team,
        bowling_team=batting_first_team,
        venue=venue,
        batting_order=batting_second_order,
        bowling_order=bowling_second_order,
        player_ratings=player_ratings,
    )
    if second.final_state.runs >= target:
        winner = 2
    elif first.final_state.runs > second.final_state.runs:
        winner = 1
    else:
        winner = None
    return MatchResult(first, second, winner)
