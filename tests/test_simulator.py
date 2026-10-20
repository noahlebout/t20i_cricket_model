import numpy as np
import pandas as pd
import pytest

from src import features, model, player_model, simulator


def make_model(
    *,
    wicket: float = 0,
    legal: float = 1,
    wide: float = 0,
    runs: float = 0,
    boundary: float = 0,
    six: float = 0,
    one: float = 0,
    two: float = 0,
) -> model.NodeProbs:
    names = [node.name for node in model.build_tree()]
    probs = np.tile([wicket, legal, wide, runs, boundary, six, one, two], (features.N_STATES, 1)).T
    return model.NodeProbs(probs, names, {})


def test_simulate_ball_uses_given_state_and_returns_updated_state() -> None:
    fitted = make_model(runs=1, boundary=1, six=1)
    state = simulator.GameState(innings_number=2, legal_balls=36, wickets=2, runs=10)

    result = simulator.simulate_ball(fitted, state, rng=4)

    assert result.state_before == state
    assert result.outcome == "six"
    assert result.runs_scored == 6
    assert result.state_after == simulator.GameState(2, 37, 2, 16)


def test_simulate_ball_samples_non_boundary_run_categories() -> None:
    fitted = make_model(runs=1, one=0, two=1)
    result = simulator.simulate_ball(fitted, simulator.GameState(), rng=1)

    assert result.outcome == "two"
    assert result.state_after.runs == 2
    assert result.state_after.legal_balls == 1


def test_wicket_and_illegal_delivery_state_updates() -> None:
    wicket = simulator.simulate_ball(make_model(wicket=1), simulator.GameState(), rng=1)
    wide = simulator.simulate_ball(make_model(legal=0, wide=1), simulator.GameState(), rng=1)

    assert wicket.is_wicket and wicket.is_legal
    assert wicket.state_after.wickets == 1
    assert wicket.state_after.legal_balls == 1
    assert wide.outcome == "wide+1" and not wide.is_legal
    assert wide.state_after.runs == 1
    assert wide.state_after.legal_balls == 0


def test_wide_and_no_ball_add_modelled_total_runs_without_using_a_legal_ball() -> None:
    fitted = make_model(legal=0, wide=1)
    fitted.illegal_run_probs = np.zeros((2, features.N_STATES, 7))
    fitted.illegal_run_probs[0, :, 4] = 1
    fitted.illegal_run_probs[1, :, 6] = 1

    wide = simulator.simulate_ball(fitted, simulator.GameState(), rng=1)
    no_ball_model = make_model(legal=0, wide=0)
    no_ball_model.illegal_run_probs = fitted.illegal_run_probs
    no_ball = simulator.simulate_ball(no_ball_model, simulator.GameState(), rng=1)

    assert wide.outcome == "wide+5"
    assert wide.runs_scored == wide.state_after.runs == 5
    assert wide.state_after.legal_balls == 0
    assert no_ball.outcome == "no_ball+7"
    assert no_ball.runs_scored == no_ball.state_after.runs == 7
    assert no_ball.state_after.legal_balls == 0


def test_record_observed_delivery_updates_actual_score_and_ball_state() -> None:
    before = simulator.GameState(innings_number=1, legal_balls=5, wickets=2, runs=17)

    delivery = simulator.record_delivery(
        before,
        runs_scored=1,
        is_legal=False,
        outcome="wide+1",
    )

    assert delivery.state_before == before
    assert delivery.runs_scored == 1
    assert delivery.state_after == simulator.GameState(
        innings_number=1,
        legal_balls=5,
        wickets=2,
        runs=18,
    )


def test_actual_wicket_advances_striker_and_over_end_rotates_strike() -> None:
    state = simulator.GameState(legal_balls=4)
    wicket = simulator.record_delivery(
        state,
        runs_scored=0,
        is_wicket=True,
        outcome="wicket",
    )
    assert simulator.advance_batting_positions(0, 1, 2, wicket) == (2, 1, 3)

    last_ball = simulator.record_delivery(
        simulator.GameState(legal_balls=5),
        runs_scored=0,
        outcome="dot",
    )
    assert simulator.advance_batting_positions(0, 1, 2, last_ball) == (1, 0, 2)


def test_observed_extra_can_supply_completed_runs_for_strike_rotation() -> None:
    delivery = simulator.record_delivery(
        simulator.GameState(),
        runs_scored=2,
        is_legal=False,
        outcome="wide+2",
        strike_runs=1,
    )

    assert simulator.advance_batting_positions(0, 1, 2, delivery) == (1, 0, 2)


def test_innings_stops_on_over_limit_or_chase_target() -> None:
    dot_model = make_model()
    near_end = simulator.GameState(legal_balls=119, runs=20)
    completed = simulator.simulate_innings(dot_model, near_end, rng=3)
    assert completed.final_state.legal_balls == simulator.MAX_LEGAL_BALLS
    assert len(completed.deliveries) == 1

    chase = simulator.simulate_innings(
        make_model(runs=1, boundary=1, six=1),
        simulator.GameState(innings_number=2, legal_balls=119, runs=20, target_runs=21),
        rng=3,
    )
    assert chase.final_state.runs == 26
    assert chase.final_state.legal_balls == 120
    assert chase.final_state.is_complete


def test_simulate_match_returns_two_innings_and_repeatable_seed() -> None:
    fitted = make_model()

    first = simulator.simulate_match(fitted, rng=12)
    second = simulator.simulate_match(fitted, rng=12)

    assert first == second
    assert first.first_innings.final_state.innings_number == 1
    assert first.second_innings.final_state.innings_number == 2
    assert first.second_innings.final_state.target_runs == first.first_innings.final_state.runs + 1
    assert first.is_tie
    assert first.winner_innings is None


def test_simulate_match_stops_chase_when_target_is_reached() -> None:
    fitted = make_model()
    fitted.probs[3, features.N_PHASES * features.N_WICKET_BUCKETS :] = 1
    fitted.probs[6, features.N_PHASES * features.N_WICKET_BUCKETS :] = 1

    result = simulator.simulate_match(fitted, rng=2)

    assert result.first_innings.final_state.runs == 0
    assert result.second_innings.final_state.runs == 1
    assert result.second_innings.final_state.legal_balls == 1
    assert result.winner_innings == 2


def test_cannot_simulate_after_innings_is_complete() -> None:
    state = simulator.GameState(legal_balls=120)
    with pytest.raises(ValueError, match="complete"):
        simulator.simulate_ball(make_model(), state)


def test_player_lineups_track_strike_and_fixed_bowler_rotation() -> None:
    fitted = make_model(runs=1, one=1)
    nodes = model.build_tree()
    priors = fitted.probs.astype(float)
    ratings = player_model.PlayerRatings(
        player_ids=[],
        batting_probs=np.empty((0, len(nodes), features.N_STATES)),
        bowling_probs=np.empty((0, len(nodes), features.N_STATES)),
        debut_batting_probs=priors,
        debut_bowling_probs=priors,
        batting_form={},
        bowling_form={},
        as_of_date=pd.Timestamp("2024-01-01"),
    )
    batting_order = tuple(f"batter-{index}" for index in range(11))
    bowling_order = tuple(f"bowler-{index}" for index in range(5))

    innings = simulator.simulate_innings(
        fitted,
        simulator.GameState(legal_balls=4),
        rng=3,
        batting_order=batting_order,
        bowling_order=bowling_order,
        player_ratings=ratings,
    )

    assert [delivery.batter_id for delivery in innings.deliveries[:3]] == [
        "batter-0",
        "batter-1",
        "batter-1",
    ]
    assert [delivery.bowler_id for delivery in innings.deliveries[:3]] == [
        "bowler-0",
        "bowler-0",
        "bowler-1",
    ]


def test_player_lineups_accept_a_20_over_schedule_with_four_over_limit() -> None:
    fitted = make_model()
    nodes = model.build_tree()
    priors = fitted.probs.astype(float)
    ratings = player_model.PlayerRatings(
        player_ids=[],
        batting_probs=np.empty((0, len(nodes), features.N_STATES)),
        bowling_probs=np.empty((0, len(nodes), features.N_STATES)),
        debut_batting_probs=priors,
        debut_bowling_probs=priors,
        batting_form={},
        bowling_form={},
        as_of_date=pd.Timestamp("2024-01-01"),
    )
    batting_order = tuple(f"batter-{index}" for index in range(11))
    bowling_schedule = tuple(
        f"bowler-{over // 4}" for over in range(20)
    )

    innings = simulator.simulate_innings(
        fitted,
        rng=3,
        batting_order=batting_order,
        bowling_order=bowling_schedule,
        player_ratings=ratings,
    )

    assert len(innings.deliveries) == simulator.MAX_LEGAL_BALLS
    assert [delivery.bowler_id for delivery in innings.deliveries[:30:24]] == [
        "bowler-0",
        "bowler-1",
    ]
    with pytest.raises(ValueError, match="four overs"):
        simulator.simulate_innings(
            fitted,
            batting_order=batting_order,
            bowling_order=("bowler-0",) * 20,
            player_ratings=ratings,
        )
