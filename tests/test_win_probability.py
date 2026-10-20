import numpy as np
import pandas as pd

from src import features, model, simulator, win_probability


def _fitted_model() -> model.NodeProbs:
    nodes = model.build_tree()
    return model.NodeProbs(
        probs=np.full((len(nodes), features.N_STATES), 0.5),
        node_names=[node.name for node in nodes],
        metadata={},
        team_names=["A", "B"],
    )


def test_forecast_of_completed_tied_chase_splits_win_probability() -> None:
    fitted = _fitted_model()
    state = simulator.GameState(
        innings_number=2,
        legal_balls=120,
        runs=50,
        target_runs=51,
    )

    probability, standard_error, low, median, high = win_probability._forecast_state(
        fitted,
        state,
        first_team="A",
        second_team="B",
        venue=None,
        first_runs=50,
        n_simulations=2,
        seed=1,
    )
    assert probability == 0.5
    assert standard_error == 0
    assert (low, median, high) == (50.0, 50.0, 50.0)


def test_forecast_state_simulates_the_remaining_match() -> None:
    fitted = _fitted_model()

    probability, standard_error, low, median, high = win_probability._forecast_state(
        fitted,
        simulator.GameState(innings_number=1),
        first_team="A",
        second_team="B",
        venue=None,
        first_runs=None,
        n_simulations=2,
        seed=1,
    )

    assert 0 <= probability <= 1
    assert standard_error >= 0
    assert 0 <= low <= median <= high


def test_vectorized_rollout_matches_a_deterministic_six_at_the_last_ball() -> None:
    fitted = _fitted_model()
    fitted.probs[:] = 0.0
    fitted.probs[fitted.node_names.index("legal")] = 1.0
    fitted.probs[fitted.node_names.index("runs")] = 1.0
    fitted.probs[fitted.node_names.index("boundary")] = 1.0
    fitted.probs[fitted.node_names.index("six")] = 1.0
    probability_table = np.tile(
        fitted.probs.T[None, :, :],
        (features.N_INNINGS, 1, 1),
    )

    scores = win_probability._simulate_innings_batch(
        probability_table,
        None,
        simulator.GameState(legal_balls=119),
        n_simulations=4,
        rng=np.random.default_rng(1),
    )

    assert scores.tolist() == [6, 6, 6, 6]


def test_vectorized_rollout_samples_wide_runs_without_advancing_the_over() -> None:
    fitted = _fitted_model()
    fitted.probs[:] = 0.0
    fitted.probs[fitted.node_names.index("legal")] = 0.5
    fitted.probs[fitted.node_names.index("wide")] = 1.0
    fitted.probs[fitted.node_names.index("runs")] = 0.0
    probability_table = np.tile(
        fitted.probs.T[None, :, :],
        (features.N_INNINGS, 1, 1),
    )
    illegal_run_probs = np.zeros((2, features.N_STATES, 2))
    illegal_run_probs[0, :, 1] = 1.0
    illegal_run_probs[1, :, 0] = 1.0

    scores = win_probability._simulate_innings_batch(
        probability_table,
        illegal_run_probs,
        simulator.GameState(legal_balls=119),
        n_simulations=100,
        rng=np.random.default_rng(2),
    )

    assert np.all(scores % 2 == 0)
    assert np.any(scores > 0)


def test_analyze_match_links_each_ball_to_the_previous_forecast(
    monkeypatch, tmp_path
) -> None:
    fitted = _fitted_model()
    matches = pd.DataFrame(
        {
            "match_id": ["match-1"],
            "date": ["2025-01-01"],
            "team_1": ["A"],
            "team_2": ["B"],
            "first_batting_team": ["A"],
            "winner": ["A"],
            "venue": ["Ground"],
            "has_super_over": [False],
        }
    )
    deliveries = pd.DataFrame(
        {
            "match_id": ["match-1", "match-1"],
            "innings_number": [1, 2],
            "legal_balls_before": [0, 0],
            "wickets_before": [0, 0],
            "runs_before": [0, 0],
            "is_legal": [True, True],
            "is_wicket": [False, False],
            "total_runs": [1, 0],
        }
    )

    def fake_forecast(_fitted_model, state, **_kwargs):
        _ = _fitted_model, _kwargs
        probability = 0.5 + state.runs / 100
        return probability, 0.01, 100.0, 120.0, 140.0

    monkeypatch.setattr(win_probability, "_forecast_state", fake_forecast)

    metadata, results = win_probability.analyze_match(
        fitted,
        deliveries,
        matches,
        "match-1",
        n_simulations=10,
    )

    assert metadata["n_deliveries"] == 2
    assert results["win_probability_before"].tolist() == [0.5, 0.51]
    assert results["win_probability_after"].tolist() == [0.51, 0.5]
    assert np.allclose(results["win_probability_change"], [0.01, -0.01])
    assert results["actual_first_team_win"].tolist() == [1.0, 1.0]
    win_probability.plot_match(metadata, results, tmp_path)
    assert (tmp_path / "win_probability_timeline.png").stat().st_size > 0
    assert (tmp_path / "first_innings_projection.png").stat().st_size > 0


def test_analyze_match_rejects_matches_used_to_fit_the_model() -> None:
    matches = pd.DataFrame(
        {
            "match_id": ["match-1"],
            "date": ["2024-12-31"],
            "team_1": ["A"],
            "team_2": ["B"],
            "first_batting_team": ["A"],
            "winner": ["A"],
        }
    )

    try:
        win_probability.analyze_match(
            _fitted_model(),
            pd.DataFrame(
                columns=[
                    "match_id",
                    "innings_number",
                    "legal_balls_before",
                    "wickets_before",
                    "runs_before",
                    "is_legal",
                    "is_wicket",
                    "total_runs",
                ]
            ),
            matches,
            "match-1",
        )
    except ValueError as exc:
        assert "must be held out" in str(exc)
    else:
        raise AssertionError("Expected a pre-cutoff match to be rejected")
