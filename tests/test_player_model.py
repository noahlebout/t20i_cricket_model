import numpy as np
import pandas as pd

from src import features, model, player_model


def _fixture_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    match_dates = {
        "old-1": "2024-01-01",
        "old-2": "2024-01-02",
        "future": "2024-01-03",
    }
    matches = pd.DataFrame(
        {
            "match_id": list(match_dates),
            "date": list(match_dates.values()),
            "team_1": ["A"] * 3,
            "team_2": ["B"] * 3,
            "winner": ["A"] * 3,
        }
    )
    rows = []
    for match_id, runs in (("old-1", 0), ("old-2", 1), ("future", 6)):
        first_batter_id = "future-player" if match_id == "future" else "batter-1"
        first_bowler_id = "future-bowler" if match_id == "future" else "bowler-1"
        for innings_number, batting_team, batter_id, bowler_id, total_runs in (
            (1, "A", first_batter_id, first_bowler_id, runs),
            (2, "B", "other-batter", "other-bowler", 0),
        ):
            rows.append(
                {
                    "match_id": match_id,
                    "innings_number": innings_number,
                    "over": 0,
                    "legal_balls_before": 0,
                    "wickets_before": 0,
                    "batting_team": batting_team,
                    "bowling_team": "B" if batting_team == "A" else "A",
                    "batter_id": batter_id,
                    "bowler_id": bowler_id,
                    "batter": batter_id,
                    "bowler": bowler_id,
                    "is_wicket": False,
                    "is_legal": True,
                    "wides": 0,
                    "noballs": 0,
                    "total_runs": total_runs,
                    "batter_runs": total_runs,
                    "extras_runs": 0,
                    "byes": 0,
                    "legbyes": 0,
                    "wicket_kind": None,
                    "player_out": None,
                    "penalty": 0,
                }
            )
    deliveries = pd.DataFrame(rows)
    return deliveries, matches


def test_player_form_excludes_matches_on_and_after_as_of_date() -> None:
    deliveries, matches = _fixture_data()
    fitted = model.fit(deliveries, matches)

    ratings = player_model.build_player_ratings(
        deliveries,
        matches,
        fitted,
        as_of_date="2024-01-03",
    )

    assert ratings.as_of_date == pd.Timestamp("2024-01-03")
    assert ratings.batting_form["batter-1"]["matches"] == 2
    assert ratings.batting_form["batter-1"]["runs"] == 1
    assert ratings.batting_form["batter-1"]["balls_faced"] == 2
    assert ratings.bowling_form["bowler-1"]["matches"] == 2


def test_state_averages_and_bowling_usage_use_only_prior_appearances() -> None:
    deliveries, matches = _fixture_data()
    target = deliveries.index[
        (deliveries["match_id"] == "old-2")
        & (deliveries["batter_id"] == "batter-1")
    ][0]
    deliveries.loc[target, ["batter_runs", "total_runs", "is_wicket", "player_out", "wicket_kind"]] = [
        20,
        20,
        True,
        "batter-1",
        "bowled",
    ]
    comparison = deliveries.loc[[target]].copy()
    comparison.loc[:, "batter_id"] = "comparison-batter"
    comparison.loc[:, "batter"] = "comparison-batter"
    comparison.loc[:, "bowler_id"] = "comparison-bowler"
    comparison.loc[:, "bowler"] = "comparison-bowler"
    comparison.loc[:, "batter_runs"] = 0
    comparison.loc[:, "total_runs"] = 0
    comparison.loc[:, "player_out"] = "comparison-batter"
    comparison.loc[:, "legal_balls_before"] = 1
    deliveries = pd.concat([deliveries, comparison], ignore_index=True)
    fitted = model.fit(deliveries, matches)

    ratings = player_model.build_player_ratings(
        deliveries,
        matches,
        fitted,
        as_of_date="2024-01-03",
    )
    state = features.state_id_of(0, 0, 1)

    assert ratings.batting_state_dismissals["batter-1"][state] == 1
    assert 10 < ratings.batting_state_averages["batter-1"][state] < 20
    assert ratings.bowling_state_wickets["bowler-1"][state] == 1
    assert ratings.bowling_usage["bowler-1"]["avg_overs"] == 1
    assert ratings.bowling_usage["bowler-1"]["preferred_overs"] == [0]
    assert "future-batter" not in ratings.batting_state_averages
    assert "future-bowler" not in ratings.bowling_usage


def test_state_average_adjustment_moves_run_and_wicket_probabilities() -> None:
    fitted = model.fit(*_fixture_data())
    fitted.probs[fitted.node_names.index("wicket"), 0] = 0.2
    nodes = model.build_tree()
    baseline = np.full(features.N_STATES, 10.0)
    batting_averages = np.full(features.N_STATES, 20.0)
    counts = np.full(features.N_STATES, 30.0)
    ratings = player_model.PlayerRatings(
        player_ids=[],
        batting_probs=np.empty((0, len(nodes), features.N_STATES)),
        bowling_probs=np.empty((0, len(nodes), features.N_STATES)),
        debut_batting_probs=fitted.probs,
        debut_bowling_probs=fitted.probs,
        batting_form={},
        bowling_form={},
        as_of_date=pd.Timestamp("2024-01-03"),
        batting_state_averages={"batter": batting_averages},
        batting_state_dismissals={"batter": counts},
        baseline_batting_averages=baseline,
    )

    probabilities = model.get_node_probs(
        fitted,
        0,
        batting_player_id="batter",
        player_ratings=ratings,
    )

    assert probabilities[fitted.node_names.index("runs")] > fitted.probs[
        fitted.node_names.index("runs"), 0
    ]
    assert probabilities[fitted.node_names.index("wicket")] < fitted.probs[
        fitted.node_names.index("wicket"), 0
    ]


def test_bowling_schedule_selects_historical_bowlers_and_respects_over_preferences() -> None:
    ids = [f"bowler-{index}" for index in range(6)]
    preferred = np.full(20, 0.01)
    preferred[0] = 0.8
    ratings = player_model.PlayerRatings(
        player_ids=[],
        batting_probs=np.empty((0, 0, features.N_STATES)),
        bowling_probs=np.empty((0, 0, features.N_STATES)),
        debut_batting_probs=np.empty((0, features.N_STATES)),
        debut_bowling_probs=np.empty((0, features.N_STATES)),
        batting_form={},
        bowling_form={},
        as_of_date=pd.Timestamp("2024-01-03"),
        bowling_usage={
            player_id: {
                "avg_overs": float(6 - index),
                "over_probabilities": preferred if index == 0 else np.full(20, 0.05),
                "preferred_overs": [0] if index == 0 else [1],
            }
            for index, player_id in enumerate(ids)
        },
    )

    schedule = player_model.build_bowling_schedule(ids, ratings)
    counts = pd.Series(schedule).value_counts()

    assert len(schedule) == 20
    assert set(schedule) == set(ids)
    assert counts.max() <= 4
    assert len(counts.unique()) > 1
    assert schedule[0] == "bowler-0"
    assert all(
        schedule[index] != schedule[index + 1]
        for index in range(len(schedule) - 1)
    )


def test_bowling_schedule_fills_to_five_when_history_is_sparse() -> None:
    ids = [f"bowler-{index}" for index in range(7)]
    ratings = player_model.PlayerRatings(
        player_ids=[],
        batting_probs=np.empty((0, 0, features.N_STATES)),
        bowling_probs=np.empty((0, 0, features.N_STATES)),
        debut_batting_probs=np.empty((0, features.N_STATES)),
        debut_bowling_probs=np.empty((0, features.N_STATES)),
        batting_form={},
        bowling_form={},
        as_of_date=pd.Timestamp("2024-01-03"),
        bowling_usage={
            "bowler-4": {
                "avg_overs": 3.0,
                "over_probabilities": np.full(20, 0.05),
                "preferred_overs": [0],
            },
            "bowler-6": {
                "avg_overs": 2.0,
                "over_probabilities": np.full(20, 0.05),
                "preferred_overs": [1],
            },
        },
    )

    schedule = player_model.build_bowling_schedule(ids, ratings)

    assert len(set(schedule)) == 5
    assert "bowler-4" in schedule
    assert "bowler-6" in schedule


def test_debut_prior_uses_only_training_player_first_matches(monkeypatch) -> None:
    deliveries, matches = _fixture_data()
    monkeypatch.setattr(model, "TRAIN_CUTOFF_DATE", "2024-01-03")
    fitted = model.fit(deliveries, matches)
    runs_index = fitted.node_names.index("runs")
    baseline = fitted.probs[runs_index, 0]
    debut_prior = fitted.debut_batting_probs[runs_index, 0]

    assert debut_prior < baseline
    ratings = player_model.build_player_ratings(
        deliveries,
        matches,
        fitted,
        as_of_date="2024-01-01",
    )
    np.testing.assert_array_equal(
        ratings.probabilities("batting", "brand-new-player", 0),
        fitted.debut_batting_probs[:, 0],
    )


def test_team_form_guide_excludes_future_matches() -> None:
    deliveries, matches = _fixture_data()

    guide = player_model.team_form_guide(
        deliveries,
        matches,
        as_of_date="2024-01-03",
    )

    assert guide.set_index("team").loc["A", "matches"] == 2
    assert guide.set_index("team").loc["A", "avg_runs_scored"] == 0.5


def test_player_ratings_probabilities_return_debut_prior_for_unseen_player() -> None:
    nodes = model.build_tree()
    baseline = np.full((len(nodes), features.N_STATES), 0.5)
    ratings = player_model.PlayerRatings(
        player_ids=[],
        batting_probs=np.empty((0, len(nodes), features.N_STATES)),
        bowling_probs=np.empty((0, len(nodes), features.N_STATES)),
        debut_batting_probs=baseline,
        debut_bowling_probs=baseline,
        batting_form={},
        bowling_form={},
        as_of_date=pd.Timestamp("2024-01-01"),
    )

    np.testing.assert_array_equal(
        ratings.probabilities("batting", "new-player", 0),
        baseline[:, 0],
    )
