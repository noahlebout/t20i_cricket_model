import numpy as np
import pandas as pd
import pytest

from src import features, model


def test_fit_learns_smoothed_batting_and_bowling_team_probabilities(tmp_path) -> None:
    rows = []
    for index in range(20):
        batting_strong = index < 10
        rows.append(
            {
                "match_id": "old",
                "innings_number": 1,
                "legal_balls_before": 0,
                "wickets_before": 0,
                "batting_team": "BatStrong" if batting_strong else "BatWeak",
                "bowling_team": "BowlWeak" if batting_strong else "BowlStrong",
                "is_wicket": not batting_strong and index == 10,
                "is_legal": True,
                "wides": 0,
                "total_runs": 6 if batting_strong else 0,
            }
        )
    deliveries = pd.DataFrame(rows)
    matches = pd.DataFrame(
        {"match_id": ["old"], "date": ["2024-01-01"], "venue": ["Ground A"]}
    )

    fitted = model.fit(deliveries, matches)

    bat_strong_probs = model.get_node_probs(fitted, 0, batting_team="BatStrong")
    bat_weak_probs = model.get_node_probs(fitted, 0, batting_team="BatWeak")
    bowling_strong_probs = model.get_node_probs(fitted, 0, bowling_team="BowlStrong")
    bowling_weak_probs = model.get_node_probs(fitted, 0, bowling_team="BowlWeak")
    runs_index = fitted.node_names.index("runs")
    wicket_index = fitted.node_names.index("wicket")

    assert bat_strong_probs[runs_index] > bat_weak_probs[runs_index]
    assert bowling_strong_probs[wicket_index] > bowling_weak_probs[wicket_index]
    assert bat_strong_probs[wicket_index] == fitted.probs[wicket_index, 0]
    assert bowling_strong_probs[runs_index] == fitted.probs[runs_index, 0]

    path = tmp_path / "team_model.npz"
    model.save(fitted, path)
    loaded = model.load(path)
    assert loaded.team_names == fitted.team_names
    np.testing.assert_array_equal(loaded.batting_team_probs, fitted.batting_team_probs)
    np.testing.assert_array_equal(loaded.bowling_team_probs, fitted.bowling_team_probs)
    np.testing.assert_array_equal(loaded.illegal_run_probs, fitted.illegal_run_probs)
    assert loaded.venue_names == fitted.venue_names
    np.testing.assert_array_equal(loaded.venue_probs, fitted.venue_probs)


def test_unknown_team_is_reported() -> None:
    nodes = model.build_tree()
    fitted = model.NodeProbs(
        probs=np.full((len(nodes), features.N_STATES), 0.5),
        node_names=[node.name for node in nodes],
        metadata={},
    )

    with pytest.raises(ValueError, match="not fitted with team data"):
        model.get_node_probs(fitted, 0, batting_team="Unknown")


def test_fit_learns_distinct_wide_and_no_ball_total_run_distributions() -> None:
    rows = []
    for _ in range(40):
        rows.extend(
            [
                {
                    "match_id": "old",
                    "innings_number": 1,
                    "legal_balls_before": 0,
                    "wickets_before": 0,
                    "is_wicket": False,
                    "is_legal": False,
                    "wides": 1,
                    "noballs": 0,
                    "total_runs": 2,
                },
                {
                    "match_id": "old",
                    "innings_number": 1,
                    "legal_balls_before": 0,
                    "wickets_before": 0,
                    "is_wicket": False,
                    "is_legal": False,
                    "wides": 0,
                    "noballs": 1,
                    "total_runs": 5,
                },
            ]
        )
    deliveries = pd.DataFrame(rows)
    matches = pd.DataFrame({"match_id": ["old"], "date": ["2024-01-01"]})

    fitted = model.fit(deliveries, matches)

    assert fitted.illegal_run_probs is not None
    assert fitted.illegal_run_probs.shape == (2, features.N_STATES, 5)
    assert fitted.illegal_run_probs[0, 0, 1] == pytest.approx(1)
    assert fitted.illegal_run_probs[1, 0, 4] == pytest.approx(1)


def test_fit_learns_venue_effects_and_rejects_unseen_venue() -> None:
    rows = []
    for index in range(40):
        for match_id, venue, runs in (
            ("match-a", "Ground A", 6),
            ("match-b", "Ground B", 0),
        ):
            rows.append(
                {
                    "match_id": match_id,
                    "innings_number": 1,
                    "legal_balls_before": 0,
                    "wickets_before": 0,
                    "is_wicket": False,
                    "is_legal": True,
                    "wides": 0,
                    "total_runs": runs,
                }
            )
    deliveries = pd.DataFrame(rows)
    matches = pd.DataFrame(
        {
            "match_id": ["match-a", "match-b"],
            "date": ["2024-01-01", "2024-01-02"],
            "venue": ["Ground A", "Ground B"],
        }
    )

    fitted = model.fit(deliveries, matches)

    assert fitted.venue_probs is not None
    runs_index = fitted.node_names.index("runs")
    ground_a = model.get_node_probs(fitted, 0, venue="ground a")
    ground_b = model.get_node_probs(fitted, 0, venue="Ground B")
    assert ground_a[runs_index] > ground_b[runs_index]
    with pytest.raises(ValueError, match="Unknown venue"):
        model.get_node_probs(fitted, 0, venue="Unseen Ground")
