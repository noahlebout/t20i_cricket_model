import numpy as np
import pandas as pd

from src import evaluation, features, model


def test_node_calibration_reports_scores_for_reached_rows() -> None:
    nodes = model.build_tree()
    fitted = model.NodeProbs(
        probs=np.full((len(nodes), features.N_STATES), 0.5),
        node_names=[node.name for node in nodes],
        metadata={},
    )
    rows = pd.DataFrame(
        {
            "state_id": [0, 0],
            "is_wicket": [False, True],
            "is_legal": [True, True],
            "wides": [0, 0],
            "total_runs": [0, 0],
        }
    )

    metrics = evaluation.node_calibration(fitted, rows)

    assert metrics["wicket"]["observations"] == 2
    assert metrics["wicket"]["observed_rate"] == 0.5
    assert metrics["wicket"]["brier_score"] == 0.25
    assert sum(item["count"] for item in metrics["wicket"]["reliability_bins"]) == 2
    assert metrics["legal"]["observations"] == 1
    assert metrics["runs"]["observations"] == 1


def test_node_calibration_compares_team_model_on_known_team_rows() -> None:
    nodes = model.build_tree()
    node_names = [node.name for node in nodes]
    baseline = np.full((len(nodes), features.N_STATES), 0.5)
    batting_team_probs = np.full((2, len(nodes), features.N_STATES), 0.5)
    runs_index = node_names.index("runs")
    batting_team_probs[0, runs_index, :] = 0.9
    batting_team_probs[1, runs_index, :] = 0.1
    fitted = model.NodeProbs(
        probs=baseline,
        node_names=node_names,
        metadata={},
        team_names=["BatA", "BatB"],
        batting_team_probs=batting_team_probs,
        bowling_team_probs=np.full((2, len(nodes), features.N_STATES), 0.5),
    )
    rows = pd.DataFrame(
        {
            "state_id": [0, 0],
            "is_wicket": [False, False],
            "is_legal": [True, True],
            "wides": [0, 0],
            "total_runs": [1, 0],
            "batting_team": ["BatA", "BatB"],
            "bowling_team": ["BatB", "BatA"],
        }
    )

    metrics = evaluation.node_calibration(fitted, rows)["runs"]["team_comparison"]

    assert metrics["eligible_observations"] == 2
    assert metrics["coverage"] == 1
    assert metrics["team_brier_score"] < metrics["baseline_brier_score"]


def test_illegal_run_calibration_scores_wides_and_no_balls_separately() -> None:
    nodes = model.build_tree()
    fitted = model.NodeProbs(
        probs=np.full((len(nodes), features.N_STATES), 0.5),
        node_names=[node.name for node in nodes],
        metadata={},
        illegal_run_probs=np.zeros((2, features.N_STATES, 3)),
    )
    fitted.illegal_run_probs[0, :, 0] = 1
    fitted.illegal_run_probs[1, :, 2] = 1
    heldout = pd.DataFrame(
        {
            "state_id": [0, 0],
            "is_legal": [False, False],
            "wides": [1, 0],
            "total_runs": [1, 3],
        }
    )

    metrics = evaluation.illegal_run_calibration(fitted, heldout)

    assert metrics["wide"]["observations"] == 1
    assert metrics["wide"]["actual_mean_runs"] == 1
    assert metrics["wide"]["predicted_mean_runs"] == 1
    assert metrics["no_ball"]["observations"] == 1
    assert metrics["no_ball"]["actual_mean_runs"] == 3
    assert metrics["no_ball"]["predicted_mean_runs"] == 3


def test_heldout_venue_comparison_uses_only_venues_seen_in_training() -> None:
    nodes = model.build_tree()
    node_names = [node.name for node in nodes]
    venue_probs = np.full((1, len(nodes), features.N_STATES), 0.5)
    runs_index = node_names.index("runs")
    venue_probs[0, runs_index, :] = 0.9
    fitted = model.NodeProbs(
        probs=np.full((len(nodes), features.N_STATES), 0.5),
        node_names=node_names,
        metadata={},
        venue_names=["Ground A"],
        venue_probs=venue_probs,
    )
    heldout = pd.DataFrame(
        {
            "state_id": [0, 0],
            "is_wicket": [False, False],
            "is_legal": [True, True],
            "wides": [0, 0],
            "total_runs": [1, 0],
            "venue": ["Ground A", "New Ground"],
        }
    )

    metrics = evaluation.node_calibration(fitted, heldout)["runs"]["venue_comparison"]

    assert metrics["eligible_observations"] == 1
    assert metrics["coverage"] == 0.5
    assert metrics["venue_brier_score"] < metrics["baseline_brier_score"]


def test_heldout_split_is_on_or_after_cutoff() -> None:
    deliveries = pd.DataFrame(
        {
            "match_id": ["old", "new"],
            "innings_number": [1, 1],
            "legal_balls_before": [0, 0],
            "wickets_before": [0, 0],
            "is_wicket": [False, False],
            "is_legal": [True, True],
            "wides": [0, 0],
            "total_runs": [0, 0],
        }
    )
    matches = pd.DataFrame(
        {
            "match_id": ["old", "new"],
            "date": ["2024-12-31", "2025-01-01"],
        }
    )

    heldout = evaluation.heldout_deliveries(deliveries, matches)

    assert heldout["match_id"].tolist() == ["new"]
    assert heldout["state_id"].tolist() == [0]
