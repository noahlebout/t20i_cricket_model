import pandas as pd
import pytest

from run_simulator import (
    _parse_batting_order,
    _parse_observed_delivery,
    _player_id_overrides,
    _project_bowling_schedule,
    _resolve_player_ids,
)


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ("4", (4, False, True, "four", None)),
        ("W", (0, True, True, "wicket", 0)),
        ("W:1", (1, True, True, "wicket+1", None)),
        ("WD:2:1", (2, False, False, "wide+2", 1)),
        ("NB:3:1", (3, False, False, "no_ball+3", 1)),
    ],
)
def test_parse_actual_delivery(entry, expected) -> None:
    assert _parse_observed_delivery(entry) == expected


@pytest.mark.parametrize("entry", ["WD:0", "NB:x", "WD:2:-1", "nonsense"])
def test_invalid_actual_delivery_is_rejected(entry) -> None:
    with pytest.raises(ValueError):
        _parse_observed_delivery(entry)


def test_parse_quit_command() -> None:
    assert _parse_observed_delivery("q") is None


def test_parse_batting_order_uses_fixture_order_when_empty() -> None:
    players = ["Player 1", "Player 2", "Player 3"]

    assert _parse_batting_order("", players) == players


def test_parse_batting_order_reorders_by_number_without_names() -> None:
    players = ["Player 1", "Player 2", "Player 3"]

    assert _parse_batting_order("3, 1, 2", players) == [
        "Player 3",
        "Player 1",
        "Player 2",
    ]


@pytest.mark.parametrize("value", ["1,2", "1,1,3", "1,2,4", "Player 1,2,3"])
def test_parse_batting_order_rejects_invalid_selection(value) -> None:
    with pytest.raises(ValueError):
        _parse_batting_order(value, ["Player 1", "Player 2", "Player 3"])


def test_project_bowling_schedule_respects_four_over_limit_and_rotation() -> None:
    bowlers = ["a", "b", "c", "d", "e"]
    schedule = _project_bowling_schedule(
        bowlers,
        tuple(bowlers[index % len(bowlers)] for index in range(20)),
        ["a", "b", "c"],
    )

    assert len(schedule) == 20
    assert max(pd.Series(schedule).value_counts()) <= 4
    assert all(
        schedule[index] != schedule[index + 1]
        for index in range(len(schedule) - 1)
    )
    assert schedule[:3] == ("a", "b", "c")


def test_project_bowling_schedule_accepts_user_sequence_with_valid_completion() -> None:
    bowlers = ["1", "2", "3", "4", "5"]
    selected = ["1", "2", "1", "2", "1", "3", "4", "3", "5", "3", "5", "2", "5"]
    valid_schedule = (
        "1", "2", "1", "2", "1", "3", "4", "3", "5", "3",
        "5", "2", "5", "4", "5", "4", "3", "4", "1", "2",
    )

    schedule = _project_bowling_schedule(
        bowlers,
        valid_schedule,
        selected,
    )

    assert schedule == valid_schedule


def test_resolve_player_ids_accepts_a_valid_ambiguous_name_override() -> None:
    deliveries = pd.DataFrame(
        {
            "match_id": ["old", "old"],
            "batter": ["RK Singh", "Other"],
            "batter_id": ["correct-id", "other-id"],
            "bowler": ["Other", "RK Singh"],
            "bowler_id": ["other-id", "different-id"],
        }
    )
    matches = pd.DataFrame(
        {
            "match_id": ["old"],
            "date": ["2024-12-01"],
        }
    )

    resolved = _resolve_player_ids(
        ["RK Singh"],
        deliveries,
        matches,
        as_of_date=pd.Timestamp("2025-01-22"),
        player_id_overrides={"RK Singh": "correct-id"},
    )

    assert resolved == ["correct-id"]


def test_resolve_player_ids_rejects_an_override_not_in_prior_history() -> None:
    deliveries = pd.DataFrame(
        {
            "match_id": ["old"],
            "batter": ["RK Singh"],
            "batter_id": ["correct-id"],
            "bowler": ["Other"],
            "bowler_id": ["other-id"],
        }
    )
    matches = pd.DataFrame(
        {
            "match_id": ["old"],
            "date": ["2024-12-01"],
        }
    )

    with pytest.raises(ValueError, match="must match one of its historical player IDs"):
        _resolve_player_ids(
            ["RK Singh"],
            deliveries,
            matches,
            as_of_date=pd.Timestamp("2025-01-22"),
            player_id_overrides={"RK Singh": "unknown-id"},
        )


def test_player_id_overrides_must_name_a_player_in_the_batting_order() -> None:
    with pytest.raises(ValueError, match="is not in the batting_order"):
        _player_id_overrides(
            {"Unknown Player": "player-id"},
            "team1 player_ids",
            ["Known Player"],
        )
