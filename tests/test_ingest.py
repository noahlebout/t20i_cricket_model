import pandas as pd

from src import ingest


def test_exclude_invalid_matches_removes_all_rows_for_overlong_innings() -> None:
    matches = pd.DataFrame({"match_id": ["valid", "overlong"]})
    deliveries = pd.DataFrame(
        {
            "match_id": ["valid"] * 120 + ["overlong"] * 121,
            "innings_number": [1] * 241,
            "is_legal": [True] * 241,
            "is_wicket": [False] * 241,
        }
    )

    filtered_matches, filtered_deliveries, excluded = ingest.exclude_invalid_matches(
        matches, deliveries
    )

    assert filtered_matches["match_id"].tolist() == ["valid"]
    assert filtered_deliveries["match_id"].unique().tolist() == ["valid"]
    assert excluded == ["overlong"]
