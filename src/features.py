"""Label every Delivery with exaclty one outcome class
This module is the single source of truth for:
    - the list of outcome clsasses and their integer codes
    - what each outcome does to the match
    - the decision tree that turns a delivery row into a class

Reads: data/processed/deliveries.parquet
Writes: data/processed/deliveries_labeled.parquet

Columns used for deliveries:
    is_wicket, is_legal, wides, noballs, total_runs
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"


PHASE_STARTS = [0, 36, 90]
PHASE_NAMES = ["powerplay", "middle", "death"]
N_PHASES = len(PHASE_STARTS)


N_WICKET_BUCKETS = 10

N_INNINGS = 2
N_STATES = N_PHASES * N_WICKET_BUCKETS * N_INNINGS

def phase_of(legal_balls_before: int) -> int:
    """Return the phase index (0, 1, 2) for a given number of legal balls bowled."""
    return np.searchsorted(PHASE_STARTS, legal_balls_before, side="right") - 1

def wickets_bucket_of(wickets_before: int) -> int:
    """Return the wicket bucket index (0 to 9) for a given number of wickets fallen."""
    return np.minimum(wickets_before, N_WICKET_BUCKETS - 1)

def state_id_of(legal_balls_before: int, wickets_before: int, innings_number: int) -> int:
    """Return a unique state ID for the given match state."""
    phase = phase_of(legal_balls_before)
    wickets_bucket = wickets_bucket_of(wickets_before)
    return (innings_number - 1) * (N_PHASES * N_WICKET_BUCKETS) + (phase * N_WICKET_BUCKETS) + wickets_bucket


def add_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add state features to the deliveries DataFrame."""
    df = df.copy()
    df["phase"] = df["legal_balls_before"].apply(phase_of)
    df["wickets_bucket"] = df["wickets_before"].apply(wickets_bucket_of)
    df["state_id"] = df.apply(
        lambda row: state_id_of(row["legal_balls_before"], row["wickets_before"], row["innings_number"]),
        axis=1
    )
    return df

def state_sample_size(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with the sample size for each state."""
    return df.groupby("state_id").size().reset_index(name="sample_size")

def check_features(before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Check that the features have been added correctly."""
    assert "phase" in after.columns, "Missing 'phase' column"
    assert "wickets_bucket" in after.columns, "Missing 'wickets_bucket' column"
    assert "state_id" in after.columns, "Missing 'state_id' column"
    assert len(before) == len(after), "Row count mismatch after adding features"


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "deliveries.parquet")
    out = add_state_features(df)
    check_features(df, out)

    sizes = state_sample_size(out)
    print("Smallest states:")
    print(sizes.head(15).to_string(index=False))
    print(f"\n{len(sizes)} non-empty states of {N_STATES} possible")


if __name__ == "__main__":
    main()