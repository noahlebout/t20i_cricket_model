"""Run the T20 data-to-model pipeline from download through evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from download_data import (
    EXTRACT_DIR,
    RAW_DIR,
    T20I_URL,
    ZIP_PATH,
    download_file,
    extract_json_files,
    write_manifest,
)
from src import evaluation, ingest, model


def ensure_raw_data(*, refresh: bool = False) -> None:
    """Download and extract match files when missing or explicitly refreshed."""
    has_json = EXTRACT_DIR.exists() and any(EXTRACT_DIR.glob("*.json"))
    if has_json and not refresh:
        count = sum(1 for _ in EXTRACT_DIR.glob("*.json"))
        print(f"Using existing raw data: {count:,} JSON files in {EXTRACT_DIR}")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {T20I_URL}")
    download_file(T20I_URL, ZIP_PATH)
    count = extract_json_files(ZIP_PATH, EXTRACT_DIR)
    write_manifest(count, ZIP_PATH)
    print(f"Extracted {count:,} match files to {EXTRACT_DIR}")


def run_pipeline(
    *,
    refresh_data: bool = False,
    simulations: int = 1000,
    seed: int = 42,
) -> None:
    """Download, ingest, fit, save, and evaluate the model."""
    if simulations <= 0:
        raise ValueError("simulations must be positive")

    ensure_raw_data(refresh=refresh_data)

    print("\n[1/4] Ingesting and validating data")
    if ingest.run_ingestion() != 0:
        raise RuntimeError("Ingestion validation failed; model fitting was stopped")

    print("\n[2/4] Fitting and validating the model")
    deliveries = pd.read_parquet(model.PROCESSED / "deliveries.parquet")
    matches = pd.read_parquet(model.PROCESSED / "matches.parquet")
    fitted_model = model.fit(deliveries, matches)
    training = model.training_deliveries(deliveries, matches)
    model.check_model(fitted_model, training)

    print("\n[3/4] Saving the fitted model")
    model.save(fitted_model)
    print(
        f"Saved {len(fitted_model.node_names)} outcome nodes, "
        f"{len(fitted_model.team_names)} teams, {len(fitted_model.venue_names)} venues, "
        "and illegal-run distributions "
        f"to {model.MODEL_DIR / 'node_probs.npz'}"
    )

    print("\n[4/4] Evaluating on held-out matches")
    heldout = evaluation.heldout_deliveries(deliveries, matches)
    report = evaluation.evaluate(
        fitted_model,
        heldout,
        n_simulations=simulations,
        seed=seed,
    )
    evaluation.print_report(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="re-download and replace the local Cricsheet JSON archive",
    )
    parser.add_argument("--simulations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_pipeline(
        refresh_data=args.refresh_data,
        simulations=args.simulations,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
