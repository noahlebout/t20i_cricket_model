"""Download Cricsheet T20 international ball-by-ball data.

Usage:
    python download_data.py            # download + extract if not already present
    python download_data.py --force    # re-download and overwrite
    python download_data.py --people   # also fetch Cricsheet's people register

Outputs:
    data/raw/t20s_json.zip              (the downloaded archive)
    data/raw/t20is_json/*.json          (one JSON file per match)
    data/raw/download_manifest.json     (when/what was downloaded)
    data/raw/people.csv                 (only with --people)

Note: this archive contains BOTH men's and women's matches.
Filter on info["gender"] during ingest, not here.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

# --- Configuration -----------------------------------------------------------
# Check https://cricsheet.org/downloads/ if the URL ever stops working.
T20I_URL = "https://cricsheet.org/downloads/t20s_json.zip"
PEOPLE_URL = "https://cricsheet.org/register/people.csv"

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "t20s_json.zip"
EXTRACT_DIR = RAW_DIR / "t20is_json"
PEOPLE_PATH = RAW_DIR / "people.csv"
MANIFEST_PATH = RAW_DIR / "download_manifest.json"

HEADERS = {"User-Agent": "t20-win-predictor (personal research project)"}
TIMEOUT = 30  # seconds
CHUNK_SIZE = 1024 * 64


# --- Helpers -----------------------------------------------------------------
def download_file(url: str, dest: Path) -> None:
    """Stream a URL to dest, via a temp file so a failed download never leaves a corrupt file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=TIMEOUT) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0)) or None
            with open(tmp, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=dest.name
            ) as bar:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def extract_json_files(zip_path: Path, out_dir: Path) -> int:
    """Extract only the .json match files (flattened) into out_dir. Returns the count."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise zipfile.BadZipFile(f"Corrupt member in archive: {bad}")

        members = [m for m in zf.namelist() if m.lower().endswith(".json")]
        for name in tqdm(members, desc="Extracting", unit="file"):
            target = out_dir / Path(name).name  # flatten; also guards against path traversal
            with zf.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    return count


def write_manifest(n_files: int, zip_path: Path) -> None:
    manifest = {
        "source_url": T20I_URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "zip_size_bytes": zip_path.stat().st_size,
        "n_json_files": n_files,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


# --- Main --------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Download Cricsheet T20I data.")
    parser.add_argument("--force", action="store_true", help="re-download even if data exists")
    parser.add_argument("--people", action="store_true", help="also download people.csv")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    already_have = EXTRACT_DIR.exists() and any(EXTRACT_DIR.glob("*.json"))
    if already_have and not args.force:
        n = sum(1 for _ in EXTRACT_DIR.glob("*.json"))
        print(f"Data already present ({n} match files in {EXTRACT_DIR}). Use --force to refresh.")
    else:
        print(f"Downloading {T20I_URL}")
        try:
            download_file(T20I_URL, ZIP_PATH)
        except requests.RequestException as e:
            print(f"Download failed: {e}", file=sys.stderr)
            return 1

        try:
            n = extract_json_files(ZIP_PATH, EXTRACT_DIR)
        except zipfile.BadZipFile as e:
            print(f"Archive problem: {e}. Try again with --force.", file=sys.stderr)
            return 1

        write_manifest(n, ZIP_PATH)
        print(f"Extracted {n} match files to {EXTRACT_DIR}")

    if args.people:
        if PEOPLE_PATH.exists() and not args.force:
            print(f"{PEOPLE_PATH.name} already present. Use --force to refresh.")
        else:
            print(f"Downloading {PEOPLE_URL}")
            try:
                download_file(PEOPLE_URL, PEOPLE_PATH)
            except requests.RequestException as e:
                print(f"People download failed: {e}", file=sys.stderr)
                return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())