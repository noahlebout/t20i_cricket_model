"""Ingest Cricsheet T20I Json files into 3 parquet tables: matches, deliveries, and players.

Inputs:
    data/raw/t20is_json/*.json          (one JSON file per match)

outputs:
    data/processed/matches.parquet      (one row per match)
    data/processed/deliveries.parquet   (one row per delivery)
    data/processed/players.parquet      (one row per player)
    
    
Design:
parse_match() pure parsing of One file, no filtering
build_tables() loops ovre all files, builds dataframes once
filter_matches() applies the scope rules (gender, date, etc) to the matches table
validate() sanity check;
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "t20is_json"
OUTPUT_DIR = ROOT / "data" / "processed"


FILTERS = {
    "gender": "male",  # only ingest men's matches
    "team_type": "international",  # only ingest international matches
    "match_type": {"T20", "IT20"},    # only ingest T20I matches
    "overs": 20,  # only ingest 20-over matches
}

NON_DISMISSAL_KINDS = {"retired hurt", "retired not out"}

def parse_match(path: Path) -> tuple[dict, list[dict], list[dict]]:
    """Parse a single match JSON file into 3 tables: match, deliveries, players.

    Returns:
        match: dict of match-level info (one row)
        deliveries: list of dicts, one per delivery
        players: list of dicts, one per player
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # --- Match-level info ------------------------------------------------------
    info = data["info"]
    match_id = path.stem
    registry = info.get("registry", {}).get("people", {})
    teams = info.get("teams", [])
    outcome = info.get("outcome", {})
    by = outcome.get("by", {})
    toss = info.get("toss", {})
    dates = info.get("dates", [])
    innings_list = data.get("innings", [])
    regular_innings = [i for i in innings_list if not i.get("super_over")]

    match_row = {
        "match_id": match_id,
        "date": dates[0] if dates else None,
        "gender": info.get("gender"),
        "match_type": info.get("match_type"),
        "team_type": info.get("team_type"),
        "season": str(info.get("season")) if info.get("season") is not None else None,
        "event_name": info.get("event", {}).get("name"),
        "venue": info.get("venue"),
        "city": info.get("city"),
        "team_1": teams[0] if len(teams) > 0 else None,
        "team_2": teams[1] if len(teams) > 1 else None,
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "overs": info.get("overs"),
        "winner": outcome.get("winner"),
        "result": outcome.get("result"),
        "by_runs": by.get("runs"),
        "by_wickets": by.get("wickets"),
        "method": outcome.get("method"),
        "eliminator": outcome.get("eliminator"),
        "n_innings": len(regular_innings),
        "has_super_over": len(innings_list) != len(regular_innings),
        "first_batting_team": regular_innings[0]["team"] if regular_innings else None,
    }

    delivery_rows: list[dict] = []
    for innings_number, inn in enumerate(innings_list, start=1):
        batting_team = inn["team"]
        bowling_team = next(t for t in teams if t != batting_team)
        is_super_over = inn.get("super_over", False)
        target = inn.get("target",{})

        runs_before = 0
        wickets_before = 0
        legal_before = 0
        for over in inn.get("overs",{}):
            over_no = over["over"]
            for ball_idx, d in enumerate(over["deliveries"]):
                runs = d["runs"]
                extras = d.get("extras", {})
                is_legal = ("wides" not in extras) and ("noballs" not in extras)

                dismissals = [
                    w for w in d.get("wickets",[]) if w["kind"] not in NON_DISMISSAL_KINDS
                ]

                first_wicket = dismissals[0] if dismissals else None

                delivery_rows.append(
                    {
                        "match_id": match_id,
                        "innings_number": innings_number,
                        "batting_team": batting_team,
                        "bowling_team": bowling_team,
                        "is_super_over": is_super_over,
                        
                        "runs_before": runs_before,
                        
                        "over": over_no,
                        "ball_in_over": ball_idx,
                        "legal_balls_before": legal_before,
                        "wickets_before": wickets_before,
                        "target_runs": target.get("runs"),
                        "batter": d["batter"],
                        "bowler": d["bowler"],
                        "non_striker": d["non_striker"],
                        "batter_id": registry.get(d["batter"]),
                        "bowler_id": registry.get(d["bowler"]),
                        "non_striker_id": registry.get(d["non_striker"]),
                        "batter_runs": runs["batter"],
                        "extras_runs": runs["extras"],
                        "total_runs": runs["total"],
                        "wides": extras.get("wides", 0),
                        "noballs": extras.get("noballs", 0),
                        "byes": extras.get("byes", 0),
                        "legbyes": extras.get("legbyes", 0),
                        "penalty": extras.get("penalty", 0),
                        "is_legal": is_legal,
                        "is_wicket": first_wicket is not None,
                        "wicket_kind": first_wicket["kind"] if first_wicket else None,
                        "player_out": first_wicket["player_out"] if first_wicket else None,
                    }
                )

                runs_before += runs["total"]
                wickets_before += len(dismissals)
                legal_before += int(is_legal)


    player_rows = [
        {
            "match_id": match_id,
            "date": match_row["date"],
            "team": team,
            "name": name,
            "player_id": registry.get(name),
        }
        for team, names in info.get("players", {}).items()
        for name in names
    ]


    return match_row, delivery_rows, player_rows



def build_tables(
        raw_dir:Path = RAW_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the 3 tables from all JSON files in raw_dir.

    Returns:
        matches: DataFrame of match-level info (one row per match)
        deliveries: DataFrame of delivery-level info (one row per delivery)
        players: DataFrame of player-level info (one row per player)
    """
    files = sorted(raw_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {raw_dir}")

    match_rows, delivery_rows, player_rows = [], [], []
    failures: list[tuple[str, str]] = []

    for path in tqdm(files, desc="Parsing JSON files", unit="match"):
        try:
            m, d, p = parse_match(path)
        except (KeyError, ValueError, TypeError) as e:
            failures.append((path.name, str(e)))
            continue
        match_rows.append(m)
        delivery_rows.extend(d)
        player_rows.extend(p)

    matches = pd.DataFrame(match_rows)
    deliveries = pd.DataFrame(delivery_rows)
    player_matches = pd.DataFrame(player_rows)

    matches["date"] = pd.to_datetime(matches["date"])
    player_matches["date"] = pd.to_datetime(player_matches["date"])
    return matches, deliveries, player_matches, failures


def filter_matches(matches: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Apply rules in order. Returns (kept_matches, [(rule, n_dropped)...])"""
    log: list[tuple[str, int]] = []
    kept = matches

    def apply(rule_name: str, mask: pd.Series) -> None:
        nonlocal kept
        log.append((rule_name, int((~mask).sum())))
        kept = kept[mask]

    apply(f"gender == {FILTERS['gender']}", kept["gender"] == FILTERS["gender"])
    apply(f"team_type == {FILTERS['team_type']}", kept["team_type"] == FILTERS["team_type"])
    apply(f"match_type in {FILTERS['match_type']}", kept["match_type"].isin(FILTERS["match_type"]))
    apply("has a winner or is a tie (drops no-results / abandonded)", kept["winner"].notna() | (kept["result"] == "tie"))
    apply("not rain-affected (drops rain-affected matches)", kept["method"].isna())
    apply(f"scheduled overs == {FILTERS['overs']}", kept["overs"] == FILTERS["overs"])
    apply("exactly 2 innings", kept["n_innings"] == 2)

    return kept.copy(), log


def exclude_invalid_matches(
    matches: pd.DataFrame, deliveries: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Exclude matches with innings longer than the configured limits."""
    innings = (
        deliveries.groupby(["match_id", "innings_number"], observed=True)
        .agg(
            legal_balls=("is_legal", "sum"),
            wickets=("is_wicket", "sum"),
        )
    )
    invalid_ids = set(
        innings.index[
            (innings["legal_balls"] > FILTERS["overs"] * 6)
            | (innings["wickets"] > 10)
        ].get_level_values("match_id")
    )
    return (
        matches.loc[~matches["match_id"].isin(invalid_ids)].copy(),
        deliveries.loc[~deliveries["match_id"].isin(invalid_ids)].copy(),
        sorted(invalid_ids),
    )


def build_players(player_matches:pd.DataFrame, kept_match_ids: set[str]) -> pd.DataFrame:
    pm = player_matches[
        player_matches["match_id"].isin(kept_match_ids) & (player_matches["player_id"].notna())
    ]
    grouped = pm.groupby("player_id")
    players = pd.DataFrame(
        {
            "name": grouped["name"].agg(lambda x: x.mode().iloc[0]),
            "first_match_date": grouped["date"].min(),
            "last_match_date": grouped["date"].max(),
            "matches_played": grouped["match_id"].nunique(),
            "teams": grouped["team"].agg(lambda x: sorted(set(x)))
        }
    ).reset_index()
    return players

def set_dtypes(matches: pd.DataFrame, deliveries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Set dtypes for the 2 tables. Modifies in place."""
    for col in ["gender", "match_type", "team_type", "event_name", "venue", "city", "team_1", "team_2", "toss_winner", "toss_decision", "winner", "result", "method", "first_batting_team"]:
        matches[col] = matches[col].astype("category")

    for col in ["batting_team", "bowling_team", "batter", "bowler", "non_striker",
                "batter_id", "bowler_id", "non_striker_id", "wicket_kind", "player_out"]:
        deliveries[col] = deliveries[col].astype("category")
 
    small_ints = {
        "innings_number": "int8", "over": "int8", "ball_in_over": "int8",
        "legal_balls_before": "int16", "runs_before": "int16", "wickets_before": "int8",
        "batter_runs": "int8", "extras_runs": "int8", "total_runs": "int8",
        "wides": "int8", "noballs": "int8", "byes": "int8", "legbyes": "int8", "penalty": "int8",
    }
    deliveries = deliveries.astype(small_ints)
    deliveries["target_runs"] = deliveries["target_runs"].astype("Int16")
    return matches, deliveries

def validate(matches: pd.DataFrame, deliveries: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run sanity checks. Returns {check_name: offending_rows}; empty frame = pass."""
    problems: dict[str, pd.DataFrame] = {}
 
    inn = (
        deliveries.groupby(["match_id", "innings_number"], observed=True)
        .agg(runs=("total_runs", "sum"), wickets=("is_wicket", "sum"), legal=("is_legal", "sum"))
        .reset_index()
    )
 
    per_match = inn.groupby("match_id").size()
    problems["match does not have exactly 2 innings"] = per_match[per_match != 2].to_frame("n_innings")
    problems["innings with more than 120 legal balls"] = inn[inn["legal"] > FILTERS["overs"] * 6]
    problems["innings with more than 10 wickets"] = inn[inn["wickets"] > 10]
 
    key_cols = ["match_id", "batting_team", "bowling_team", "batter_id", "bowler_id"]
    problems["nulls in key columns"] = deliveries[deliveries[key_cols].isna().any(axis=1)]
 
    problems["total_runs != batter_runs + extras_runs"] = deliveries[
        deliveries["total_runs"] != deliveries["batter_runs"] + deliveries["extras_runs"]
    ]
 
    # Winner consistent with runs scored (skip ties / super overs / eliminators).
    wide = inn.pivot(index="match_id", columns="innings_number", values="runs")
    wide.columns = ["runs_1", "runs_2"]
    m = matches.set_index("match_id").join(wide, how="inner")
    decided = m[m["winner"].notna() & m["result"].isna() & m["eliminator"].isna() & ~m["has_super_over"]]
    # Cast to str: categorical columns with different category sets can't be compared directly.
    bat_first_won = decided["winner"].astype(str) == decided["first_batting_team"].astype(str)
    inconsistent = (bat_first_won & (decided["runs_1"] <= decided["runs_2"])) | (
        ~bat_first_won & (decided["runs_2"] <= decided["runs_1"])
    )
    problems["winner inconsistent with runs scored"] = decided[inconsistent][
        ["winner", "first_batting_team", "runs_1", "runs_2"]
    ]
 
    return problems

 
def run_ingestion() -> int:
    matches, deliveries, player_matches, failures = build_tables()
    print(f"\nParsed {len(matches)} matches, {len(deliveries):,} deliveries")
    if failures:
        print(f"\n{len(failures)} files failed to parse (first 5):")
        for name, err in failures[:5]:
            print(f"  {name}: {err}")
 
    kept, log = filter_matches(matches)
    print("\nFilter log (rule -> matches dropped):")
    for rule, n in log:
        print(f"  {n:>6}  {rule}")
    print(f"  kept {len(kept)} of {len(matches)} matches")
 
    kept_ids = set(kept["match_id"])
    deliveries = deliveries[
        deliveries["match_id"].isin(kept_ids) & ~deliveries["is_super_over"]
    ].drop(columns="is_super_over")

    kept, deliveries, invalid_ids = exclude_invalid_matches(kept, deliveries)
    if invalid_ids:
        print(
            f"\nExcluded {len(invalid_ids)} matches with an innings exceeding "
            f"{FILTERS['overs'] * 6} legal balls or 10 wickets: "
            f"{', '.join(invalid_ids[:10])}"
            + (" ..." if len(invalid_ids) > 10 else "")
        )
        kept_ids = set(kept["match_id"])
    players = build_players(player_matches, kept_ids)

    kept, deliveries = set_dtypes(kept.reset_index(drop=True), deliveries.reset_index(drop=True))
 
    problems = validate(kept, deliveries)
    print("\nValidation:")
    failed = False
    for name, df in problems.items():
        status = "ok  " if df.empty else "FAIL"
        failed |= not df.empty
        print(f"  [{status}] {name}" + ("" if df.empty else f"  ({len(df)} offenders)"))
        if not df.empty:
            print(df.head(3).to_string(max_cols=8))
 
    if failed:
        print("\nValidation failed; processed tables were not written.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(OUTPUT_DIR / "matches.parquet", index=False)
    deliveries.to_parquet(OUTPUT_DIR / "deliveries.parquet", index=False)
    players.to_parquet(OUTPUT_DIR / "players.parquet", index=False)
    print(f"\nWrote matches ({len(kept)}), deliveries ({len(deliveries):,}), players ({len(players)}) to {OUTPUT_DIR}")
 
    return 0


def main() -> int:
    return run_ingestion()


if __name__ == "__main__":
    sys.exit(main())