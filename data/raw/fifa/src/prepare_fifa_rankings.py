#!/usr/bin/env python3
"""Normalize FIFA men's world ranking data into a standard CSV format.

This script accepts a raw FIFA ranking export with common column aliases and
produces a clean English-language CSV for the training-set builder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COLUMN_ALIASES = {
    "ranking_date": ["ranking_date", "date", "update_date", "published_date", "last_update_date"],
    "team_name": ["team_name", "team", "country", "nation", "association"],
    "rank": ["rank", "ranking", "position", "rk"],
    "ranking_points": ["ranking_points", "points", "point", "score", "total_points"],
    "rank_change": ["rank_change", "change", "position_change"],
    "points_change": ["points_change", "points_delta", "point_change", "points_diff", "points_delta_total"],
    "country_code": ["country_code", "code", "iso_code", "fifa_code"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FIFA ranking CSV.")
    parser.add_argument("--input", required=True, help="Raw FIFA ranking CSV.")
    parser.add_argument("--output", required=True, help="Clean output CSV.")
    return parser.parse_args()


def canonicalize(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def rename_by_aliases(df: pd.DataFrame) -> pd.DataFrame:
    current = {canonicalize(col): col for col in df.columns}
    rename_map = {}
    for target, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            col = current.get(canonicalize(alias))
            if col is not None:
                rename_map[col] = target
                break
    return df.rename(columns=rename_map)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)
    df = rename_by_aliases(df)

    required = ["ranking_date", "team_name", "rank", "ranking_points"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Expected one of the common aliases listed in {input_path.name}."
        )

    df["ranking_date"] = pd.to_datetime(df["ranking_date"], errors="coerce")
    df = df.dropna(subset=["ranking_date"]).copy()
    df["ranking_date"] = df["ranking_date"].dt.strftime("%Y-%m-%d")
    df["team_name"] = df["team_name"].astype(str).str.strip()
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["ranking_points"] = pd.to_numeric(df["ranking_points"], errors="coerce")
    df = df.dropna(subset=["team_name", "rank", "ranking_points"]).copy()
    df["rank"] = df["rank"].astype(int)

    keep_cols = ["ranking_date", "team_name", "rank", "ranking_points"]
    for extra in ["rank_change", "points_change", "country_code"]:
        if extra in df.columns:
            keep_cols.append(extra)

    df = df[keep_cols].sort_values(["ranking_date", "rank", "team_name"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
