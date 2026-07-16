#!/usr/bin/env python3
"""Build English-language football training sets.

The script is designed to be data-source agnostic while still supporting the
most common Kaggle, FIFA, and FBref export shapes.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


MATCH_COLUMN_ALIASES = {
    "date": ["date", "match_date", "game_date"],
    "home_team": ["home_team", "home", "team_home"],
    "away_team": ["away_team", "away", "team_away"],
    "home_score": ["home_score", "home_goals", "home_result"],
    "away_score": ["away_score", "away_goals"],
    "competition": ["competition", "tournament", "event"],
    "stage": ["stage", "round", "phase"],
    "city": ["city", "venue_city"],
    "country": ["country", "venue_country"],
    "neutral": ["neutral", "is_neutral", "neutral_site"],
    "shootout": ["shootout", "penalties", "penalty_shootout"],
}

FIFA_COLUMN_ALIASES = {
    "ranking_date": ["ranking_date", "date", "update_date", "published_date"],
    "team_name": ["team_name", "team", "country", "nation"],
    "rank": ["rank", "ranking", "position"],
    "ranking_points": ["ranking_points", "points", "score"],
}

TEAM_FBREB_COLUMN_ALIASES = {
    "date": ["date", "match_date", "game_date"],
    "team": ["team", "squad", "club"],
    "opponent": ["opponent", "opp"],
    "is_home": ["is_home", "home", "venue_home"],
    "minutes": ["minutes", "min"],
    "goals": ["goals", "g"],
    "assists": ["assists", "a"],
    "shots": ["shots", "sh"],
    "shots_on_target": ["shots_on_target", "sot", "shots_on_tgt"],
    "passes_completed": ["passes_completed", "cmp", "passes_comp"],
    "passes_attempted": ["passes_attempted", "att", "passes_att"],
    "key_passes": ["key_passes", "kp"],
    "tackles": ["tackles", "tkl"],
    "interceptions": ["interceptions", "int"],
    "blocks": ["blocks", "blk"],
    "saves": ["saves", "sv"],
    "clean_sheet": ["clean_sheet", "cs"],
}

PLAYER_FBREB_COLUMN_ALIASES = {
    "date": ["date", "match_date", "game_date"],
    "team": ["team", "squad", "club"],
    "player": ["player", "name", "athlete"],
    "position": ["position", "pos"],
    "minutes": ["minutes", "min"],
    "started": ["started", "starter", "is_starting"],
    "goals": ["goals", "g"],
    "assists": ["assists", "a"],
    "shots": ["shots", "sh"],
    "shots_on_target": ["shots_on_target", "sot", "shots_on_tgt"],
    "passes_completed": ["passes_completed", "cmp", "passes_comp"],
    "passes_attempted": ["passes_attempted", "att", "passes_att"],
    "key_passes": ["key_passes", "kp"],
    "tackles": ["tackles", "tkl"],
    "interceptions": ["interceptions", "int"],
    "blocks": ["blocks", "blk"],
    "saves": ["saves", "sv"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build football training sets.")
    parser.add_argument("--matches", required=True, help="Historical matches CSV.")
    parser.add_argument("--fifa-rankings", required=True, help="FIFA rankings CSV.")
    parser.add_argument(
        "--ranking-date",
        help="Fallback ranking snapshot date when the FIFA file does not include ranking_date.",
    )
    parser.add_argument("--fbref-team-match", help="Optional FBref team-match CSV.")
    parser.add_argument("--fbref-player-match", help="Optional FBref player-match CSV.")
    parser.add_argument("--team-aliases", help="Optional team alias mapping CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for processed files.")
    return parser.parse_args()


def canonicalize_name(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_aliases(path: Optional[str]) -> dict[str, str]:
    if not path:
        return {}
    df = pd.read_csv(path)
    if {"source_name", "canonical_name"}.issubset(df.columns):
        source_col = "source_name"
        canonical_col = "canonical_name"
    elif {"current", "former"}.issubset(df.columns):
        source_col = "former"
        canonical_col = "current"
    else:
        raise ValueError(
            "Team alias file must contain either source_name/canonical_name or current/former columns."
        )
    aliases = {}
    for src, dst in zip(df[source_col], df[canonical_col], strict=False):
        src_key = canonicalize_name(src)
        dst_val = str(dst).strip()
        if src_key and dst_val:
            aliases[src_key] = dst_val
        dst_key = canonicalize_name(dst)
        if dst_key and dst_val:
            aliases[dst_key] = dst_val
    return aliases


def normalize_team_name(value: object, aliases: dict[str, str]) -> str:
    key = canonicalize_name(value)
    if not key:
        return ""
    return aliases.get(key, str(value).strip())


def normalized_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def stage_to_score(stage: object, competition: object) -> int:
    text = f"{normalized_text(stage)} {normalized_text(competition)}".lower()
    order = [
        ("friendly", 0),
        ("group", 1),
        ("qualif", 1),
        ("round of 32", 2),
        ("round of 16", 2),
        ("knockout", 2),
        ("quarter", 3),
        ("semi", 4),
        ("third place", 5),
        ("final", 6),
    ]
    for needle, score in order:
        if needle in text:
            return score
    return 1 if text else 0


def competition_importance(competition: object, stage: object, neutral: object) -> int:
    text = f"{normalized_text(competition)} {normalized_text(stage)}".lower()
    if "friendly" in text:
        return 10
    if "world cup" in text and any(term in text for term in ["quarter", "semi", "third place", "final"]):
        return 40
    if "world cup" in text:
        return 30
    if "qualif" in text or "nations league" in text:
        return 25
    if "continental" in text or "copa america" in text or "euros" in text or "afcon" in text:
        return 20
    return 15


def resolve_home_advantage(competition: object, neutral: object) -> int:
    competition_text = normalized_text(competition).lower()
    if normalize_bool_series(pd.Series([neutral])).iloc[0]:
        return 0
    if "world cup" in competition_text:
        return 0
    return 50


def compute_elo_features(matches: pd.DataFrame) -> pd.DataFrame:
    ratings: dict[str, float] = {}
    rows = []
    ordered = matches.sort_values(["date", "match_id"]).copy()
    for row in ordered.itertuples(index=False):
        home = row.home_team
        away = row.away_team
        home_rating = ratings.get(home, 1500.0)
        away_rating = ratings.get(away, 1500.0)
        home_adv = resolve_home_advantage(getattr(row, "competition", ""), getattr(row, "neutral", False))
        importance = competition_importance(getattr(row, "competition", ""), getattr(row, "stage", ""), getattr(row, "neutral", False))
        k_factor = max(10.0, importance)
        expected_home = 1.0 / (1.0 + 10 ** (((away_rating - (home_rating + home_adv)) / 400.0)))
        if row.home_score > row.away_score:
            actual_home = 1.0
        elif row.home_score < row.away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5
        delta = k_factor * (actual_home - expected_home)
        ratings[home] = home_rating + delta
        ratings[away] = away_rating - delta
        rows.append(
            {
                "match_id": row.match_id,
                "home_elo_before": home_rating,
                "away_elo_before": away_rating,
                "elo_difference": home_rating - away_rating,
                "home_home_advantage": home_adv,
                "elo_match_importance": importance,
            }
        )
    return pd.DataFrame(rows)


def rename_by_aliases(df: pd.DataFrame, alias_map: dict[str, list[str]]) -> pd.DataFrame:
    current = {canonicalize_name(col): col for col in df.columns}
    rename_map = {}
    for target, aliases in alias_map.items():
        for alias in aliases:
            col = current.get(canonicalize_name(alias))
            if col is not None:
                rename_map[col] = target
                break
    return df.rename(columns=rename_map)


def ensure_required_columns(df: pd.DataFrame, required: Iterable[str], kind: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{kind} file is missing required columns: {missing}")


def normalize_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["1", "true", "t", "yes", "y"])


def load_matches(path: str, aliases: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = rename_by_aliases(df, MATCH_COLUMN_ALIASES)
    ensure_required_columns(df, ["date", "home_team", "away_team", "home_score", "away_score"], "Matches")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["home_team"] = df["home_team"].map(lambda x: normalize_team_name(x, aliases))
    df["away_team"] = df["away_team"].map(lambda x: normalize_team_name(x, aliases))
    if "competition" not in df.columns:
        df["competition"] = ""
    else:
        df["competition"] = df["competition"].fillna("")
    if "stage" not in df.columns:
        df["stage"] = df["competition"]
    else:
        df["stage"] = df["stage"].fillna(df["competition"])
    if "city" not in df.columns:
        df["city"] = ""
    else:
        df["city"] = df["city"].fillna("")
    if "country" not in df.columns:
        df["country"] = ""
    else:
        df["country"] = df["country"].fillna("")
    if "neutral" in df.columns:
        df["neutral"] = normalize_bool_series(df["neutral"])
    else:
        df["neutral"] = False
    if "shootout" in df.columns:
        df["shootout"] = normalize_bool_series(df["shootout"])
    else:
        df["shootout"] = False
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    cutoff = pd.Timestamp.today().normalize()
    df = df.loc[df["date"] <= cutoff].copy()
    df = df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    df["match_id"] = np.arange(1, len(df) + 1)
    df["neutral"] = df["neutral"].astype(bool)
    return df


def load_fifa_rankings(path: str, aliases: dict[str, str], ranking_date: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = rename_by_aliases(df, FIFA_COLUMN_ALIASES)
    if "team_name" not in df.columns and "team" in df.columns:
        df = df.rename(columns={"team": "team_name"})
    if "ranking_points" not in df.columns and "points" in df.columns:
        df = df.rename(columns={"points": "ranking_points"})
    if "ranking_date" not in df.columns:
        if not ranking_date:
            raise ValueError(
                "FIFA rankings file has no ranking_date column. Provide --ranking-date YYYY-MM-DD."
            )
        df["ranking_date"] = ranking_date
    ensure_required_columns(df, ["ranking_date", "team_name", "rank", "ranking_points"], "FIFA rankings")
    df["ranking_date"] = pd.to_datetime(df["ranking_date"], errors="coerce")
    df = df.dropna(subset=["ranking_date"]).copy()
    df["team_name"] = df["team_name"].map(lambda x: normalize_team_name(x, aliases))
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["ranking_points"] = pd.to_numeric(df["ranking_points"], errors="coerce")
    df = df.dropna(subset=["rank", "ranking_points"]).copy()
    df["rank"] = df["rank"].astype(int)
    return df.sort_values(["team_name", "ranking_date"]).reset_index(drop=True)


def load_fbref_team_match(path: Optional[str], aliases: dict[str, str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = pd.read_csv(path)
    df = rename_by_aliases(df, TEAM_FBREB_COLUMN_ALIASES)
    required = ["date", "team"]
    ensure_required_columns(df, required, "FBref team-match")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["team"] = df["team"].map(lambda x: normalize_team_name(x, aliases))
    for col in [
        "minutes",
        "goals",
        "assists",
        "shots",
        "shots_on_target",
        "passes_completed",
        "passes_attempted",
        "key_passes",
        "tackles",
        "interceptions",
        "blocks",
        "saves",
        "clean_sheet",
    ]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "is_home" in df.columns:
        df["is_home"] = normalize_bool_series(df["is_home"])
    else:
        df["is_home"] = np.nan
    return df.sort_values(["date", "team"]).reset_index(drop=True)


def load_fbref_player_match(path: Optional[str], aliases: dict[str, str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = pd.read_csv(path)
    df = rename_by_aliases(df, PLAYER_FBREB_COLUMN_ALIASES)
    required = ["date", "team", "player"]
    ensure_required_columns(df, required, "FBref player-match")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["team"] = df["team"].map(lambda x: normalize_team_name(x, aliases))
    for col in [
        "minutes",
        "started",
        "goals",
        "assists",
        "shots",
        "shots_on_target",
        "passes_completed",
        "passes_attempted",
        "key_passes",
        "tackles",
        "interceptions",
        "blocks",
        "saves",
    ]:
        if col not in df.columns:
            df[col] = np.nan
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")
    df["started"] = normalize_bool_series(df["started"].fillna(False)) if "started" in df.columns else False
    numeric_cols = [
        "goals",
        "assists",
        "shots",
        "shots_on_target",
        "passes_completed",
        "passes_attempted",
        "key_passes",
        "tackles",
        "interceptions",
        "blocks",
        "saves",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["date", "team", "player"]).reset_index(drop=True)


def result_from_scores(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "home_win"
    if home_score < away_score:
        return "away_win"
    return "draw"


def build_long_match_history(matches: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "date": matches["date"],
            "team": matches["home_team"],
            "opponent": matches["away_team"],
            "is_home": True,
            "neutral_site": matches["neutral"],
            "competition": matches["competition"],
            "stage": matches["stage"],
            "goals_for": matches["home_score"],
            "goals_against": matches["away_score"],
            "result": np.where(matches["home_score"] > matches["away_score"], "win", np.where(matches["home_score"] < matches["away_score"], "loss", "draw")),
            "points": np.where(matches["home_score"] > matches["away_score"], 3, np.where(matches["home_score"] < matches["away_score"], 0, 1)),
        }
    )
    away = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "date": matches["date"],
            "team": matches["away_team"],
            "opponent": matches["home_team"],
            "is_home": False,
            "neutral_site": matches["neutral"],
            "competition": matches["competition"],
            "stage": matches["stage"],
            "goals_for": matches["away_score"],
            "goals_against": matches["home_score"],
            "result": np.where(matches["away_score"] > matches["home_score"], "win", np.where(matches["away_score"] < matches["home_score"], "loss", "draw")),
            "points": np.where(matches["away_score"] > matches["home_score"], 3, np.where(matches["away_score"] < matches["home_score"], 0, 1)),
        }
    )
    long_df = pd.concat([home, away], ignore_index=True)
    return long_df.sort_values(["team", "date", "match_id"]).reset_index(drop=True)


def add_rolling_team_features(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()
    df["win"] = (df["result"] == "win").astype(int)
    df["draw"] = (df["result"] == "draw").astype(int)
    df["loss"] = (df["result"] == "loss").astype(int)
    df["goal_diff"] = df["goals_for"] - df["goals_against"]
    group = df.groupby("team", sort=False)
    df["days_since_last_match"] = group["date"].transform(lambda s: s.diff().dt.days)
    for window in [5, 10, 20]:
        df[f"matches_played_last_{window}"] = (
            group["match_id"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).count())
            .fillna(0)
            .astype(int)
        )
        df[f"win_rate_last_{window}"] = (
            group["win"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"avg_goals_for_last_{window}"] = (
            group["goals_for"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"avg_goals_against_last_{window}"] = (
            group["goals_against"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"avg_goal_diff_last_{window}"] = (
            group["goal_diff"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"avg_points_last_{window}"] = (
            group["points"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
    df["matches_played_before"] = group.cumcount()
    df["win_rate_before"] = (
        group["win"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    df["avg_goals_for_before"] = (
        group["goals_for"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    df["avg_goals_against_before"] = (
        group["goals_against"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    df["avg_goal_diff_before"] = (
        group["goal_diff"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    df["avg_points_before"] = (
        group["points"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    return df


def add_world_cup_features(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()
    is_world_cup = df["competition"].astype(str).str.contains("world cup", case=False, na=False)
    wc = df.loc[is_world_cup].copy()
    if wc.empty:
        for col in [
            "wc_matches_played_before",
            "wc_win_rate_before",
            "wc_avg_goals_for_before",
            "wc_avg_goals_against_before",
            "wc_avg_goal_diff_before",
        ]:
            df[col] = np.nan
        return df
    wc["win"] = (wc["result"] == "win").astype(int)
    wc["goal_diff"] = wc["goals_for"] - wc["goals_against"]
    group = wc.groupby("team", sort=False)
    wc["wc_matches_played_before"] = group.cumcount()
    wc["wc_win_rate_before"] = group["win"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    wc["wc_avg_goals_for_before"] = group["goals_for"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    wc["wc_avg_goals_against_before"] = group["goals_against"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    wc["wc_avg_goal_diff_before"] = group["goal_diff"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    df = df.merge(
        wc[
            [
                "match_id",
                "team",
                "wc_matches_played_before",
                "wc_win_rate_before",
                "wc_avg_goals_for_before",
                "wc_avg_goals_against_before",
                "wc_avg_goal_diff_before",
            ]
        ],
        on=["match_id", "team"],
        how="left",
    )
    return df


def compute_h2h_features(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    history = {}
    for row in matches.sort_values(["date", "match_id"]).itertuples(index=False):
        team_a = row.home_team
        team_b = row.away_team
        key = tuple(sorted([team_a, team_b]))
        past = history.get(key, [])
        a_wins = sum(1 for item in past if item["winner"] == team_a)
        b_wins = sum(1 for item in past if item["winner"] == team_b)
        draws = sum(1 for item in past if item["winner"] == "draw")
        last_meeting_days = np.nan
        if past:
            last_meeting_days = (row.date - past[-1]["date"]).days
        rows.append(
            {
                "match_id": row.match_id,
                "h2h_matches_played": len(past),
                "h2h_home_wins": a_wins,
                "h2h_away_wins": b_wins,
                "h2h_draws": draws,
                "h2h_home_win_rate": a_wins / len(past) if past else np.nan,
                "h2h_away_win_rate": b_wins / len(past) if past else np.nan,
                "h2h_draw_rate": draws / len(past) if past else np.nan,
                "h2h_last_meeting_days": last_meeting_days,
            }
        )
        winner = result_from_scores(row.home_score, row.away_score)
        winner_team = row.home_team if winner == "home_win" else row.away_team if winner == "away_win" else "draw"
        history.setdefault(key, []).append({"date": row.date, "winner": winner_team})
    return pd.DataFrame(rows)


def add_fifa_features(matches: pd.DataFrame, rankings: pd.DataFrame, team_col: str) -> pd.DataFrame:
    merged_rows = []
    ranking_groups = {team: grp.sort_values("ranking_date") for team, grp in rankings.groupby("team_name")}
    for row in matches.sort_values(["date", "match_id"]).itertuples(index=False):
        team = getattr(row, team_col)
        grp = ranking_groups.get(team)
        if grp is None or grp.empty:
            merged_rows.append({"match_id": row.match_id, team_col: team})
            continue
        eligible = grp[grp["ranking_date"] <= row.date]
        if eligible.empty:
            merged_rows.append({"match_id": row.match_id, team_col: team})
            continue
        latest = eligible.iloc[-1]
        merged_rows.append(
            {
                "match_id": row.match_id,
                team_col: team,
                "fifa_rank": int(latest["rank"]),
                "fifa_ranking_points": float(latest["ranking_points"]),
                "fifa_ranking_date": latest["ranking_date"],
                "fifa_rank_days_old": int((row.date - latest["ranking_date"]).days),
            }
        )
    return pd.DataFrame(merged_rows)


def aggregate_fbref_team_match(team_match: pd.DataFrame) -> pd.DataFrame:
    if team_match is None or team_match.empty:
        return pd.DataFrame()
    numeric_cols = [
        "minutes",
        "goals",
        "assists",
        "shots",
        "shots_on_target",
        "passes_completed",
        "passes_attempted",
        "key_passes",
        "tackles",
        "interceptions",
        "blocks",
        "saves",
        "clean_sheet",
    ]
    agg = team_match.copy()
    for col in numeric_cols:
        if col not in agg.columns:
            agg[col] = np.nan
    group_cols = ["date", "team"]
    summary = agg.groupby(group_cols, as_index=False).agg(
        fbref_minutes=("minutes", "sum"),
        fbref_goals=("goals", "sum"),
        fbref_assists=("assists", "sum"),
        fbref_shots=("shots", "sum"),
        fbref_shots_on_target=("shots_on_target", "sum"),
        fbref_passes_completed=("passes_completed", "sum"),
        fbref_passes_attempted=("passes_attempted", "sum"),
        fbref_key_passes=("key_passes", "sum"),
        fbref_tackles=("tackles", "sum"),
        fbref_interceptions=("interceptions", "sum"),
        fbref_blocks=("blocks", "sum"),
        fbref_saves=("saves", "sum"),
        fbref_clean_sheets=("clean_sheet", "sum"),
        fbref_team_match_rows=("team", "size"),
    )
    summary["date"] = pd.to_datetime(summary["date"], errors="coerce")
    return summary.sort_values(["team", "date"]).reset_index(drop=True)


def aggregate_fbref_player_match(player_match: pd.DataFrame) -> pd.DataFrame:
    if player_match is None or player_match.empty:
        return pd.DataFrame()
    agg = player_match.copy()
    agg["started"] = normalize_bool_series(agg["started"].fillna(False)) if "started" in agg.columns else False
    if "minutes" not in agg.columns:
        agg["minutes"] = np.nan
    group_cols = ["date", "team"]
    summary = agg.groupby(group_cols, as_index=False).agg(
        fbref_player_minutes=("minutes", "sum"),
        fbref_players_used=("player", "nunique"),
        fbref_starters=("started", "sum"),
        fbref_player_goals=("goals", "sum"),
        fbref_player_assists=("assists", "sum"),
        fbref_player_shots=("shots", "sum"),
        fbref_player_shots_on_target=("shots_on_target", "sum"),
        fbref_player_passes_completed=("passes_completed", "sum"),
        fbref_player_passes_attempted=("passes_attempted", "sum"),
        fbref_player_key_passes=("key_passes", "sum"),
        fbref_player_tackles=("tackles", "sum"),
        fbref_player_interceptions=("interceptions", "sum"),
        fbref_player_blocks=("blocks", "sum"),
        fbref_player_saves=("saves", "sum"),
    )
    summary["date"] = pd.to_datetime(summary["date"], errors="coerce")
    return summary.sort_values(["team", "date"]).reset_index(drop=True)


def rolling_team_form(df: pd.DataFrame, team_col: str, value_cols: list[str], windows: list[int]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.sort_values([team_col, "date"]).copy()
    group = out.groupby(team_col, sort=False)
    for col in value_cols:
        if col not in out.columns:
            continue
        for window in windows:
            out[f"{col}_last_{window}"] = group[col].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )
    return out


def build_team_match_dataset(
    matches: pd.DataFrame,
    rankings: pd.DataFrame,
    fbref_team_match: Optional[pd.DataFrame],
    fbref_player_match: Optional[pd.DataFrame],
) -> pd.DataFrame:
    long_df = build_long_match_history(matches)
    long_df = add_rolling_team_features(long_df)
    long_df = add_world_cup_features(long_df)

    fifa_team = add_fifa_features(long_df.rename(columns={"team": "team_name"}), rankings, "team_name")
    fifa_team = fifa_team.rename(columns={"team_name": "team"})
    long_df = long_df.merge(fifa_team, on=["match_id", "team"], how="left")

    fbref_sources = []
    if fbref_team_match is not None and not fbref_team_match.empty:
        fbref_sources.append(aggregate_fbref_team_match(fbref_team_match))
    if fbref_player_match is not None and not fbref_player_match.empty:
        fbref_sources.append(aggregate_fbref_player_match(fbref_player_match))
    if fbref_sources:
        fbref = fbref_sources[0]
        for extra in fbref_sources[1:]:
            fbref = fbref.merge(extra, on=["date", "team"], how="outer")
        long_df = long_df.merge(fbref, on=["date", "team"], how="left")
        fbref_numeric_cols = [
            "fbref_minutes",
            "fbref_goals",
            "fbref_assists",
            "fbref_shots",
            "fbref_shots_on_target",
            "fbref_passes_completed",
            "fbref_passes_attempted",
            "fbref_key_passes",
            "fbref_tackles",
            "fbref_interceptions",
            "fbref_blocks",
            "fbref_saves",
            "fbref_clean_sheets",
            "fbref_team_match_rows",
            "fbref_player_minutes",
            "fbref_players_used",
            "fbref_starters",
            "fbref_player_goals",
            "fbref_player_assists",
            "fbref_player_shots",
            "fbref_player_shots_on_target",
            "fbref_player_passes_completed",
            "fbref_player_passes_attempted",
            "fbref_player_key_passes",
            "fbref_player_tackles",
            "fbref_player_interceptions",
            "fbref_player_blocks",
            "fbref_player_saves",
        ]
        long_df = rolling_team_form(long_df, "team", fbref_numeric_cols, [5, 10, 20])

    long_df["team_side"] = np.where(long_df["is_home"], "home", "away")
    long_df["goal_diff"] = long_df["goals_for"] - long_df["goals_against"]
    return long_df


MATCH_FEATURE_COLUMNS = [
    "matches_played_before",
    "win_rate_before",
    "avg_goals_for_before",
    "avg_goals_against_before",
    "avg_goal_diff_before",
    "avg_points_before",
    "matches_played_last_5",
    "win_rate_last_5",
    "avg_goals_for_last_5",
    "avg_goals_against_last_5",
    "avg_goal_diff_last_5",
    "avg_points_last_5",
    "matches_played_last_10",
    "win_rate_last_10",
    "avg_goals_for_last_10",
    "avg_goals_against_last_10",
    "avg_goal_diff_last_10",
    "avg_points_last_10",
    "matches_played_last_20",
    "win_rate_last_20",
    "avg_goals_for_last_20",
    "avg_goals_against_last_20",
    "avg_goal_diff_last_20",
    "avg_points_last_20",
    "days_since_last_match",
    "wc_matches_played_before",
    "wc_win_rate_before",
    "wc_avg_goals_for_before",
    "wc_avg_goals_against_before",
    "wc_avg_goal_diff_before",
    "fifa_rank",
    "fifa_ranking_points",
    "fifa_rank_days_old",
    "elo_before",
    "fbref_minutes",
    "fbref_goals",
    "fbref_assists",
    "fbref_shots",
    "fbref_shots_on_target",
    "fbref_passes_completed",
    "fbref_passes_attempted",
    "fbref_key_passes",
    "fbref_tackles",
    "fbref_interceptions",
    "fbref_blocks",
    "fbref_saves",
    "fbref_clean_sheets",
    "fbref_team_match_rows",
    "fbref_player_minutes",
    "fbref_players_used",
    "fbref_starters",
    "fbref_player_goals",
    "fbref_player_assists",
    "fbref_player_shots",
    "fbref_player_shots_on_target",
    "fbref_player_passes_completed",
    "fbref_player_passes_attempted",
    "fbref_player_key_passes",
    "fbref_player_tackles",
    "fbref_player_interceptions",
    "fbref_player_blocks",
    "fbref_player_saves",
]


def build_match_dataset(matches: pd.DataFrame, team_dataset: pd.DataFrame) -> pd.DataFrame:
    base = matches.copy().sort_values(["date", "match_id"]).reset_index(drop=True)
    base["competition"] = base["competition"].fillna("")
    base["stage"] = base["stage"].fillna("")
    base["neutral_site"] = base["neutral"].astype(bool)
    base["stage_encoded"] = [
        stage_to_score(stage, competition)
        for stage, competition in zip(base["stage"], base["competition"], strict=False)
    ]
    base["match_outcome"] = np.select(
        [base["home_score"] > base["away_score"], base["home_score"] < base["away_score"]],
        ["home_win", "away_win"],
        default="draw",
    )
    base["target_home_win"] = (base["match_outcome"] == "home_win").astype(int)
    base["target_draw"] = (base["match_outcome"] == "draw").astype(int)
    base["target_away_win"] = (base["match_outcome"] == "away_win").astype(int)
    base["match_total_goals"] = base["home_score"] + base["away_score"]
    base["home_goal_diff"] = base["home_score"] - base["away_score"]

    elo = compute_elo_features(base)
    h2h = compute_h2h_features(base)

    home = team_dataset[team_dataset["team_side"] == "home"].copy()
    away = team_dataset[team_dataset["team_side"] == "away"].copy()
    selected = [col for col in MATCH_FEATURE_COLUMNS if col in team_dataset.columns]
    home = home[["match_id"] + selected].rename(columns={col: f"home_{col}" for col in selected})
    away = away[["match_id"] + selected].rename(columns={col: f"away_{col}" for col in selected})

    merged = base.merge(elo, on="match_id", how="left").merge(h2h, on="match_id", how="left")
    merged = merged.merge(home, on="match_id", how="left").merge(away, on="match_id", how="left")

    for feature in MATCH_FEATURE_COLUMNS:
        diff_col = f"{feature}_difference"
        if diff_col not in merged.columns:
            merged[diff_col] = np.nan

    diff_rules = {
        "matches_played_before": "home_minus_away",
        "win_rate_before": "home_minus_away",
        "avg_goals_for_before": "home_minus_away",
        "avg_goals_against_before": "away_minus_home",
        "avg_goal_diff_before": "home_minus_away",
        "avg_points_before": "home_minus_away",
        "matches_played_last_5": "home_minus_away",
        "win_rate_last_5": "home_minus_away",
        "avg_goals_for_last_5": "home_minus_away",
        "avg_goals_against_last_5": "away_minus_home",
        "avg_goal_diff_last_5": "home_minus_away",
        "avg_points_last_5": "home_minus_away",
        "matches_played_last_10": "home_minus_away",
        "win_rate_last_10": "home_minus_away",
        "avg_goals_for_last_10": "home_minus_away",
        "avg_goals_against_last_10": "away_minus_home",
        "avg_goal_diff_last_10": "home_minus_away",
        "avg_points_last_10": "home_minus_away",
        "matches_played_last_20": "home_minus_away",
        "win_rate_last_20": "home_minus_away",
        "avg_goals_for_last_20": "home_minus_away",
        "avg_goals_against_last_20": "away_minus_home",
        "avg_goal_diff_last_20": "home_minus_away",
        "avg_points_last_20": "home_minus_away",
        "days_since_last_match": "home_minus_away",
        "wc_matches_played_before": "home_minus_away",
        "wc_win_rate_before": "home_minus_away",
        "wc_avg_goals_for_before": "home_minus_away",
        "wc_avg_goals_against_before": "away_minus_home",
        "wc_avg_goal_diff_before": "home_minus_away",
        "fifa_rank": "away_minus_home",
        "fifa_ranking_points": "home_minus_away",
        "fifa_rank_days_old": "away_minus_home",
        "fbref_minutes": "home_minus_away",
        "fbref_goals": "home_minus_away",
        "fbref_assists": "home_minus_away",
        "fbref_shots": "home_minus_away",
        "fbref_shots_on_target": "home_minus_away",
        "fbref_passes_completed": "home_minus_away",
        "fbref_passes_attempted": "home_minus_away",
        "fbref_key_passes": "home_minus_away",
        "fbref_tackles": "home_minus_away",
        "fbref_interceptions": "home_minus_away",
        "fbref_blocks": "home_minus_away",
        "fbref_saves": "home_minus_away",
        "fbref_clean_sheets": "home_minus_away",
        "fbref_team_match_rows": "home_minus_away",
        "fbref_player_minutes": "home_minus_away",
        "fbref_players_used": "home_minus_away",
        "fbref_starters": "home_minus_away",
        "fbref_player_goals": "home_minus_away",
        "fbref_player_assists": "home_minus_away",
        "fbref_player_shots": "home_minus_away",
        "fbref_player_shots_on_target": "home_minus_away",
        "fbref_player_passes_completed": "home_minus_away",
        "fbref_player_passes_attempted": "home_minus_away",
        "fbref_player_key_passes": "home_minus_away",
        "fbref_player_tackles": "home_minus_away",
        "fbref_player_interceptions": "home_minus_away",
        "fbref_player_blocks": "home_minus_away",
        "fbref_player_saves": "home_minus_away",
    }

    for feature, rule in diff_rules.items():
        home_col = f"home_{feature}"
        away_col = f"away_{feature}"
        if home_col not in merged.columns or away_col not in merged.columns:
            continue
        if rule == "away_minus_home":
            merged[f"{feature}_difference"] = merged[away_col] - merged[home_col]
        else:
            merged[f"{feature}_difference"] = merged[home_col] - merged[away_col]

    keep_columns = [
        "match_id",
        "date",
        "home_team",
        "away_team",
        "competition",
        "stage",
        "stage_encoded",
        "neutral_site",
        "home_score",
        "away_score",
        "match_outcome",
        "target_home_win",
        "target_draw",
        "target_away_win",
        "match_total_goals",
        "home_goal_diff",
        "home_elo_before",
        "away_elo_before",
        "elo_difference",
        "home_home_advantage",
        "elo_match_importance",
        "h2h_matches_played",
        "h2h_home_wins",
        "h2h_away_wins",
        "h2h_draws",
        "h2h_home_win_rate",
        "h2h_away_win_rate",
        "h2h_draw_rate",
        "h2h_last_meeting_days",
    ]
    for feature in MATCH_FEATURE_COLUMNS:
        diff_col = f"{feature}_difference"
        if diff_col in merged.columns:
            keep_columns.append(diff_col)

    merged = merged[keep_columns].sort_values(["date", "match_id"]).reset_index(drop=True)
    return merged


def write_outputs(match_df: pd.DataFrame, team_df: pd.DataFrame, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    feature_path = out / "match_feature_set.csv"
    match_path = out / "match_training_set.csv"
    team_path = out / "team_match_training_set.csv"
    match_df.to_csv(feature_path, index=False)
    match_df.to_csv(match_path, index=False)
    team_df.to_csv(team_path, index=False)
    print(f"Wrote {feature_path}")
    print(f"Wrote {match_path}")
    print(f"Wrote {team_path}")


def main() -> None:
    args = parse_args()
    aliases = load_aliases(args.team_aliases)
    matches = load_matches(args.matches, aliases)
    rankings = load_fifa_rankings(args.fifa_rankings, aliases, args.ranking_date)
    fbref_team = load_fbref_team_match(args.fbref_team_match, aliases)
    fbref_player = load_fbref_player_match(args.fbref_player_match, aliases)
    team_df = build_team_match_dataset(matches, rankings, fbref_team, fbref_player)
    match_df = build_match_dataset(matches, team_df)
    write_outputs(match_df, team_df, args.output_dir)


if __name__ == "__main__":
    main()
