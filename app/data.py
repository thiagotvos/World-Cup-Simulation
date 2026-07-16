"""Data loading and light preprocessing helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def normalize_key(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def normalize_team_name(value: str) -> str:
    return " ".join(value.strip().lower().split())


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == "":
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == "":
            return default
        return int(float(text))
    except (TypeError, ValueError):
        return default


def parse_date(value: Any) -> datetime:
    text = str(value).strip()
    if not text:
        return datetime.min
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return datetime.min


@dataclass
class MatchRecord:
    team_home: str
    team_away: str
    score_home: int
    score_away: int
    competition: str = ""
    stage: str = ""
    date: datetime = datetime.min
    attrs: Dict[str, Any] = field(default_factory=dict)

    @property
    def result_label(self) -> str:
        if self.score_home > self.score_away:
            return "home"
        if self.score_home < self.score_away:
            return "away"
        return "draw"


@dataclass
class TeamProfile:
    team: str
    elo: float = 0.0
    fifa_rank: float = 0.0


@dataclass
class TournamentConfig:
    groups: Dict[str, list[str]]

    def group_names(self) -> list[str]:
        return list(self.groups.keys())


def load_match_records(csv_path: str | Path) -> list[MatchRecord]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Match file not found: {path}")

    records: list[MatchRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized = {normalize_key(key): value for key, value in row.items() if key}
            team_home = str(
                normalized.get("team_home")
                or normalized.get("home_team")
                or normalized.get("home")
                or ""
            ).strip()
            team_away = str(
                normalized.get("team_away")
                or normalized.get("away_team")
                or normalized.get("away")
                or ""
            ).strip()
            if not team_home or not team_away:
                continue
            score_home = parse_int(
                normalized.get("score_home")
                or normalized.get("home_score")
                or normalized.get("goals_home")
                or normalized.get("home_goals")
            )
            score_away = parse_int(
                normalized.get("score_away")
                or normalized.get("away_score")
                or normalized.get("goals_away")
                or normalized.get("away_goals")
            )
            competition = str(normalized.get("competition") or normalized.get("tournament") or "").strip()
            stage = str(normalized.get("stage") or normalized.get("round") or "").strip()
            date = parse_date(normalized.get("date") or normalized.get("match_date") or "")
            attrs = {key: value for key, value in normalized.items() if key not in {
                "team_home",
                "home_team",
                "home",
                "team_away",
                "away_team",
                "away",
                "score_home",
                "home_score",
                "goals_home",
                "home_goals",
                "score_away",
                "away_score",
                "goals_away",
                "away_goals",
                "competition",
                "tournament",
                "stage",
                "round",
                "date",
                "match_date",
            }}
            records.append(
                MatchRecord(
                    team_home=team_home,
                    team_away=team_away,
                    score_home=score_home,
                    score_away=score_away,
                    competition=competition,
                    stage=stage,
                    date=date,
                    attrs=attrs,
                )
            )
    records.sort(key=lambda item: item.date)
    return records


def load_recent_form(
    csv_path: str | Path,
    history_window: int = 5,
) -> dict[str, list[tuple[int, int]]]:
    """Return each team's real (goals_for, goals_against) for its last
    `history_window` matches, oldest first, so a simulated tournament can
    start from each team's actual current form instead of a blank slate."""
    records = load_match_records(csv_path)
    recent: dict[str, list[tuple[int, int]]] = {}
    for record in records:
        home_key = normalize_team_name(record.team_home)
        away_key = normalize_team_name(record.team_away)
        for key, goals_for, goals_against in (
            (home_key, record.score_home, record.score_away),
            (away_key, record.score_away, record.score_home),
        ):
            bucket = recent.setdefault(key, [])
            bucket.append((goals_for, goals_against))
            if len(bucket) > history_window:
                bucket.pop(0)
    return recent


def load_team_profiles(csv_path: str | Path | None) -> dict[str, TeamProfile]:
    if not csv_path:
        return {}
    path = Path(csv_path)
    if not path.exists():
        return {}

    profiles: dict[str, TeamProfile] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized = {normalize_key(key): value for key, value in row.items() if key}
            team = str(normalized.get("team") or normalized.get("country") or "").strip()
            if not team:
                continue
            profiles[normalize_team_name(team)] = TeamProfile(
                team=team,
                elo=parse_float(normalized.get("elo") or normalized.get("rating")),
                fifa_rank=parse_float(
                    normalized.get("fifa_rank") or normalized.get("ranking") or normalized.get("rank")
                ),
            )
    return profiles


def load_tournament_config(json_path: str | Path | None) -> Optional[TournamentConfig]:
    if not json_path:
        return None
    path = Path(json_path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    groups = data.get("groups", {})
    if not isinstance(groups, dict):
        raise ValueError("Tournament configuration must contain a 'groups' object.")
    normalized: dict[str, list[str]] = {}
    for group_name, teams in groups.items():
        normalized[str(group_name)] = [str(team).strip() for team in teams if str(team).strip()]
    return TournamentConfig(groups=normalized)


def build_demo_tournament() -> TournamentConfig:
    groups: Dict[str, list[str]] = {}
    for index, group_name in enumerate((chr(ord("A") + i) for i in range(12)), start=1):
        base = (index - 1) * 4
        groups[group_name] = [
            f"Team {base + 1}",
            f"Team {base + 2}",
            f"Team {base + 3}",
            f"Team {base + 4}",
        ]
    return TournamentConfig(groups=groups)

