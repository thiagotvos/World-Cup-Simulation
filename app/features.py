"""Feature engineering and team state management."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .constants import COMPETITION_CATEGORIES, DEFAULT_HISTORY_WINDOW, STAGE_CATEGORIES
from .data import MatchRecord, TeamProfile, normalize_key, normalize_team_name, parse_float


def categorize_stage(value: str) -> str:
    text = (value or "").strip().lower()
    if not text:
        return "other"
    if "group" in text:
        return "group"
    if "round of 32" in text or "round_of_32" in text or "round-of-32" in text:
        return "round_of_32"
    if "round of 16" in text or "round_of_16" in text or "round-of-16" in text:
        return "round_of_16"
    if "quarter" in text:
        return "quarterfinal"
    if "semi" in text:
        return "semifinal"
    if "final" in text:
        return "final"
    if "friendly" in text:
        return "friendly"
    if "qual" in text:
        return "qualifier"
    return "other"


def categorize_competition(value: str) -> str:
    text = (value or "").strip().lower()
    if not text:
        return "other"
    if "world cup" in text or "worldcup" in text:
        return "world_cup"
    if "qual" in text:
        return "world_cup_qualifier"
    if "friendly" in text:
        return "friendly"
    if "cup" in text or "tournament" in text or "championship" in text:
        return "tournament"
    return "other"


def _extract_numeric_source(attrs: dict[str, Any], aliases: list[str], default: float = 0.0) -> float:
    for alias in aliases:
        if alias in attrs:
            return parse_float(attrs[alias], default)
    return default


@dataclass
class TeamState:
    team: str
    elo: float = 0.0
    fifa_rank: float = 0.0
    history_window: int = DEFAULT_HISTORY_WINDOW
    history: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_HISTORY_WINDOW))

    @classmethod
    def from_profile(cls, profile: TeamProfile | None, history_window: int = DEFAULT_HISTORY_WINDOW) -> "TeamState":
        if profile is None:
            return cls(team="", elo=0.0, fifa_rank=0.0, history_window=history_window, history=deque(maxlen=history_window))
        return cls(
            team=profile.team,
            elo=profile.elo,
            fifa_rank=profile.fifa_rank,
            history_window=history_window,
            history=deque(maxlen=history_window),
        )

    def clone(self) -> "TeamState":
        copied = TeamState(
            team=self.team,
            elo=self.elo,
            fifa_rank=self.fifa_rank,
            history_window=self.history_window,
            history=deque(self.history, maxlen=self.history_window),
        )
        return copied

    def apply_match(self, goals_for: int, goals_against: int) -> None:
        self.history.append((int(goals_for), int(goals_against)))

    @property
    def games_played(self) -> int:
        return len(self.history)

    @property
    def recent_points_per_match(self) -> float:
        if not self.history:
            return 0.0
        points = 0.0
        for goals_for, goals_against in self.history:
            if goals_for > goals_against:
                points += 3.0
            elif goals_for == goals_against:
                points += 1.0
        return points / len(self.history)

    @property
    def recent_win_rate(self) -> float:
        if not self.history:
            return 0.0
        wins = sum(1 for goals_for, goals_against in self.history if goals_for > goals_against)
        return wins / len(self.history)

    @property
    def recent_goal_diff(self) -> float:
        if not self.history:
            return 0.0
        total = sum(goals_for - goals_against for goals_for, goals_against in self.history)
        return total / len(self.history)

    @property
    def recent_goals_for(self) -> float:
        if not self.history:
            return 0.0
        total = sum(goals_for for goals_for, _ in self.history)
        return total / len(self.history)

    @property
    def recent_goals_against(self) -> float:
        if not self.history:
            return 0.0
        total = sum(goals_against for _, goals_against in self.history)
        return total / len(self.history)


class FeatureEncoder:
    """Builds a consistent numeric feature vector for each match."""

    def __init__(self, history_window: int = DEFAULT_HISTORY_WINDOW):
        self.history_window = history_window
        self.stage_categories = list(STAGE_CATEGORIES)
        self.competition_categories = list(COMPETITION_CATEGORIES)
        self.feature_names = [
            "elo_diff",
            "fifa_strength_diff",
            "recent_points_per_match_diff",
            "recent_win_rate_diff",
            "recent_goal_diff_diff",
            "recent_goals_for_diff",
            "recent_goals_against_diff",
        ]
        self.feature_names.extend([f"stage_{item}" for item in self.stage_categories])
        self.feature_names.extend([f"competition_{item}" for item in self.competition_categories])

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def transform(self, record: MatchRecord, home_state: TeamState, away_state: TeamState) -> np.ndarray:
        elo_home = _extract_numeric_source(
            record.attrs,
            ["elo_home", "home_elo", "team_home_elo", "rating_home", "home_rating"],
            home_state.elo,
        )
        elo_away = _extract_numeric_source(
            record.attrs,
            ["elo_away", "away_elo", "team_away_elo", "rating_away", "away_rating"],
            away_state.elo,
        )
        home_rank = _extract_numeric_source(
            record.attrs,
            ["fifa_home", "home_fifa_rank", "team_home_fifa_rank", "rank_home"],
            home_state.fifa_rank,
        )
        away_rank = _extract_numeric_source(
            record.attrs,
            ["fifa_away", "away_fifa_rank", "team_away_fifa_rank", "rank_away"],
            away_state.fifa_rank,
        )

        numeric_values = [
            elo_home - elo_away,
            away_rank - home_rank,
            home_state.recent_points_per_match - away_state.recent_points_per_match,
            home_state.recent_win_rate - away_state.recent_win_rate,
            home_state.recent_goal_diff - away_state.recent_goal_diff,
            home_state.recent_goals_for - away_state.recent_goals_for,
            home_state.recent_goals_against - away_state.recent_goals_against,
        ]

        stage_key = categorize_stage(record.stage)
        competition_key = categorize_competition(record.competition)

        stage_one_hot = [1.0 if category == stage_key else 0.0 for category in self.stage_categories]
        competition_one_hot = [1.0 if category == competition_key else 0.0 for category in self.competition_categories]

        vector = np.asarray(numeric_values + stage_one_hot + competition_one_hot, dtype=np.float32)
        return vector

    def extract_initial_states(
        self,
        profiles: dict[str, TeamProfile] | None = None,
    ) -> dict[str, TeamState]:
        states: dict[str, TeamState] = {}
        if not profiles:
            return states
        for team_key, profile in profiles.items():
            states[team_key] = TeamState.from_profile(profile, history_window=self.history_window)
        return states

    def ensure_state(
        self,
        states: dict[str, TeamState],
        team_name: str,
        profiles: dict[str, TeamProfile] | None = None,
    ) -> TeamState:
        team_key = normalize_team_name(team_name)
        if team_key not in states:
            profile = (profiles or {}).get(team_key)
            states[team_key] = TeamState.from_profile(profile, history_window=self.history_window)
            if not states[team_key].team:
                states[team_key].team = team_name
        return states[team_key]

