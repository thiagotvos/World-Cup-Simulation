"""Prediction helpers for loading and using a trained model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import MatchRecord, TeamProfile, normalize_team_name
from .features import FeatureEncoder, TeamState
from .model import ModelBundle


@dataclass
class PredictionService:
    bundle: ModelBundle
    team_profiles: dict[str, TeamProfile] | None = None

    @classmethod
    def load(cls, model_dir: str | Path, team_profiles: dict[str, TeamProfile] | None = None) -> "PredictionService":
        bundle = ModelBundle.load(model_dir)
        return cls(bundle=bundle, team_profiles=team_profiles or {})

    def make_states(self, home_team: str, away_team: str) -> tuple[TeamState, TeamState]:
        encoder = self.bundle.encoder
        home_profile = (self.team_profiles or {}).get(normalize_team_name(home_team))
        away_profile = (self.team_profiles or {}).get(normalize_team_name(away_team))
        home_state = TeamState.from_profile(home_profile, history_window=encoder.history_window)
        away_state = TeamState.from_profile(away_profile, history_window=encoder.history_window)
        home_state.team = home_team
        away_state.team = away_team
        return home_state, away_state

    def predict_match(self, home_team: str, away_team: str, competition: str = "World Cup", stage: str = "group") -> dict[str, float]:
        home_state, away_state = self.make_states(home_team, away_team)
        record = MatchRecord(
            team_home=home_team,
            team_away=away_team,
            score_home=0,
            score_away=0,
            competition=competition,
            stage=stage,
            attrs={},
        )
        features = self.bundle.encoder.transform(record, home_state, away_state)
        return self.bundle.predict_match(features)

