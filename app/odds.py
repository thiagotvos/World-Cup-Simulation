"""Helpers for sportsbook odds baselines."""

from __future__ import annotations

from dataclasses import dataclass


def decimal_odds_to_implied_probability(odds: float) -> float:
    if odds <= 0:
        return 0.0
    return 1.0 / odds


def normalize_probabilities(*probabilities: float) -> list[float]:
    total = sum(max(value, 0.0) for value in probabilities)
    if total <= 0:
        return [0.0 for _ in probabilities]
    return [max(value, 0.0) / total for value in probabilities]


@dataclass
class OddsBaseline:
    home_win: float
    draw: float
    away_win: float

    @classmethod
    def from_decimal_odds(cls, home_odds: float, draw_odds: float, away_odds: float) -> "OddsBaseline":
        implied = normalize_probabilities(
            decimal_odds_to_implied_probability(home_odds),
            decimal_odds_to_implied_probability(draw_odds),
            decimal_odds_to_implied_probability(away_odds),
        )
        return cls(home_win=implied[0], draw=implied[1], away_win=implied[2])

    def as_dict(self) -> dict[str, float]:
        return {
            "home_win": self.home_win,
            "draw": self.draw,
            "away_win": self.away_win,
        }

