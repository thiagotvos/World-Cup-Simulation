"""Shared constants used by the backend pipeline."""

from __future__ import annotations

DEFAULT_HISTORY_WINDOW = 5

STAGE_CATEGORIES = [
    "group",
    "round_of_32",
    "round_of_16",
    "quarterfinal",
    "semifinal",
    "final",
    "friendly",
    "qualifier",
    "other",
]

COMPETITION_CATEGORIES = [
    "world_cup",
    "world_cup_qualifier",
    "friendly",
    "tournament",
    "other",
]

GROUP_NAMES = [chr(ord("A") + index) for index in range(12)]

