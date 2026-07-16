"""Tournament simulation logic for the World Cup project."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from itertools import combinations
from typing import Any, Callable, Optional

import numpy as np

from .data import TeamProfile, TournamentConfig, normalize_team_name
from .features import FeatureEncoder, TeamState
from .model import ModelBundle, sample_scoreline


@dataclass
class MatchTimelineEntry:
    stage: str
    group: str | None
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    winner: str
    expected_home_goals: float
    expected_away_goals: float
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    decided_by: str = "regular_time"
    regulation_home_goals: int | None = None
    regulation_away_goals: int | None = None
    penalty_home_goals: int | None = None
    penalty_away_goals: int | None = None


@dataclass
class StandingRow:
    team: str
    points: int = 0
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    goal_difference: int = 0

    def register(self, goals_for: int, goals_against: int) -> None:
        self.played += 1
        self.goals_for += goals_for
        self.goals_against += goals_against
        self.goal_difference = self.goals_for - self.goals_against
        if goals_for > goals_against:
            self.wins += 1
            self.points += 3
        elif goals_for == goals_against:
            self.draws += 1
            self.points += 1
        else:
            self.losses += 1


@dataclass
class GroupResult:
    group_name: str
    standings: list[StandingRow]
    matches: list[MatchTimelineEntry]


@dataclass
class TournamentSimulationResult:
    champion: str
    runner_up: str
    third_place: str | None
    group_results: list[GroupResult]
    knockout_results: list[MatchTimelineEntry]
    best_third_placed: list[str]
    timeline: list[MatchTimelineEntry]


def _sort_standings(rows: list[StandingRow]) -> list[StandingRow]:
    return sorted(
        rows,
        key=lambda row: (
            row.points,
            row.goal_difference,
            row.goals_for,
            row.team.lower(),
        ),
        reverse=True,
    )


def _build_team_states(
    profiles: dict[str, TeamProfile] | None,
    encoder: FeatureEncoder,
) -> dict[str, TeamState]:
    states: dict[str, TeamState] = {}
    for name, profile in (profiles or {}).items():
        states[normalize_team_name(name)] = TeamState.from_profile(profile, history_window=encoder.history_window)
    return states


def _get_state(
    states: dict[str, TeamState],
    team_name: str,
    profiles: dict[str, TeamProfile] | None,
    encoder: FeatureEncoder,
) -> TeamState:
    key = normalize_team_name(team_name)
    if key not in states:
        profile = (profiles or {}).get(key)
        state = TeamState.from_profile(profile, history_window=encoder.history_window)
        state.team = team_name
        states[key] = state
    if not states[key].team:
        states[key].team = team_name
    return states[key]


def simulate_match(
    bundle: ModelBundle,
    home_state: TeamState,
    away_state: TeamState,
    stage: str,
    group: str | None,
    rng: np.random.Generator,
    knockout: bool = False,
    home_team: str | None = None,
    away_team: str | None = None,
) -> tuple[MatchTimelineEntry, int, int]:
    context = {
        "competition": "world cup",
        "stage": stage,
        "attrs": {},
    }
    from .data import MatchRecord

    match = MatchRecord(
        team_home=home_team or home_state.team,
        team_away=away_team or away_state.team,
        score_home=0,
        score_away=0,
        competition="World Cup",
        stage=stage,
        attrs={},
    )
    features = bundle.encoder.transform(match, home_state, away_state)
    prediction = bundle.predict_match(features)
    home_goals, away_goals = sample_scoreline(
        prediction["expected_home_goals"],
        prediction["expected_away_goals"],
        rng,
    )

    decided_by = "regular_time"
    winner = match.team_home if home_goals > away_goals else match.team_away if away_goals > home_goals else "draw"
    regulation_home_goals: int | None = None
    regulation_away_goals: int | None = None
    penalty_home_goals: int | None = None
    penalty_away_goals: int | None = None

    if knockout and home_goals == away_goals:
        regulation_home_goals = home_goals
        regulation_away_goals = away_goals
        # Extra time: two 15-minute periods, so scale the 90-minute expected
        # goals down to a 30-minute rate.
        extra_home_goals, extra_away_goals = sample_scoreline(
            prediction["expected_home_goals"] * (30.0 / 90.0),
            prediction["expected_away_goals"] * (30.0 / 90.0),
            rng,
        )
        home_goals += extra_home_goals
        away_goals += extra_away_goals
        if home_goals != away_goals:
            decided_by = "extra_time"
            winner = match.team_home if home_goals > away_goals else match.team_away

    if knockout and home_goals == away_goals:
        decided_by = "penalties"
        home_penalty_prob = float(np.clip(0.5 + (prediction["home_win_prob"] - prediction["away_win_prob"]) * 0.25, 0.35, 0.65))
        away_penalty_prob = 1.0 - home_penalty_prob
        penalty_home_goals = 0
        penalty_away_goals = 0
        for shot_index in range(5):
            if rng.random() < home_penalty_prob:
                penalty_home_goals += 1
            if rng.random() < away_penalty_prob:
                penalty_away_goals += 1

            remaining_shots = 4 - shot_index
            if penalty_home_goals > penalty_away_goals + remaining_shots:
                break
            if penalty_away_goals > penalty_home_goals + remaining_shots:
                break

        while penalty_home_goals == penalty_away_goals:
            if rng.random() < home_penalty_prob:
                penalty_home_goals += 1
            if rng.random() < away_penalty_prob:
                penalty_away_goals += 1
        winner = match.team_home if penalty_home_goals > penalty_away_goals else match.team_away

    entry = MatchTimelineEntry(
        stage=stage,
        group=group,
        home_team=match.team_home,
        away_team=match.team_away,
        home_goals=home_goals,
        away_goals=away_goals,
        winner=winner,
        expected_home_goals=prediction["expected_home_goals"],
        expected_away_goals=prediction["expected_away_goals"],
        home_win_prob=prediction["home_win_prob"],
        draw_prob=prediction["draw_prob"],
        away_win_prob=prediction["away_win_prob"],
        decided_by=decided_by,
        regulation_home_goals=regulation_home_goals,
        regulation_away_goals=regulation_away_goals,
        penalty_home_goals=penalty_home_goals,
        penalty_away_goals=penalty_away_goals,
    )
    return entry, home_goals, away_goals


def simulate_group_stage(
    bundle: ModelBundle,
    tournament: TournamentConfig,
    team_profiles: dict[str, TeamProfile] | None,
    rng: np.random.Generator,
) -> tuple[list[GroupResult], dict[str, TeamState], list[MatchTimelineEntry]]:
    states = _build_team_states(team_profiles, bundle.encoder)
    timeline: list[MatchTimelineEntry] = []
    results: list[GroupResult] = []

    for group_name, teams in tournament.groups.items():
        standings = {team: StandingRow(team=team) for team in teams}
        matches: list[MatchTimelineEntry] = []
        for home_index, away_index in combinations(range(len(teams)), 2):
            home_team = teams[home_index]
            away_team = teams[away_index]
            home_state = _get_state(states, home_team, team_profiles, bundle.encoder)
            away_state = _get_state(states, away_team, team_profiles, bundle.encoder)
            entry, home_goals, away_goals = simulate_match(
                bundle=bundle,
                home_state=home_state,
                away_state=away_state,
                stage="group",
                group=group_name,
                rng=rng,
                knockout=False,
                home_team=home_team,
                away_team=away_team,
            )
            standings[home_team].register(home_goals, away_goals)
            standings[away_team].register(away_goals, home_goals)
            home_state.apply_match(home_goals, away_goals)
            away_state.apply_match(away_goals, home_goals)
            matches.append(entry)
            timeline.append(entry)
        ranked = _sort_standings(list(standings.values()))
        results.append(GroupResult(group_name=group_name, standings=ranked, matches=matches))
    return results, states, timeline


def select_best_third_placed(group_results: list[GroupResult], count: int = 8) -> list[StandingRow]:
    third_placed = [group_result.standings[2] for group_result in group_results if len(group_result.standings) >= 3]
    return _sort_standings(third_placed)[:count]


def build_round_of_32_entrants(group_results: list[GroupResult], best_third: list[StandingRow]) -> list[str]:
    winners = [result.standings[0].team for result in group_results]
    runners_up = [result.standings[1].team for result in group_results]
    thirds = [row.team for row in best_third]
    entrants = winners + runners_up + thirds
    return entrants[:32]


def simulate_knockout_round(
    bundle: ModelBundle,
    entrants: list[str],
    states: dict[str, TeamState],
    team_profiles: dict[str, TeamProfile] | None,
    rng: np.random.Generator,
    round_name: str,
) -> tuple[list[MatchTimelineEntry], list[str], list[str]]:
    entries: list[MatchTimelineEntry] = []
    winners: list[str] = []
    losers: list[str] = []
    for index in range(0, len(entrants), 2):
        home_team = entrants[index]
        away_team = entrants[index + 1]
        home_state = _get_state(states, home_team, team_profiles, bundle.encoder)
        away_state = _get_state(states, away_team, team_profiles, bundle.encoder)
        entry, home_goals, away_goals = simulate_match(
            bundle=bundle,
            home_state=home_state,
            away_state=away_state,
            stage=round_name,
            group=None,
            rng=rng,
            knockout=True,
            home_team=home_team,
            away_team=away_team,
        )
        entries.append(entry)
        winners.append(entry.winner)
        losers.append(away_team if entry.winner == home_team else home_team)
        home_state.apply_match(home_goals, away_goals)
        away_state.apply_match(away_goals, home_goals)
    return entries, winners, losers


def simulate_single_tournament(
    bundle: ModelBundle,
    tournament: TournamentConfig,
    team_profiles: dict[str, TeamProfile] | None = None,
    seed: int | None = None,
) -> TournamentSimulationResult:
    rng = np.random.default_rng(seed)
    group_results, states, timeline = simulate_group_stage(bundle, tournament, team_profiles, rng)
    best_third = select_best_third_placed(group_results, count=8)
    best_third_names = [row.team for row in best_third]
    entrants = build_round_of_32_entrants(group_results, best_third)

    knockout_results: list[MatchTimelineEntry] = []
    round_name = "round_of_32"
    current_entrants = entrants
    semifinal_losers: list[str] = []
    runner_up = ""
    while len(current_entrants) > 1:
        round_entries, winners, losers = simulate_knockout_round(
            bundle=bundle,
            entrants=current_entrants,
            states=states,
            team_profiles=team_profiles,
            rng=rng,
            round_name=round_name,
        )
        knockout_results.extend(round_entries)
        timeline.extend(round_entries)
        if len(current_entrants) == 4:
            semifinal_losers = losers
        if len(current_entrants) == 2:
            runner_up = losers[0] if losers else ""
        current_entrants = winners
        if len(current_entrants) == 16:
            round_name = "round_of_16"
        elif len(current_entrants) == 8:
            round_name = "quarterfinal"
        elif len(current_entrants) == 4:
            round_name = "semifinal"
        elif len(current_entrants) == 2:
            round_name = "final"
        else:
            break

    champion = current_entrants[0] if current_entrants else ""

    third_place = None
    if len(semifinal_losers) == 2:
        third_place_entry, _, _ = simulate_knockout_round(
            bundle=bundle,
            entrants=semifinal_losers,
            states=states,
            team_profiles=team_profiles,
            rng=rng,
            round_name="third_place",
        )
        knockout_results.extend(third_place_entry)
        timeline.extend(third_place_entry)
        third_place = third_place_entry[0].winner

    return TournamentSimulationResult(
        champion=champion,
        runner_up=runner_up,
        third_place=third_place,
        group_results=group_results,
        knockout_results=knockout_results,
        best_third_placed=best_third_names,
        timeline=timeline,
    )


def estimate_tournament_outcomes(
    bundle: ModelBundle,
    tournament: TournamentConfig,
    team_profiles: dict[str, TeamProfile] | None = None,
    seed: int | None = None,
    runs: int = 100,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    champion_counts: dict[str, int] = {}
    runner_up_counts: dict[str, int] = {}
    top8_counts: dict[str, int] = {}
    last_result: TournamentSimulationResult | None = None

    total_runs = max(runs, 1)
    for run_index in range(total_runs):
        result_seed = int(rng.integers(0, 1_000_000_000))
        result = simulate_single_tournament(bundle, tournament, team_profiles, seed=result_seed)
        last_result = result
        champion_counts[result.champion] = champion_counts.get(result.champion, 0) + 1
        if result.runner_up:
            runner_up_counts[result.runner_up] = runner_up_counts.get(result.runner_up, 0) + 1
        for team in result.best_third_placed:
            top8_counts[team] = top8_counts.get(team, 0) + 1
        if progress_callback is not None:
            progress_callback(run_index + 1, total_runs)

    runs = max(runs, 1)
    champion_probabilities = {
        team: count / runs for team, count in sorted(champion_counts.items(), key=lambda item: item[1], reverse=True)
    }
    runner_up_probabilities = {
        team: count / runs for team, count in sorted(runner_up_counts.items(), key=lambda item: item[1], reverse=True)
    }
    best_third_probabilities = {
        team: count / runs for team, count in sorted(top8_counts.items(), key=lambda item: item[1], reverse=True)
    }

    return {
        "runs": runs,
        "champion_probabilities": champion_probabilities,
        "runner_up_probabilities": runner_up_probabilities,
        "best_third_probabilities": best_third_probabilities,
        "last_run": last_result,
    }
