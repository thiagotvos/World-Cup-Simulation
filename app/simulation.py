"""Tournament simulation logic for the World Cup project."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable

import numpy as np

from .data import MatchRecord, TeamProfile, TournamentConfig, normalize_team_name
from .features import FeatureEncoder, TeamState
from .model import ModelBundle, match_result_probabilities, sample_scoreline


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


def _seed_recent_form(
    state: TeamState,
    team_key: str,
    recent_form: dict[str, list[tuple[int, int]]] | None,
) -> None:
    if not recent_form:
        return
    for goals_for, goals_against in recent_form.get(team_key, []):
        state.apply_match(goals_for, goals_against)


def _build_team_states(
    profiles: dict[str, TeamProfile] | None,
    encoder: FeatureEncoder,
    recent_form: dict[str, list[tuple[int, int]]] | None = None,
) -> dict[str, TeamState]:
    states: dict[str, TeamState] = {}
    for name, profile in (profiles or {}).items():
        key = normalize_team_name(name)
        state = TeamState.from_profile(profile, history_window=encoder.history_window)
        _seed_recent_form(state, key, recent_form)
        states[key] = state
    return states


def _get_state(
    states: dict[str, TeamState],
    team_name: str,
    profiles: dict[str, TeamProfile] | None,
    encoder: FeatureEncoder,
    recent_form: dict[str, list[tuple[int, int]]] | None = None,
) -> TeamState:
    key = normalize_team_name(team_name)
    if key not in states:
        profile = (profiles or {}).get(key)
        state = TeamState.from_profile(profile, history_window=encoder.history_window)
        state.team = team_name
        _seed_recent_form(state, key, recent_form)
        states[key] = state
    if not states[key].team:
        states[key].team = team_name
    return states[key]


# Knockout matches are historically tighter and lower-scoring than the
# average match: recent World Cups have averaged roughly 2.15 goals per
# knockout match vs. ~2.6 in the group stage (teams that reach the
# knockout stage are more evenly matched on average, and play more
# cautiously with elimination on the line). The training data has no
# reliable per-match round/stage labels to teach the network this
# directly, so it's applied here as an explicit adjustment instead.
KNOCKOUT_GOAL_DAMPENING = 0.82
# Extra time is even more defensive: since 2018, only about 3 of 16
# extra-time periods have produced any goal at all.
EXTRA_TIME_GOAL_DAMPENING = 0.7


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
    expected_home_goals = prediction["expected_home_goals"]
    expected_away_goals = prediction["expected_away_goals"]
    home_win_prob = prediction["home_win_prob"]
    draw_prob = prediction["draw_prob"]
    away_win_prob = prediction["away_win_prob"]
    if knockout:
        expected_home_goals *= KNOCKOUT_GOAL_DAMPENING
        expected_away_goals *= KNOCKOUT_GOAL_DAMPENING
        home_win_prob, draw_prob, away_win_prob = match_result_probabilities(expected_home_goals, expected_away_goals)
    home_goals, away_goals = sample_scoreline(
        expected_home_goals,
        expected_away_goals,
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
        # goals down to a 30-minute rate, then dampen further for fatigue
        # and caution.
        extra_home_goals, extra_away_goals = sample_scoreline(
            expected_home_goals * (30.0 / 90.0) * EXTRA_TIME_GOAL_DAMPENING,
            expected_away_goals * (30.0 / 90.0) * EXTRA_TIME_GOAL_DAMPENING,
            rng,
        )
        home_goals += extra_home_goals
        away_goals += extra_away_goals
        if home_goals != away_goals:
            decided_by = "extra_time"
            winner = match.team_home if home_goals > away_goals else match.team_away

    if knockout and home_goals == away_goals:
        decided_by = "penalties"
        home_penalty_prob = float(np.clip(0.5 + (home_win_prob - away_win_prob) * 0.25, 0.35, 0.65))
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
        expected_home_goals=expected_home_goals,
        expected_away_goals=expected_away_goals,
        home_win_prob=home_win_prob,
        draw_prob=draw_prob,
        away_win_prob=away_win_prob,
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
    recent_form: dict[str, list[tuple[int, int]]] | None = None,
) -> tuple[list[GroupResult], dict[str, TeamState], list[MatchTimelineEntry]]:
    states = _build_team_states(team_profiles, bundle.encoder, recent_form)
    timeline: list[MatchTimelineEntry] = []
    results: list[GroupResult] = []

    for group_name, teams in tournament.groups.items():
        standings = {team: StandingRow(team=team) for team in teams}
        matches: list[MatchTimelineEntry] = []
        for home_index, away_index in combinations(range(len(teams)), 2):
            home_team = teams[home_index]
            away_team = teams[away_index]
            home_state = _get_state(states, home_team, team_profiles, bundle.encoder, recent_form)
            away_state = _get_state(states, away_team, team_profiles, bundle.encoder, recent_form)
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


# Fixed Round of 32 bracket template for the 2026-style 12-group, 48-team
# format: which group position meets which. Group winners never meet
# another winner in this round, third-placed teams always face a group
# winner (never a side from their own group), and no team faces a side
# from its own group. Source: FIFA 2026 World Cup tournament regulations
# (Round of 32 fixture list).
_ROUND_OF_32_TEMPLATE = [
    (("runner_up", "A"), ("runner_up", "B")),
    (("winner", "E"), ("third", None)),
    (("winner", "F"), ("runner_up", "C")),
    (("winner", "C"), ("runner_up", "F")),
    (("winner", "I"), ("third", None)),
    (("runner_up", "E"), ("runner_up", "I")),
    (("winner", "A"), ("third", None)),
    (("winner", "L"), ("third", None)),
    (("winner", "D"), ("third", None)),
    (("winner", "G"), ("third", None)),
    (("runner_up", "K"), ("runner_up", "L")),
    (("winner", "H"), ("runner_up", "J")),
    (("winner", "B"), ("third", None)),
    (("winner", "J"), ("runner_up", "H")),
    (("winner", "K"), ("third", None)),
    (("runner_up", "D"), ("runner_up", "G")),
]


def _assign_thirds_to_slots(third_groups: list[str], winner_slots: list[str]) -> dict[str, str]:
    """Assign each qualifying third-placed team's group to a "winner vs
    third" slot such that no slot gets the third-placed team from its own
    group (they already played in the group stage)."""
    count = len(winner_slots)
    for shift in range(count):
        rotated = third_groups[shift:] + third_groups[:shift]
        if all(rotated[i] != winner_slots[i] for i in range(count)):
            return dict(zip(winner_slots, rotated))
    return dict(zip(winner_slots, third_groups))


def build_round_of_32_entrants(group_results: list[GroupResult], best_third: list[StandingRow]) -> list[str]:
    """Pairs teams using the fixed Round of 32 template (see
    _ROUND_OF_32_TEMPLATE) instead of just lining up all winners against
    each other, so the bracket matches how the real tournament avoids top
    teams meeting too early."""
    winner_of = {result.group_name: result.standings[0].team for result in group_results if result.standings}
    runner_up_of = {
        result.group_name: result.standings[1].team for result in group_results if len(result.standings) > 1
    }
    group_of_third_team = {
        result.standings[2].team: result.group_name for result in group_results if len(result.standings) > 2
    }

    third_groups_in_rank_order = [group_of_third_team[row.team] for row in best_third if row.team in group_of_third_team]
    winner_slots_needing_third = [
        group for side_a, side_b in _ROUND_OF_32_TEMPLATE for kind, group in (side_a, side_b) if kind == "winner"
        and any(other_kind == "third" for other_kind, _ in (side_a, side_b))
    ]
    third_assignment = _assign_thirds_to_slots(third_groups_in_rank_order, winner_slots_needing_third)
    third_team_of_group = {group: team for team, group in group_of_third_team.items()}

    def resolve(kind: str, group: str | None) -> str:
        if kind == "winner":
            return winner_of[group]
        if kind == "runner_up":
            return runner_up_of[group]
        # "third" slots are keyed by the group of the winner they play against,
        # resolved via third_assignment before this function is called.
        raise ValueError(f"Unresolved slot kind: {kind}")

    entrants: list[str] = []
    for side_a, side_b in _ROUND_OF_32_TEMPLATE:
        for kind, group in (side_a, side_b):
            if kind == "third":
                winner_group = side_a[1] if side_a[0] == "winner" else side_b[1]
                third_group = third_assignment[winner_group]
                entrants.append(third_team_of_group[third_group])
            else:
                entrants.append(resolve(kind, group))
    return entrants


def simulate_knockout_round(
    bundle: ModelBundle,
    entrants: list[str],
    states: dict[str, TeamState],
    team_profiles: dict[str, TeamProfile] | None,
    rng: np.random.Generator,
    round_name: str,
    recent_form: dict[str, list[tuple[int, int]]] | None = None,
) -> tuple[list[MatchTimelineEntry], list[str], list[str]]:
    entries: list[MatchTimelineEntry] = []
    winners: list[str] = []
    losers: list[str] = []
    for index in range(0, len(entrants), 2):
        home_team = entrants[index]
        away_team = entrants[index + 1]
        home_state = _get_state(states, home_team, team_profiles, bundle.encoder, recent_form)
        away_state = _get_state(states, away_team, team_profiles, bundle.encoder, recent_form)
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
    recent_form: dict[str, list[tuple[int, int]]] | None = None,
) -> TournamentSimulationResult:
    rng = np.random.default_rng(seed)
    group_results, states, timeline = simulate_group_stage(bundle, tournament, team_profiles, rng, recent_form)
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
            recent_form=recent_form,
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
            recent_form=recent_form,
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
    recent_form: dict[str, list[tuple[int, int]]] | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    champion_counts: dict[str, int] = {}
    runner_up_counts: dict[str, int] = {}
    top8_counts: dict[str, int] = {}
    last_result: TournamentSimulationResult | None = None

    total_runs = max(runs, 1)
    for run_index in range(total_runs):
        result_seed = int(rng.integers(0, 1_000_000_000))
        result = simulate_single_tournament(bundle, tournament, team_profiles, seed=result_seed, recent_form=recent_form)
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
