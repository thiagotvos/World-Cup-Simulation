"""Command-line interface and helpers for tournament simulation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

from .data import TeamProfile, TournamentConfig, build_demo_tournament, load_team_profiles, load_tournament_config
from .model import ModelBundle
from .simulation import estimate_tournament_outcomes, simulate_single_tournament


def simulate_world_cup(
    model_dir: str,
    tournament: TournamentConfig | None = None,
    tournament_json: str | None = None,
    team_profiles: dict[str, TeamProfile] | None = None,
    profiles_csv: str | None = None,
    runs: int = 1,
    seed: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
):
    bundle = ModelBundle.load(model_dir)
    tournament = tournament or load_tournament_config(tournament_json) or build_demo_tournament()
    profiles = team_profiles or load_team_profiles(profiles_csv)
    if runs <= 1:
        result = simulate_single_tournament(bundle, tournament, profiles, seed=seed)
        return {
            "mode": "single_run",
            "result": result,
        }
    summary = estimate_tournament_outcomes(
        bundle, tournament, profiles, seed=seed, runs=runs, progress_callback=progress_callback
    )
    return {
        "mode": "monte_carlo",
        "summary": summary,
    }


def _print_progress(completed: int, total: int, start_time: float) -> None:
    elapsed = time.monotonic() - start_time
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = (total - completed) / rate if rate > 0 else 0.0
    print(
        f"\rSimulating... {completed}/{total} runs "
        f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)",
        end="",
        file=sys.stderr,
        flush=True,
    )
    if completed == total:
        print(file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a World Cup tournament")
    parser.add_argument("--model-dir", default="model/saved_model")
    parser.add_argument("--tournament", default=None, help="Optional path to a tournament JSON file.")
    parser.add_argument("--profiles", default=None, help="Optional path to team profiles CSV file.")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start_time = time.monotonic()
    progress_callback = (
        (lambda completed, total: _print_progress(completed, total, start_time)) if args.runs > 1 else None
    )
    payload = simulate_world_cup(
        model_dir=args.model_dir,
        tournament_json=args.tournament,
        profiles_csv=args.profiles,
        runs=args.runs,
        seed=args.seed,
        progress_callback=progress_callback,
    )
    print(json.dumps(payload, default=lambda obj: getattr(obj, "__dict__", str(obj)), indent=2))


if __name__ == "__main__":
    main()
