"""Compare the trained model against the sportsbook-odds baseline on real
World Cup 2026 results (see docs/evaluation_plan.md)."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from .data import load_team_profiles
from .odds import OddsBaseline
from .predict import PredictionService

RESULT_LABELS = ("home", "draw", "away")

# The FootyStats export spells a few teams differently from data/raw/team_profiles.csv.
TEAM_NAME_ALIASES = {
    "Congo DR": "DR Congo",
    "Cape Verde Islands": "Cape Verde",
    "USMNT": "United States",
}


def _canonical_team(name: str) -> str:
    return TEAM_NAME_ALIASES.get(name, name)


def _parse_optional_float(value: str) -> float | None:
    if value in ("", "N/A", "0.00"):
        return None
    return float(value)


def load_completed_matches(csv_path: str | Path) -> list[dict[str, Any]]:
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    matches = []
    for row in rows:
        if row.get("status") != "complete":
            continue
        home_goals = int(row["home_team_goal_count"])
        away_goals = int(row["away_team_goal_count"])
        if home_goals > away_goals:
            result = "home"
        elif home_goals < away_goals:
            result = "away"
        else:
            result = "draw"
        matches.append(
            {
                "home_team": _canonical_team(row["home_team_name"]),
                "away_team": _canonical_team(row["away_team_name"]),
                # Group-stage rounds are numbered 1-3; knockout rounds are unlabeled ("N/A")
                # in this export, so they collapse into a single "knockout" stage bucket.
                "stage": "group" if row.get("Game Week") in {"1", "2", "3"} else "knockout",
                "result": result,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_xg": _parse_optional_float(row["team_a_xg"]),
                "away_xg": _parse_optional_float(row["team_b_xg"]),
                "odds_home": _parse_optional_float(row["odds_ft_home_team_win"]),
                "odds_draw": _parse_optional_float(row["odds_ft_draw"]),
                "odds_away": _parse_optional_float(row["odds_ft_away_team_win"]),
            }
        )
    return matches


def _log_loss(probs: list[float], label_index: int) -> float:
    p = max(min(probs[label_index], 1 - 1e-12), 1e-12)
    return -math.log(p)


def _macro_f1(confusion: dict[str, dict[str, int]]) -> float:
    f1_scores = []
    for label in RESULT_LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in RESULT_LABELS if other != label)
        fn = sum(confusion[label][other] for other in RESULT_LABELS if other != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1_scores.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return sum(f1_scores) / len(f1_scores)


def _score_predictions(results: list[str], probs: list[list[float]]) -> dict[str, float]:
    confusion = {a: {b: 0 for b in RESULT_LABELS} for a in RESULT_LABELS}
    correct = 0
    total_log_loss = 0.0
    for result, match_probs in zip(results, probs):
        predicted = RESULT_LABELS[max(range(3), key=lambda i: match_probs[i])]
        confusion[result][predicted] += 1
        if predicted == result:
            correct += 1
        total_log_loss += _log_loss(match_probs, RESULT_LABELS.index(result))
    n = len(results)
    return {
        "accuracy": correct / n if n else 0.0,
        "log_loss": total_log_loss / n if n else 0.0,
        "macro_f1": _macro_f1(confusion),
        "n_matches": n,
    }


def evaluate_against_baseline(
    model_dir: str,
    matches_csv: str,
    profiles_csv: str | None = None,
) -> dict[str, Any]:
    profiles = load_team_profiles(profiles_csv) if profiles_csv else {}
    service = PredictionService.load(model_dir, team_profiles=profiles)
    matches = load_completed_matches(matches_csv)

    results = []
    model_probs = []
    baseline_probs = []
    xg_errors_home = []
    xg_errors_away = []
    goal_errors_home = []
    goal_errors_away = []

    for match in matches:
        prediction = service.predict_match(
            match["home_team"], match["away_team"], competition="World Cup", stage=match["stage"]
        )
        results.append(match["result"])
        model_probs.append([prediction["home_win_prob"], prediction["draw_prob"], prediction["away_win_prob"]])

        goal_errors_home.append(abs(prediction["expected_home_goals"] - match["home_goals"]))
        goal_errors_away.append(abs(prediction["expected_away_goals"] - match["away_goals"]))
        if match["home_xg"] is not None and match["away_xg"] is not None:
            xg_errors_home.append(abs(prediction["expected_home_goals"] - match["home_xg"]))
            xg_errors_away.append(abs(prediction["expected_away_goals"] - match["away_xg"]))

        if match["odds_home"] and match["odds_draw"] and match["odds_away"]:
            baseline = OddsBaseline.from_decimal_odds(match["odds_home"], match["odds_draw"], match["odds_away"])
            baseline_probs.append([baseline.home_win, baseline.draw, baseline.away_win])
        else:
            baseline_probs.append(None)

    baseline_results = [r for r, p in zip(results, baseline_probs) if p is not None]
    baseline_probs_present = [p for p in baseline_probs if p is not None]

    def _mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "n_matches_evaluated": len(matches),
        "model": _score_predictions(results, model_probs),
        "sportsbook_baseline": _score_predictions(baseline_results, baseline_probs_present),
        "goal_mae_vs_actual": {
            "home": _mean(goal_errors_home),
            "away": _mean(goal_errors_away),
        },
        "goal_mae_vs_match_xg": {
            "home": _mean(xg_errors_home),
            "away": _mean(xg_errors_away),
            "n_matches_with_xg": len(xg_errors_home),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the model against the sportsbook-odds baseline.")
    parser.add_argument("--model-dir", default="model/saved_model")
    parser.add_argument(
        "--matches",
        default="data/raw/international-world-cup-matches-2026-to-2026-stats.csv",
        help="Path to the WC2026 match-level stats CSV (xG + odds).",
    )
    parser.add_argument("--profiles", default="data/raw/team_profiles.csv")
    args = parser.parse_args()
    payload = evaluate_against_baseline(
        model_dir=args.model_dir,
        matches_csv=args.matches,
        profiles_csv=args.profiles,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
