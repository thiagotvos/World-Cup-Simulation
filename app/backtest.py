"""Backtest the model's match-outcome predictions against real historical
World Cup results (1930-2022), grouped by edition. See docs/metrics_report.md
for the write-up of these results."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

from .data import load_match_records, load_team_profiles
from .predict import PredictionService

RESULT_LABELS = ("home", "draw", "away")

# The model was trained on matches up to 2016-03-24 and validated on matches
# up to 2021-10-10 (see app/train.py's chronological split). Editions at or
# before that cutoff were part of what the model learned from; only editions
# strictly after it are genuine held-out (never-seen) predictions.
TRAIN_CUTOFF_YEAR = 2016
VAL_CUTOFF_YEAR = 2021


def _edition_status(year: int) -> str:
    if year <= TRAIN_CUTOFF_YEAR:
        return "in_sample_train"
    if year <= VAL_CUTOFF_YEAR:
        return "in_sample_validation"
    return "held_out"


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


def backtest_world_cups(
    model_dir: str,
    matches_csv: str,
    profiles_csv: str | None = None,
    include_2026: bool = False,
) -> dict:
    profiles = load_team_profiles(profiles_csv) if profiles_csv else {}
    service = PredictionService.load(model_dir, team_profiles=profiles)

    records = load_match_records(matches_csv)
    wc_records = [
        r
        for r in records
        if r.competition == "FIFA World Cup" and (include_2026 or r.date.year != 2026)
    ]

    by_year: dict[int, list] = defaultdict(list)
    for record in wc_records:
        by_year[record.date.year].append(record)

    editions = {}
    all_results: list[str] = []
    all_probs: list[list[float]] = []
    held_out_results: list[str] = []
    held_out_probs: list[list[float]] = []

    for year in sorted(by_year):
        year_results = []
        year_probs = []
        for record in by_year[year]:
            prediction = service.predict_match(
                record.team_home, record.team_away, competition="World Cup", stage=record.stage or "group"
            )
            if record.score_home > record.score_away:
                actual = "home"
            elif record.score_home < record.score_away:
                actual = "away"
            else:
                actual = "draw"
            probs = [prediction["home_win_prob"], prediction["draw_prob"], prediction["away_win_prob"]]
            year_results.append(actual)
            year_probs.append(probs)

        status = _edition_status(year)
        metrics = _score_predictions(year_results, year_probs)
        metrics["status"] = status
        editions[str(year)] = metrics
        all_results.extend(year_results)
        all_probs.extend(year_probs)
        if status == "held_out":
            held_out_results.extend(year_results)
            held_out_probs.extend(year_probs)

    overall = _score_predictions(all_results, all_probs)
    held_out_overall = _score_predictions(held_out_results, held_out_probs)

    return {
        "editions": editions,
        "overall_all_editions": overall,
        "overall_held_out_only": held_out_overall,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the model against real historical World Cup results.")
    parser.add_argument("--model-dir", default="model/saved_model")
    parser.add_argument("--matches", default="data/raw/matches.csv")
    parser.add_argument("--profiles", default="data/raw/team_profiles.csv")
    parser.add_argument("--include-2026", action="store_true", help="Also include the in-progress 2026 edition.")
    args = parser.parse_args()
    payload = backtest_world_cups(
        model_dir=args.model_dir,
        matches_csv=args.matches,
        profiles_csv=args.profiles,
        include_2026=args.include_2026,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
