# World Cup Prediction Project

This project builds a deep-learning-based World Cup prediction pipeline for the men's tournament. It learns from historical match data, predicts scorelines and match outcomes, and simulates the tournament bracket to estimate advancement chances.

## Documentation

- `docs/problem_statement.md`
- `docs/data_plan.md`
- `docs/model_architecture.md`
- `docs/system_architecture.md`
- `docs/tasks.md`
- `docs/evaluation_plan.md`
- `docs/data_guide.md` — plain-language overview of the data sources, for non-technical readers
- `docs/model_guide.md` — plain-language overview of how the model reaches a prediction, for non-technical readers
- `docs/metrics_report.md` — how accurate the model is, compared against the sportsbook market and against real historical World Cups

## Final Project Scope

The final system has four fixed stages:

1. Load and preprocess historical international football match data.
2. Convert each match into a numeric feature vector.
3. Train a small multilayer perceptron (MLP) to predict expected goals for both teams.
4. Convert the predicted scorelines into match results and use them in a tournament simulation.

## Baselines

The project also includes a sportsbook-odds and real-xG baseline, evaluated against the actual 2026 World Cup matches (see `app/evaluate.py`). Publicly available sportsbook odds are converted into implied probabilities and used as a comparison point for the MLP predictions.

## Final Deliverables

- A trained MLP classifier for match outcome prediction
- A tournament simulator that produces World Cup advancement probabilities and score-by-score results
- A comparison against sportsbook-odds and real expected-goals baselines
- A written explanation of the features, model, and simulation procedure

## How To Run

### 1. Train the model

```bash
python -m app.train --matches data/raw/matches.csv --profiles data/raw/team_profiles.csv --output model/saved_model
```

### 2. Run the web application

```bash
python -m app.app
```

Then open the local server in your browser. The app already ships with the real 2026 group draw, team flags, and a trained model, so it runs end to end out of the box.

### 3. Simulate from the command line

```bash
python -m app.simulate --model-dir model/saved_model --tournament data/raw/tournament.json --profiles data/raw/team_profiles.csv --runs 1
```

Add `--runs 200` to run a Monte Carlo estimate (advancement/championship probabilities averaged over many simulated tournaments) instead of a single playthrough. A progress line is printed while it runs.

### 4. Evaluate against the sportsbook/xG baseline

```bash
python -m app.evaluate --model-dir model/saved_model --profiles data/raw/team_profiles.csv
```

Compares the model's predictions for the already-played 2026 World Cup matches against real sportsbook odds and real expected-goals (xG) data.

### 5. Backtest against real past World Cups

```bash
python -m app.backtest --model-dir model/saved_model --profiles data/raw/team_profiles.csv
```

Compares the model's predicted outcome for every official World Cup match from 1930 to 2022 against what actually happened, broken down by edition. See `docs/metrics_report.md` for the write-up, including which editions are genuinely held-out vs. part of training.

## Project Notes

- The backend loads a saved model and does not retrain on every button click.
- The frontend sends one request per simulation and renders the matches step by step.
- Real team data (Elo, FIFA ranking) lives in `data/raw/team_profiles.csv`; the backend falls back to it automatically if a request doesn't provide profiles.
- The real 2026 group draw and team metadata (flags, codes) live in `web/config.js`.
- Flag images live under `web/assets/flags/`.
- `data/raw/fifa/` is a separate, unused exploratory pipeline kept for reference — it was not integrated into the final model.
