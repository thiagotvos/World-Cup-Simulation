# World Cup Prediction Project

This project builds a deep-learning-based World Cup prediction pipeline for the men's tournament. It learns from historical match data, predicts scorelines and match outcomes, and simulates the tournament bracket to estimate advancement chances.

## Documentation

- `docs/problem_statement.md`
- `docs/data_plan.md`
- `docs/model_architecture.md`
- `docs/system_architecture.md`
- `docs/tasks.md`
- `docs/evaluation_plan.md`

## Final Project Scope

The final system has four fixed stages:

1. Load and preprocess historical international football match data.
2. Convert each match into a numeric feature vector.
3. Train a small multilayer perceptron (MLP) to predict expected goals for both teams.
4. Convert the predicted scorelines into match results and use them in a tournament simulation.

## Baselines

The project also includes a sportsbook-odds baseline. Publicly available sportsbook odds are converted into implied probabilities and used as a comparison point for the MLP predictions.

## Final Deliverables

- A trained MLP classifier for match outcome prediction
- A tournament simulator that produces World Cup advancement probabilities and score-by-score results
- Plots and tables showing model performance
- A written explanation of the features, model, and simulation procedure
- A comparison against sportsbook-odds baselines

## How To Run

### 1. Train the model

```bash
python -m app.train --matches data/raw/matches.csv --profiles data/raw/team_profiles.csv --output model/saved_model
```

If you do not have a profiles file, omit the `--profiles` argument.

### 2. Run the web application

```bash
python -m app.app
```

Then open the local server in your browser.

### 3. Simulate from the command line

```bash
python -m app.simulate --model-dir model/saved_model --runs 1
```

## Project Notes

- The backend loads a saved model and does not retrain on every button click.
- The frontend sends one request per simulation and renders the matches step by step.
- The repository includes a demo model in `model/saved_model` so the interface can run immediately.
- To swap demo teams for real national teams, edit `web/config.js` and provide:
  - `window.WC_TOURNAMENT` with the actual group draw
  - `window.WC_TEAM_META` with team names, codes, and optional flag image paths
- Place flag images under `web/assets/flags/` and reference them from `web/config.js`.
