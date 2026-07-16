# Model Architecture

## Overview

The final model is a multilayer perceptron (MLP) trained on tabular match features.

This is the core learning component of the project. It replaces manual weighting rules with learned feature interactions and predicts expected goals for both teams.

## Input Layer

The input is a fixed-length numeric vector built from match features and team-strength differences.

Input examples:

- Elo difference
- FIFA ranking difference
- recent form features
- goal-based features
- stage indicators

All numeric features are standardized before training.

## Network Design

The final architecture is:

- Input layer
- Dense(64, ReLU)
- Dropout(0.2)
- Dense(32, ReLU)
- Dropout(0.2)
- Dense(16, ReLU)
- Dense(2, linear or softplus output)

The two outputs represent:

- expected home goals
- expected away goals

This architecture is small enough for fast training but expressive enough to capture non-linear feature interactions.

## Output Layer

The output is a pair of expected goal values, one for each team.

These expected goals are later converted into:

- scoreline samples
- win/draw/loss probabilities
- group standings
- knockout advancement probabilities

## Training Configuration

The model is trained with:

- Loss: Poisson loss or mean squared error on goal counts
- Optimizer: Adam
- Metric: MAE or RMSE on goals
- Batch size: 32 or 64
- Early stopping on validation loss
- Learning-rate reduction on plateau

## Score Interpretation

The model outputs expected goals for both teams. These expected goals are then used to sample an actual scoreline during simulation.

This scoreline-based design is important because the tournament simulation needs:

- match winners
- draws
- goal difference
- goals scored
- third-place ranking logic

## Baseline Comparison

The final model is compared against a sportsbook-odds baseline.

Sportsbook odds are converted into implied win/draw/loss probabilities and evaluated with the same result-level metrics used for the MLP.

Because World Cup matches are played on neutral ground, the model does not include a home-advantage or venue feature for tournament simulation.

## Simulation Layer

The trained MLP is not the final output of the project. Its predictions are passed into a Monte Carlo simulation engine.

Simulation procedure:

1. Predict the expected goals for both teams.
2. Sample a scoreline from those expectations.
3. Compute the match result and update group standings.
4. Rank the third-placed teams.
5. Build the round of 32 bracket.
6. Repeat the full tournament many times if a probabilistic summary is needed.

The simulation produces:

- scoreline for every match
- group standings
- best third-placed teams
- round-of-32 bracket
- title probability
- semifinal probability
- quarterfinal probability
- round-of-16 probability

## Why This Architecture

This architecture is the right size for a one-week project:

- It is genuinely deep learning.
- It is easy to implement.
- It trains quickly on tabular data.
- It supports score-by-score visualization in the frontend.
- It remains explainable in a final report.
- It can be evaluated against a strong market-based baseline.
