# Problem Statement

We are building a World Cup simulation system for the men's tournament that predicts match scorelines, match outcomes, and tournament progression from historical football data using a compact neural network.

The model takes two teams, match context, and team-strength features as input and outputs the expected goals for each team. These predicted scorelines are then converted into win, draw, and loss probabilities and used to simulate the full tournament bracket.

## Objective

The objective is to estimate how likely each team is to advance through the current men's World Cup and win the title, using a model that is simple enough to train, explain, and demonstrate within one week.

## Core Questions

- What scoreline is most likely for a given match?
- What is the probability that Team A wins, draws, or loses against Team B?
- Which teams are most likely to qualify from the group stage?
- Which teams are most likely to advance as one of the best third-placed teams?
- Which teams are most likely to win the tournament?

## Modeling Principle

The project does not rely on manually assigned weights. Instead, the neural network learns the relationship between the match features and the expected goals directly from historical match data.

This keeps the project data-driven, technically grounded, and aligned with a deep learning workflow.

## Baseline Comparison

The neural network will be evaluated against a sportsbook-odds baseline.

Publicly available sportsbook odds are converted into implied win/draw/loss probabilities and used as a benchmark to measure whether the learned model adds predictive value beyond the market consensus.

## Scope

The project focuses on the men's World Cup only.

Player-level statistics are excluded from the initial version to keep the pipeline small, reproducible, and feasible within the project timeline.

The simulation includes:

- group stage
- ranking of third-placed teams
- round of 32
- round of 16
- quarterfinals
- semifinals
- final
