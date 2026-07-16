# Data Plan

## Data Sources

The project uses publicly available historical football datasets. The primary source is a match-level dataset of men's World Cup matches.

Preferred data types:

- Historical men's World Cup match results
- FIFA ranking information
- Elo-style team strength information
- Goal statistics
- Tournament stage or competition type

If the World Cup-only dataset is too small for stable training, the dataset is augmented with other official men's international matches while keeping World Cup matches as the main evaluation focus.

In addition, publicly available sportsbook odds will be collected for the same matches whenever possible. These odds are not the main training signal; they are used as a baseline for comparison.

## Data Structure

Each row represents one match and contains:

- `team_home`
- `team_away`
- `score_home`
- `score_away`
- `competition`
- `stage`
- `date`

Additional team attributes are merged into the match table before training.

## Feature Engineering

The model input is built from match-level differences and contextual variables.

Final feature set:

- Elo difference
- FIFA ranking difference
- Recent win rate difference
- Recent goal difference difference
- Recent goals scored per match difference
- Recent goals conceded per match difference
- Tournament stage encoding

For every match, the feature vector is constructed so that positive values indicate an advantage for the home team.

For World Cup matches, no home-advantage or neutral-site feature is included because the tournament is played on neutral ground.

## Sportsbook Odds Baseline

For each match, sportsbook odds will be converted into implied probabilities for:

- home win
- draw
- away win

These implied probabilities will be stored separately from the training features and used only for baseline evaluation.

## Target Variable

The main learning target is a pair of score values:

- home goals
- away goals

The model predicts expected goals for both teams, and those predictions are later converted into win/draw/loss probabilities and simulated scorelines.

## Preprocessing Rules

- Remove rows with missing essential match fields.
- Standardize numeric features.
- Encode categorical context variables when required.
- Split the data into train, validation, and test sets before fitting the model.

## Assumptions

- The initial project version focuses on the current men's World Cup.
- Player-level statistics are not required.
- If multiple datasets contain overlapping fields, the match-level table is the source of truth.
- Sportsbook odds are treated as benchmark data, not as the primary prediction target.
- The tournament is modeled as neutral-site, so no home-advantage feature is used.
