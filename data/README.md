# Data Folder

This folder stores input data for the training pipeline.

## Expected Files

- `raw/matches.csv`: historical match-level data
- `raw/team_profiles.csv`: optional team ratings and FIFA rank data
- `raw/tournament.json`: optional tournament group configuration

## Match File Columns

The training script can work with the following columns:

- `team_home`
- `team_away`
- `score_home`
- `score_away`
- `date`
- `competition`
- `stage`

Optional rating columns:

- `elo_home`
- `elo_away`
- `fifa_home`
- `fifa_away`

The pipeline is intentionally tolerant of missing optional fields.

