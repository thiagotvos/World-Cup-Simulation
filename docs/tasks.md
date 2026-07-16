# Implementation Tasks

This document splits the project into backend and frontend tasks so the implementation can be planned in small, clear steps.

## Backend Tasks

### 1. Data Preparation

- collect the historical World Cup dataset
- collect FIFA ranking data
- collect Elo-style team strength data
- collect sportsbook odds for baseline comparison
- clean missing values and unify team names
- build the final match-level table

### 2. Feature Engineering

- compute team-strength differences
- compute recent form features
- compute goal-based features
- encode tournament stage
- standardize all numeric inputs
- define the final feature vector format

### 3. Model Training

- build the MLP score prediction model
- choose the final loss function and optimizer
- train the model on historical matches
- validate the model during training
- save the trained model and preprocessing artifacts

### 4. Baseline Handling

- convert sportsbook odds into implied probabilities
- store the baseline predictions separately from the model inputs
- evaluate the model against the odds baseline

### 5. Simulation Logic

- convert predicted expected goals into sampled scorelines
- compute match winners, draws, and goal difference
- implement group-stage standings
- implement third-place ranking
- build the round-of-32 bracket
- simulate the knockout rounds

### 6. API Layer

- create a backend endpoint for starting a simulation
- create an endpoint for returning one match result at a time or a full tournament summary
- load the saved model on startup or on first request
- keep the model in memory to avoid retraining
- return simulation results as JSON

### 7. Testing and Validation

- test the training pipeline end to end
- test prediction on a single match
- test score sampling
- test group-stage ranking logic
- test third-place qualification logic
- test the API response format

## Frontend Tasks

### 1. Landing Screen

- build a simple landing page
- add a main title and short description
- add a single `Start Simulation` button
- keep the layout minimal and easy to read

### 2. Simulation Display

- show the group-stage matches one by one
- animate each match result as it is returned
- show the scoreline, winner, and match status
- update the standings after each group-stage match
- display the best third-place teams when the group stage ends
- show the knockout bracket as it progresses

### 3. Results Visualization

- display the simulated champion
- display the final bracket
- show advancement probabilities if available
- show scoreline summaries for each stage

### 4. User Experience

- add a loading state while the backend is computing
- prevent repeated clicks while a simulation is running
- show a clear completion message when the simulation ends

### 5. Optional Extras

- add a replay button to run the simulation again
- add a mini summary panel with strongest teams
- add simple charts for probabilities or standings if time allows

## Recommended Order

1. Finish the backend data pipeline.
2. Train and save the model.
3. Implement the simulation logic.
4. Build the API endpoints.
5. Create the frontend page.
6. Connect the frontend to the backend.
7. Add animation and polish.

## Scope Rule

If time becomes tight, keep the backend correct first and simplify the frontend animation before adding extra visual features.
