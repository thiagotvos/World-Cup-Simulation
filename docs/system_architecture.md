# System Architecture

## Overview

The project is organized as a two-stage system:

1. **Offline training stage**: prepares the data, trains the MLP, evaluates the model, and saves the trained artifacts.
2. **Online inference stage**: loads the saved artifacts, runs score prediction and tournament simulation, and returns results to the web interface.

This separation ensures that the model is trained only once and reused many times during the user-facing simulation.

## Main Components

### 1. Data Pipeline

The data pipeline loads historical men's World Cup match data and converts it into model-ready features.

Responsibilities:

- load raw match data
- clean missing values
- merge team-level statistics
- create match-level feature vectors
- split data into train, validation, and test sets

### 2. Training Pipeline

The training pipeline fits the MLP on the processed dataset.

Responsibilities:

- standardize numeric features
- train the neural network
- monitor validation loss
- evaluate the final model
- save the trained model artifact

### 3. Model Artifact Storage

After training, the following artifacts are saved:

- trained MLP weights
- preprocessing objects
- feature metadata
- label mapping

These artifacts are reused during inference so the model does not need to be retrained when the user clicks the start button in the web app.

### 4. Inference and Simulation Service

The inference service loads the saved artifacts and exposes prediction functionality.

Responsibilities:

- receive match or tournament input
- preprocess the input using the saved feature pipeline
- predict expected goals for both teams
- sample match scores from the predicted goal distributions
- update group standings and third-place rankings
- build the knockout bracket
- return aggregated tournament outcomes

### 5. Web Frontend

The web frontend provides the user interface.

Recommended implementation:

- HTML for structure
- CSS for styling
- JavaScript for button actions and result rendering
- a lightweight Python backend for prediction requests

The frontend does not run model training. It only sends requests to the backend and displays the returned results.

## Runtime Flow

When the user clicks the simulation button, the following steps occur:

1. The frontend sends a request to the backend.
2. The backend loads the already trained model if it is not already in memory.
3. The backend preprocesses the selected match or tournament data.
4. The MLP produces expected goals for both teams.
5. The simulation engine samples scores and resolves the group stage.
6. The backend ranks the best third-placed teams and builds the round-of-32 bracket.
7. The backend returns the results to the frontend.
8. The frontend displays each match result step by step.

## Recommended File Structure

```text
project/
  app/
    train.py
    simulate.py
    predict.py
    app.py
  model/
    saved_model/
    preprocessing/
  data/
    raw/
    processed/
  web/
    index.html
    styles.css
    script.js
  docs/
    problem_statement.md
    data_plan.md
    model_architecture.md
    system_architecture.md
    evaluation_plan.md
```

## Why This Architecture Works

This design is appropriate for a one-week project because it:

- avoids retraining on every button click
- keeps the model and the interface separated
- makes the project easy to explain
- supports score-by-score animation in the frontend
- supports group-stage logic and third-place qualification
- allows a clean comparison against sportsbook odds

## Implementation Choice

The recommended implementation stack is:

- **Python** for training and inference
- **TensorFlow/Keras** for the MLP
- **Flask or FastAPI** for the backend
- **HTML/CSS/JavaScript** for the user interface

This stack is simple enough for a short project and flexible enough to support a polished demo.
