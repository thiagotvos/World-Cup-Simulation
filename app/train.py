"""Training entry point for the World Cup MLP model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from .data import load_match_records, load_team_profiles
from .features import FeatureEncoder
from .model import ModelBundle, StandardScaler, build_model, match_result_probabilities


def _one_hot_result(record):
    if record.score_home > record.score_away:
        return [1.0, 0.0, 0.0]
    if record.score_home < record.score_away:
        return [0.0, 0.0, 1.0]
    return [0.0, 1.0, 0.0]


# Exhibition/regional competitions between entities that never play official
# FIFA national-team football (CONIFA, Island Games, Viva World Cup, ...).
# Their lopsided scorelines add noise without teaching anything about the
# teams this model actually needs to predict.
FRINGE_COMPETITION_KEYWORDS = (
    "conifa",
    "island games",
    "viva world cup",
    "peoples, cultures and tribes",
)

# Winsorize goal counts so a handful of extreme blowouts (e.g. 31-0) don't
# dominate the MSE/Huber loss gradient relative to hundreds of normal matches.
MAX_GOALS_PER_TEAM = 10


def clean_records(records):
    cleaned = []
    for record in records:
        competition_text = (record.competition or "").lower()
        if any(keyword in competition_text for keyword in FRINGE_COMPETITION_KEYWORDS):
            continue
        record.score_home = min(record.score_home, MAX_GOALS_PER_TEAM)
        record.score_away = min(record.score_away, MAX_GOALS_PER_TEAM)
        cleaned.append(record)
    return cleaned


def build_dataset(records, profiles=None, history_window: int = 5):
    encoder = FeatureEncoder(history_window=history_window)
    states = encoder.extract_initial_states(profiles)
    features = []
    targets = []
    labels = []

    for record in records:
        home_state = encoder.ensure_state(states, record.team_home, profiles)
        away_state = encoder.ensure_state(states, record.team_away, profiles)
        feature_vector = encoder.transform(record, home_state, away_state)
        features.append(feature_vector)
        targets.append([float(record.score_home), float(record.score_away)])
        labels.append(_one_hot_result(record))

        home_state.apply_match(record.score_home, record.score_away)
        away_state.apply_match(record.score_away, record.score_home)

    return encoder, np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def split_chronologically(x, y, labels, validation_ratio: float = 0.1, test_ratio: float = 0.1):
    total = len(x)
    if total < 10:
        raise ValueError("Not enough records to train the model.")
    test_size = max(1, int(total * test_ratio))
    val_size = max(1, int(total * validation_ratio))
    train_size = total - val_size - test_size
    if train_size <= 0:
        raise ValueError("Dataset is too small for the requested splits.")
    return (
        x[:train_size],
        y[:train_size],
        labels[:train_size],
        x[train_size:train_size + val_size],
        y[train_size:train_size + val_size],
        labels[train_size:train_size + val_size],
        x[train_size + val_size:],
        y[train_size + val_size:],
        labels[train_size + val_size:],
    )


def train_model(
    matches_csv: str,
    profiles_csv: str | None,
    output_dir: str,
    history_window: int = 5,
    epochs: int = 50,
    batch_size: int = 32,
):
    records = load_match_records(matches_csv)
    if not records:
        raise ValueError("No valid match records were loaded.")
    records = clean_records(records)
    if not records:
        raise ValueError("No match records left after cleaning.")
    profiles = load_team_profiles(profiles_csv)

    encoder, x, y_goals, y_result = build_dataset(records, profiles=profiles, history_window=history_window)
    x_train, y_train, result_train, x_val, y_val, result_val, x_test, y_test, result_test = split_chronologically(
        x, y_goals, y_result
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    model = build_model(input_dim=x_train_scaled.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(),
        metrics=["mae"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
    ]

    history = model.fit(
        x_train_scaled,
        y_train,
        validation_data=(x_val_scaled, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=callbacks,
    )

    predictions = model.predict(x_test_scaled, verbose=0)
    goal_mae = float(np.mean(np.abs(predictions - y_test)))
    goal_rmse = float(np.sqrt(np.mean((predictions - y_test) ** 2)))

    result_probabilities = np.asarray(
        [match_result_probabilities(pred[0], pred[1]) for pred in predictions],
        dtype=np.float32,
    )
    predicted_labels = np.argmax(result_probabilities, axis=1)
    actual_labels = np.argmax(result_test, axis=1)
    accuracy = float(np.mean(predicted_labels == actual_labels))

    eps = 1e-9
    log_loss = float(
        -np.mean(np.sum(result_test * np.log(np.clip(result_probabilities, eps, 1.0)), axis=1))
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    bundle = ModelBundle(model=model, scaler=scaler, encoder=encoder, feature_names=encoder.feature_names)
    bundle.save(output_path)
    report = {
        "goal_mae": goal_mae,
        "goal_rmse": goal_rmse,
        "result_accuracy": accuracy,
        "result_log_loss": log_loss,
        "train_size": len(x_train),
        "val_size": len(x_val),
        "test_size": len(x_test),
        "history": history.history,
    }
    (output_path / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the World Cup prediction model")
    parser.add_argument("--matches", required=True, help="Path to the historical match CSV file.")
    parser.add_argument("--profiles", default=None, help="Optional path to team profile CSV file.")
    parser.add_argument("--output", default="model/saved_model", help="Directory where model artifacts will be saved.")
    parser.add_argument("--history-window", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    report = train_model(
        matches_csv=args.matches,
        profiles_csv=args.profiles,
        output_dir=args.output,
        history_window=args.history_window,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
