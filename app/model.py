"""Neural network model, preprocessing scaler, and prediction bundle."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from .features import FeatureEncoder, TeamState


@dataclass
class StandardScaler:
    mean_: list[float] | None = None
    scale_: list[float] | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        x = np.asarray(x, dtype=np.float32)
        self.mean_ = x.mean(axis=0).astype(np.float32).tolist()
        scale = x.std(axis=0).astype(np.float32)
        scale = np.where(scale < 1e-6, 1.0, scale)
        self.scale_ = scale.tolist()
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted.")
        x = np.asarray(x, dtype=np.float32)
        mean = np.asarray(self.mean_, dtype=np.float32)
        scale = np.asarray(self.scale_, dtype=np.float32)
        return (x - mean) / scale

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean_, "scale": self.scale_}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StandardScaler":
        return cls(mean_=payload.get("mean"), scale_=payload.get("scale"))


def build_model(input_dim: int) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(2, activation="softplus"),
        ]
    )


def poisson_pmf(k: int, lam: float) -> float:
    lam = max(float(lam), 1e-6)
    return math.exp(-lam + (k * math.log(lam)) - math.lgamma(k + 1))


def match_result_probabilities(home_lambda: float, away_lambda: float, max_goals: int = 10) -> tuple[float, float, float]:
    home_prob = 0.0
    draw_prob = 0.0
    away_prob = 0.0
    for home_goals in range(max_goals + 1):
        p_home = poisson_pmf(home_goals, home_lambda)
        for away_goals in range(max_goals + 1):
            p_away = poisson_pmf(away_goals, away_lambda)
            joint = p_home * p_away
            if home_goals > away_goals:
                home_prob += joint
            elif home_goals < away_goals:
                away_prob += joint
            else:
                draw_prob += joint
    total = home_prob + draw_prob + away_prob
    if total <= 0:
        return 0.0, 1.0, 0.0
    return home_prob / total, draw_prob / total, away_prob / total


def sample_scoreline(home_lambda: float, away_lambda: float, rng: np.random.Generator) -> tuple[int, int]:
    home_lambda = max(float(home_lambda), 1e-6)
    away_lambda = max(float(away_lambda), 1e-6)
    return int(rng.poisson(home_lambda)), int(rng.poisson(away_lambda))


@dataclass
class ModelBundle:
    model: tf.keras.Model
    scaler: StandardScaler
    encoder: FeatureEncoder
    feature_names: list[str]

    def predict_expected_goals(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        x = self.scaler.transform(x)
        prediction = self.model.predict(x, verbose=0)[0]
        return np.asarray(prediction, dtype=np.float32)

    def predict_match(self, features: np.ndarray) -> dict[str, float]:
        expected_home, expected_away = self.predict_expected_goals(features)
        home_win, draw, away_win = match_result_probabilities(float(expected_home), float(expected_away))
        return {
            "expected_home_goals": float(expected_home),
            "expected_away_goals": float(expected_away),
            "home_win_prob": float(home_win),
            "draw_prob": float(draw),
            "away_win_prob": float(away_win),
        }

    def save(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.save(output_path / "model.keras")
        (output_path / "scaler.json").write_text(json.dumps(self.scaler.to_dict(), indent=2), encoding="utf-8")
        metadata = {
            "feature_names": self.feature_names,
            "history_window": self.encoder.history_window,
            "stage_categories": self.encoder.stage_categories,
            "competition_categories": self.encoder.competition_categories,
        }
        (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, output_dir: str | Path) -> "ModelBundle":
        output_path = Path(output_dir)
        model = tf.keras.models.load_model(output_path / "model.keras")
        scaler_payload = json.loads((output_path / "scaler.json").read_text(encoding="utf-8"))
        metadata = json.loads((output_path / "metadata.json").read_text(encoding="utf-8"))
        encoder = FeatureEncoder(history_window=int(metadata.get("history_window", 5)))
        scaler = StandardScaler.from_dict(scaler_payload)
        return cls(
            model=model,
            scaler=scaler,
            encoder=encoder,
            feature_names=list(metadata.get("feature_names", [])),
        )

