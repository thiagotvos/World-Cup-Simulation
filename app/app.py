"""Flask application exposing training and simulation endpoints."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

from .data import TeamProfile, TournamentConfig, build_demo_tournament, load_team_profiles, load_tournament_config
from .model import ModelBundle
from .simulate import simulate_world_cup
from .simulation import TournamentSimulationResult
from .train import train_model


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = BASE_DIR / "model" / "saved_model"
DEFAULT_TRAIN_FILE = BASE_DIR / "data" / "raw" / "matches.csv"
DEFAULT_PROFILES_FILE = BASE_DIR / "data" / "raw" / "team_profiles.csv"

app = Flask(__name__, static_folder=str(BASE_DIR / "web"), static_url_path="/")

_BUNDLE: ModelBundle | None = None


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def load_bundle() -> ModelBundle:
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = ModelBundle.load(DEFAULT_MODEL_DIR)
    return _BUNDLE


def _serialize(obj: Any):
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _serialize(value) for key, value in obj.items()}
    return obj


@app.get("/api/health")
def health():
    model_ready = DEFAULT_MODEL_DIR.exists() and (DEFAULT_MODEL_DIR / "model.keras").exists()
    return jsonify({"status": "ok", "model_ready": model_ready})


@app.post("/api/train")
def train_endpoint():
    payload = request.get_json(force=True, silent=True) or {}
    matches = payload.get("matches", str(DEFAULT_TRAIN_FILE))
    profiles = payload.get("profiles")
    output = payload.get("output", str(DEFAULT_MODEL_DIR))
    history_window = int(payload.get("history_window", 5))
    epochs = int(payload.get("epochs", 50))
    batch_size = int(payload.get("batch_size", 32))
    report = train_model(
        matches_csv=matches,
        profiles_csv=profiles,
        output_dir=output,
        history_window=history_window,
        epochs=epochs,
        batch_size=batch_size,
    )
    global _BUNDLE
    _BUNDLE = None
    return jsonify(report)


@app.post("/api/simulate")
def simulate_endpoint():
    payload = request.get_json(force=True, silent=True) or {}
    model_dir = payload.get("model_dir", str(DEFAULT_MODEL_DIR))
    tournament_payload = payload.get("tournament")
    profiles_payload = payload.get("profiles")
    runs = int(payload.get("runs", 1))
    seed = payload.get("seed", 42)

    tournament_config = None
    if isinstance(tournament_payload, dict):
        groups = tournament_payload.get("groups", {})
        tournament_config = TournamentConfig(groups={str(key): list(value) for key, value in groups.items()})
    else:
        tournament_path = payload.get("tournament_path")
        tournament_config = load_tournament_config(tournament_path) or build_demo_tournament()

    team_profiles: dict[str, TeamProfile] = {}
    if isinstance(profiles_payload, dict) and profiles_payload:
        for team, values in profiles_payload.items():
            team_profiles[team.lower()] = TeamProfile(
                team=team,
                elo=float(values.get("elo", 0.0)) if isinstance(values, dict) else 0.0,
                fifa_rank=float(values.get("fifa_rank", 0.0)) if isinstance(values, dict) else 0.0,
            )
    elif isinstance(profiles_payload, str):
        team_profiles = load_team_profiles(profiles_payload)
    else:
        # No usable profiles were sent by the caller: fall back to the real
        # team ratings instead of simulating every match with unknown (zeroed
        # out) strength for both sides.
        team_profiles = load_team_profiles(str(DEFAULT_PROFILES_FILE))

    payload = simulate_world_cup(
        model_dir=model_dir,
        tournament=tournament_config,
        team_profiles=team_profiles,
        profiles_csv=profiles_payload if isinstance(profiles_payload, str) else None,
        matches_csv=str(DEFAULT_TRAIN_FILE),
        runs=runs,
        seed=seed,
    )
    if payload["mode"] == "single_run":
        result: TournamentSimulationResult = payload["result"]
        return jsonify(
            {
                "mode": "single_run",
                "result": _serialize(result),
            }
        )
    summary = payload["summary"]
    summary["last_run"] = _serialize(summary["last_run"])
    return jsonify({"mode": "monte_carlo", "summary": _serialize(summary)})


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
