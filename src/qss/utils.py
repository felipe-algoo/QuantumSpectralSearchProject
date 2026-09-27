from __future__ import annotations
from pathlib import Path
import json
import hashlib
import numpy as np


def ensure_directory(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def hash_configuration(config_dict: dict) -> str:
    payload = json.dumps(config_dict, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_json(obj, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def normalized_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    obs = np.asarray(observed, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    denom = float(obs.max() - obs.min()) if obs.max() != obs.min() else 1.0
    return float(np.sqrt(np.mean((obs - pred) ** 2)) / denom)