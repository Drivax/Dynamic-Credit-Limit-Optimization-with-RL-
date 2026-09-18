"""Shared experiment configuration and provenance; no training side effects."""

import hashlib
import json
import platform
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

from credit_rl.config import SimulationConfig


def load_run_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {"seed", "customers", "risk_training_clients", "stress_trials", "ppo_timesteps",
                "ppo_eval_customers", "ppo"}
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError(f"Experiment config must contain {sorted(expected)}")
    for key in expected - {"ppo"}:
        if not isinstance(config[key], int) or config[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid {key}")
    return config


def write_manifest(path: Path, config: SimulationConfig, run: dict, **extra) -> None:
    packages = {}
    for name in ("credit-rl", "numpy", "pandas", "gymnasium", "scikit-learn", "matplotlib",
                 "stable-baselines3", "torch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "src" / "credit_rl").rglob("*.py"))
    sources += sorted((root / "experiments").glob("*.py"))
    path.write_text(json.dumps({
        "environment_version": "longitudinal-v1", "python": platform.python_version(),
        "packages": packages, "simulation": asdict(config), "run": run,
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        **extra,
    }, indent=2), encoding="utf-8")
