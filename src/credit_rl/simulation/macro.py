"""Exogenous regimes and immutable, serializable paths shared across customers/policies."""

import json
from dataclasses import asdict, dataclass
from enum import IntEnum
from math import isfinite
from pathlib import Path

import numpy as np

from credit_rl.config import MacroConfig


class MacroRegime(IntEnum):
    EXPANSION = 0
    NORMAL = 1
    STRESS = 2


@dataclass(frozen=True)
class MacroState:
    regime: MacroRegime = MacroRegime.NORMAL
    income_growth: float = 0.001
    spending_growth: float = 0.0
    credit_stress: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.regime, MacroRegime):
            raise ValueError("regime must be a MacroRegime")
        values = self.income_growth, self.spending_growth, self.credit_stress
        if not all(isfinite(v) for v in values):
            raise ValueError("Macro factors must be finite")
        if abs(self.income_growth) > 0.2 or abs(self.spending_growth) > 0.5 or not 0 <= self.credit_stress <= 3:
            raise ValueError("Macro factors exceed supported domain")

    @classmethod
    def from_config(cls, regime: MacroRegime, config: MacroConfig, severity: float = 1.0) -> "MacroState":
        return cls(regime, config.income_growth[regime]*severity,
                   config.spending_growth[regime]*severity, config.credit_stress[regime]*severity)


@dataclass(frozen=True)
class MacroPath:
    """H+1 states: month zero through terminal observation, H transition regimes."""
    states: tuple[MacroState, ...]
    name: str = "custom"

    def __post_init__(self) -> None:
        if not isinstance(self.states, tuple) or len(self.states) < 2:
            raise ValueError("MacroPath requires an immutable tuple of at least two states")
        if not all(isinstance(s, MacroState) for s in self.states):
            raise ValueError("Invalid macro path entry")

    @property
    def horizon(self) -> int:
        return len(self.states)-1

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"name": self.name, "states": [asdict(s) for s in self.states]}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MacroPath":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        states = tuple(MacroState(**{**s, "regime": MacroRegime(s["regime"])}) for s in data["states"])
        return cls(states, data["name"])

    @classmethod
    def constant(cls, state: MacroState, horizon: int) -> "MacroPath":
        _validate_horizon(horizon)
        return cls((state,)*(horizon+1), f"constant_{state.regime.name.lower()}")


def _validate_horizon(horizon: int) -> None:
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")


class MacroProcess:
    def __init__(self, config: MacroConfig):
        self.config = config

    def generate(self, horizon: int, rng: np.random.Generator,
                 initial: MacroState | None = None) -> MacroPath:
        """Sample a persistent Markov chain before customer behavior is simulated."""
        _validate_horizon(horizon)
        current = initial or MacroState.from_config(MacroRegime.NORMAL, self.config)
        states = [current]
        for _ in range(horizon):
            regime = MacroRegime(int(rng.choice(3, p=self.config.transition_matrix[current.regime])))
            current = MacroState.from_config(regime, self.config)
            states.append(current)
        return MacroPath(tuple(states), "markov")

    def scenario(self, name: str, horizon: int) -> MacroPath:
        """Piecewise deterministic intervention, including severity within STRESS.

        Phases are fractions of the horizon. At short horizons a phase can contain
        no monthly decision. The final observation repeats the last regime.
        """
        _validate_horizon(horizon)
        phases = self.config.scenarios[name]
        boundaries = np.cumsum([p[0] for p in phases])
        states = []
        for month in range(horizon):
            index = min(int(np.searchsorted(boundaries, month/horizon, side="right")), len(phases)-1)
            _, regime, severity = phases[index]
            states.append(MacroState.from_config(MacroRegime(regime), self.config, severity))
        return MacroPath(tuple(states + [states[-1]]), name)
