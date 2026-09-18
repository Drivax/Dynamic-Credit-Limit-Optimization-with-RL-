"""Reusable common random numbers indexed by customer, month and named channel."""

import hashlib
import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MonthlyShocks:
    income_normal: float
    income_uniform: float
    spending_normal: float
    payment_normal: float
    missed_uniform: float
    score_normal: float
    default_uniform: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isfinite(value) or (name.endswith("uniform") and not 0 <= value < 1):
                raise ValueError(f"Invalid shock {name}")


@dataclass(frozen=True)
class ShockPath:
    customer_id: str
    seed: int
    months: tuple[MonthlyShocks, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.months, tuple) or not self.months or not all(isinstance(s, MonthlyShocks) for s in self.months):
            raise ValueError("Shock path must be a nonempty immutable monthly sequence")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer")

    @classmethod
    def generate(cls, customer_id: str, seed: int, horizon: int) -> "ShockPath":
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        # Python hash() is salted across processes; use a stable, portable digest.
        identity = hashlib.blake2b(str(customer_id).encode(), digest_size=16).digest()
        words = [int.from_bytes(identity[i:i+4], "little") for i in range(0, 16, 4)]
        channels = {}
        for name in MonthlyShocks.__dataclass_fields__:
            channel = int.from_bytes(hashlib.blake2b(name.encode(), digest_size=4).digest(), "little")
            rng = np.random.default_rng(np.random.SeedSequence([seed, *words, channel]))
            channels[name] = rng.random(horizon) if name.endswith("uniform") else rng.normal(size=horizon)
        months = tuple(MonthlyShocks(**{k: float(v[t]) for k, v in channels.items()}) for t in range(horizon))
        return cls(str(customer_id), seed, months)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ShockPath":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["customer_id"], data["seed"], tuple(MonthlyShocks(**m) for m in data["months"]))
