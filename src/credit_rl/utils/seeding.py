"""Local Generator streams for simulation; explicit global seeding for training."""

import random

import numpy as np


def episode_generators(rng: np.random.Generator) -> tuple[np.random.Generator, ...]:
    """Split initialization, behavior and macro streams so actions cannot steer macro draws."""
    sequence = np.random.SeedSequence(rng.integers(0, 2**32, size=4).tolist())
    return tuple(np.random.default_rng(child) for child in sequence.spawn(3))


def seed_everything(seed: int, *, torch: bool = False) -> None:
    """For external training libraries only; environment steps never use global RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    if torch:
        import torch as torch_module

        torch_module.manual_seed(seed)
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(seed)
