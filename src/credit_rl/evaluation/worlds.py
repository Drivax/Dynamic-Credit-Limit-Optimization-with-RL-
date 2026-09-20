"""Declared evaluation-world family; training configuration is never mutated."""
from dataclasses import asdict, dataclass, replace
import numpy as np

from credit_rl.config import SimulationConfig
from credit_rl.simulation.macro import MacroPath, MacroState, MacroRegime
from credit_rl.simulation.customer import initialize_customer
from credit_rl.simulation.synthetic_snapshot import generate_synthetic_portfolio
from credit_rl.simulation.shocks import ShockPath
from .scenarios import EvaluationScenario


@dataclass(frozen=True)
class EvaluationWorld:
    world_id: str
    seed: int
    kind: str
    config: SimulationConfig
    parameters: dict
    macro_spec: dict
    varied_parameter: str = ""

    def metadata(self):
        return dict(world_id=self.world_id, world_seed=self.seed, kind=self.kind,
            parameters=self.parameters, macro=self.macro_spec, varied_parameter=self.varied_parameter,
            environment=asdict(self.config))


def perturb(base, values):
    sections = {}
    for name, value in values.items():
        section, field = name.split(".")
        if section not in ("dynamics", "default"):
            raise ValueError("Only declared DGP sections may shift")
        sections.setdefault(section, {})[field] = float(value)
    cfg = replace(base, **{key: replace(getattr(base, key), **value) for key, value in sections.items()})
    # The base validator checks behavior bounds, but does not constrain hazard signs.
    if any(getattr(cfg.default, key) < 0 for key in asdict(cfg.default) if key != "intercept"):
        raise ValueError("Risk-factor signs must be preserved")
    return cfg


def stress_path(config, spec):
    horizon = config.environment.horizon
    onset, duration, recovery = [int(spec.get(k, 0)) for k in ("onset", "duration", "recovery")]
    intensity = float(spec.get("intensity", 0))
    if min(onset, duration, recovery, intensity) < 0 or onset > horizon:
        raise ValueError("Invalid macro path specification")
    states = []
    for t in range(horizon):
        if onset <= t < onset+duration:
            severity = intensity
        elif recovery and onset+duration <= t < onset+duration+recovery:
            severity = intensity*(1-(t-onset-duration+1)/(recovery+1))
        else:
            severity = 0.
        states.append(MacroState.from_config(MacroRegime.STRESS if severity else MacroRegime.NORMAL,
                                            config.macro, severity if severity else 1.))
    return MacroPath(tuple(states+[states[-1]]), "declared_stress_path")


def sample_evaluation_world(base, settings, seed, world_id):
    rng = np.random.default_rng(seed)
    values = {name: float(rng.uniform(*bounds)) for name, bounds in settings["parameters"].items()}
    macro = {name: int(rng.integers(bounds[0], bounds[1]+1)) if name != "intensity" else float(rng.uniform(*bounds))
             for name, bounds in settings["macro_ranges"].items()}
    cfg = perturb(base, values)
    stress_path(cfg, macro)  # Validate factors against the existing macro domain.
    return EvaluationWorld(world_id, seed, "randomized", cfg, values, macro)


def build_worlds(base, settings, profile):
    normal = dict(onset=0, duration=0, recovery=0, intensity=0.)
    worlds = [EvaluationWorld("nominal", settings["world_seed"], "nominal", base, {}, normal)]
    for i in range(profile["worlds"]):
        seed = int(np.random.SeedSequence([settings["world_seed"], i]).generate_state(1)[0])
        worlds.append(sample_evaluation_world(base, settings, seed, f"random_{i:03d}"))
    if profile["sensitivity"]:
        anchor = dict(onset=6, duration=8, recovery=4, intensity=1.)
        worlds.append(EvaluationWorld("oat_reference", 0, "oat", base, {}, anchor))
        for name, bounds in settings["parameters"].items():
            for label, value in zip(("low", "high"), bounds):
                worlds.append(EvaluationWorld(f"oat_{name.replace('.', '_')}_{label}", 0, "oat",
                    perturb(base, {name:value}), {name:value}, anchor, name))
        paths = {
            "early": dict(onset=3,duration=8,recovery=4,intensity=1.),
            "late": dict(onset=12,duration=8,recovery=4,intensity=1.),
            "prolonged": dict(onset=6,duration=14,recovery=4,intensity=1.),
            "slow_recovery": dict(onset=6,duration=8,recovery=8,intensity=1.),
            "unexpected": dict(onset=12,duration=8,recovery=2,intensity=2.),
        }
        worlds.extend(EvaluationWorld("macro_"+name,0,"macro",base,{},spec) for name,spec in paths.items())
    return worlds


def make_cohort(base, *, count, seed, namespace):
    """New held-out identities. Nominal initial traits are shared across shifted worlds."""
    population = generate_synthetic_portfolio(count, seed)
    result = []
    path = stress_path(base, {})
    for i, row in enumerate(population.to_dict("records")):
        identity = f"{namespace}_{i:06d}"
        customer_seed = int(np.random.SeedSequence([seed,i,17]).generate_state(1)[0])
        state, traits = initialize_customer(row, identity, base, np.random.default_rng(customer_seed))
        result.append(EvaluationScenario(namespace,identity,state,traits,
            ShockPath.generate(identity,seed,base.environment.horizon),path,customer_seed))
    return result


def in_world(cohort, world):
    path = stress_path(world.config, world.macro_spec)
    return [replace(s, macro_path=path) for s in cohort]
