"""Immutable simulation inputs: held-out customers and reusable exogenous paths."""
from dataclasses import dataclass
import numpy as np

from credit_rl.simulation.customer import initialize_customer
from credit_rl.simulation.macro import MacroProcess
from credit_rl.simulation.shocks import ShockPath
from credit_rl.simulation.synthetic_snapshot import generate_synthetic_portfolio

PARTITION_CODES = {"train": 11, "validation": 22, "test": 33, "oot": 44}


@dataclass(frozen=True)
class EvaluationScenario:
    partition: str
    customer_id: str
    initial_state: object
    traits: object
    shock_path: object
    macro_path: object
    customer_seed: int

    def reset_options(self):
        return dict(initial_state=self.initial_state, traits=self.traits,
                    shock_path=self.shock_path, macro_path=self.macro_path)


def make_scenarios(config, settings, partition, macro="baseline", customers=None):
    code = PARTITION_CODES[partition]
    full_count = settings["population"][partition]
    count = full_count if customers is None else customers
    if not 1 <= count <= full_count:
        raise ValueError("Requested cohort subset is outside the configured population")
    master = settings["population"]["seed"]
    rng_seed = int(np.random.SeedSequence([master, code]).generate_state(1)[0])
    population = generate_synthetic_portfolio(full_count, rng_seed).iloc[:count]
    process = MacroProcess(config.macro)
    shared = None if macro == "markov" else process.scenario(macro, config.environment.horizon)
    scenarios = []
    for i, record in enumerate(population.to_dict("records")):
        identity = f"RL_{partition}_{i:05d}"
        seed = int(np.random.SeedSequence([master, code, i, 1]).generate_state(1)[0])
        state, traits = initialize_customer(record, identity, config, np.random.default_rng(seed))
        shocks = ShockPath.generate(identity, master+code, config.environment.horizon)
        path = shared or process.generate(config.environment.horizon,
            np.random.default_rng(np.random.SeedSequence([master, code, i, 2])))
        scenarios.append(EvaluationScenario(partition, identity, state, traits, shocks, path, seed))
    return scenarios


def assert_disjoint(*groups):
    seen = set()
    for scenarios in groups:
        ids = {s.customer_id for s in scenarios}
        if len(ids) != len(scenarios) or seen & ids:
            raise ValueError("Evaluation and training customer identities overlap")
        seen.update(ids)
