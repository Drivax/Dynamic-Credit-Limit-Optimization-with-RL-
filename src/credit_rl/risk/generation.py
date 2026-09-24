"""Modeling cohorts generated from the unchanged longitudinal DGP."""
from dataclasses import replace

import numpy as np
import pandas as pd

from credit_rl.simulation.customer import initialize_customer
from credit_rl.simulation.dgp import CreditDGP
from credit_rl.simulation.shocks import ShockPath
from credit_rl.simulation.synthetic_snapshot import generate_synthetic_portfolio
from .features import observable_row


def generate_cohort(config, *, customers, cohort_index, cohort_start, seeds, macro_path,
                    policy="behavior", action_probabilities=(.1, .2, .4, .2, .1)):
    """Customer-indexed independent RNG streams; cohort index identical in paired runs."""
    population_seed = np.random.SeedSequence([seeds["dataset"], cohort_index]).generate_state(1)[0]
    population = generate_synthetic_portfolio(customers, int(population_seed))
    dgp, rows, diagnostics = CreditDGP(config), [], []
    horizon = config.environment.horizon
    multipliers = config.environment.action_multipliers
    for i, record in enumerate(population.to_dict("records")):
        identity = f"cohort{cohort_index}-{i}"
        rng = np.random.default_rng(np.random.SeedSequence([seeds["dgp"], cohort_index, i]))
        state, traits = initialize_customer(record, identity, config, rng)
        state = replace(state, macro_state=macro_path.states[0])
        shocks = ShockPath.generate(identity, seeds["dgp"], horizon)
        action_rng = np.random.default_rng(np.random.SeedSequence([seeds["policy"], cohort_index, i]))
        for t in range(horizon+1):
            row = dict(customer_id=identity, month=t, cohort_start=cohort_start,
                scheduled_end=horizon, defaulted=state.defaulted, macro_state=int(state.macro_state.regime),
                **observable_row(state))
            rows.append(row)
            if state.defaulted or t == horizon:
                break
            if policy == "behavior":
                action = int(action_rng.choice(len(multipliers), p=action_probabilities))
            elif policy == "static":
                action = multipliers.index(1.)
            elif policy == "always_increase":
                action = int(np.argmax(multipliers))
            elif policy == "always_decrease":
                action = int(np.argmin(multipliers))
            else:
                raise ValueError(f"Unknown policy {policy}")
            outcome = dgp.step(state, multipliers[action], traits, shocks.months[t], macro_path.states[t+1])
            diagnostics.append(dict(customer_id=identity, month=t+1,
                p_default_true=outcome.p_default_true, realized_default=outcome.state.defaulted,
                action_multiplier=multipliers[action]))
            state = outcome.state
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)
