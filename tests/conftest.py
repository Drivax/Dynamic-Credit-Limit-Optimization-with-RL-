from dataclasses import replace

import pytest

from credit_rl.config import SimulationConfig
from credit_rl.simulation.customer import CustomerState, CustomerTraits


@pytest.fixture
def initial_state():
    return CustomerState(customer_id="customer-7", month=0, credit_limit=4000.0,
                         balance=2500.0, payment_ratio=0.4, months_delinquent=0,
                         income=3000.0, behavioral_score=650.0, initial_score=650.0,
                         tenure_months=36, monthly_spend=1500.0, late_history=(0,) * 6)


@pytest.fixture
def traits():
    return CustomerTraits(creditworthiness=0.0, spending_propensity=1.0, payment_propensity=0.4)


@pytest.fixture
def no_default_config():
    c = SimulationConfig()
    # Deterministic underflow to zero for lifecycle tests, not a research setting.
    return replace(c, default=replace(c.default, intercept=-1000.0))
