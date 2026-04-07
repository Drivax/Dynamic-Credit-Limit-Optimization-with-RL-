from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def generate_synthetic_portfolio(n_clients: int = 200_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    regions = np.array(["ile_de_france", "nord", "sud", "est", "ouest"])
    unemployment_map = {
        "ile_de_france": 0.071,
        "nord": 0.089,
        "sud": 0.083,
        "est": 0.077,
        "ouest": 0.068,
    }

    region = rng.choice(regions, size=n_clients, p=[0.24, 0.18, 0.24, 0.16, 0.18])
    unemployment = np.array([unemployment_map[item] for item in region]) + rng.normal(0.0, 0.004, n_clients)
    inflation = np.clip(rng.normal(0.024, 0.005, n_clients), 0.01, 0.05)
    age = rng.integers(21, 78, n_clients)
    tenure_months = rng.integers(2, 180, n_clients)
    income = rng.lognormal(mean=8.0, sigma=0.42, size=n_clients)
    income = np.clip(income, 1100.0, 12000.0)
    internal_score = np.clip(rng.normal(650.0, 85.0, n_clients), 350.0, 900.0)
    delinquency_30d = np.clip(rng.poisson(0.2, n_clients), 0, 4)
    late_payments_6m = np.clip(rng.poisson(0.35, n_clients), 0, 5)
    requested_limit = income * rng.uniform(1.2, 2.8, n_clients) + (internal_score - 600.0) * 9.0
    current_limit = np.clip(requested_limit, 500.0, 15000.0)
    utilization = np.clip(rng.beta(2.2, 3.6, n_clients) + 0.015 * delinquency_30d, 0.02, 0.98)
    current_balance = current_limit * utilization
    monthly_spend = np.clip(current_balance * rng.uniform(0.55, 1.4, n_clients), 80.0, None)
    debt_to_income = np.clip((current_balance + 0.35 * monthly_spend) / income, 0.01, 3.2)

    # Latent score transformed to monthly PD with a logistic link.
    latent_default = (
        -4.35
        + 3.05 * utilization
        + 0.48 * delinquency_30d
        + 0.35 * late_payments_6m
        - 0.0052 * (internal_score - 650.0)
        + 14.0 * (unemployment - 0.07)
        + 8.0 * (inflation - 0.02)
        + 0.62 * debt_to_income
        - 0.003 * np.minimum(tenure_months, 120)
        + 0.015 * np.maximum(age - 60, 0)
    )

    true_pd = np.clip(_sigmoid(latent_default), 0.002, 0.45)
    default_next_month = rng.binomial(1, true_pd)

    # Keep both realized labels and structural drivers for RL and risk modeling.
    portfolio = pd.DataFrame(
        {
            "region": region,
            "age": age,
            "tenure_months": tenure_months,
            "income": income,
            "internal_score": internal_score,
            "delinquency_30d": delinquency_30d.astype(float),
            "late_payments_6m": late_payments_6m.astype(float),
            "macro_unemployment": unemployment,
            "macro_inflation": inflation,
            "current_limit": current_limit,
            "current_balance": current_balance,
            "monthly_spend": monthly_spend,
            "utilization": utilization,
            "debt_to_income": debt_to_income,
            "true_pd": true_pd,
            "default_next_month": default_next_month.astype(int),
        }
    )
    return portfolio
