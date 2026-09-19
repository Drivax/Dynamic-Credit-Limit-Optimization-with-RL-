"""Observable lending guardrails, separate from hidden customer dynamics."""
import numpy as np


def effective_limit(limit, months_delinquent, multiplier, config, severe_delinquency_months=None):
    blocked = (severe_delinquency_months is not None
               and months_delinquent >= severe_delinquency_months and multiplier > 1)
    admitted = 1.0 if blocked else multiplier
    admitted = float(np.clip(admitted, 1-config.max_monthly_decrease, 1+config.max_monthly_increase))
    return float(np.clip(limit*admitted, config.min_limit, config.max_limit)), blocked
