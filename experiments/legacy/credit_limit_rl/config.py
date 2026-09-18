from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    min_limit: float = 500.0
    max_limit: float = 15000.0
    annual_percentage_rate: float = 0.18
    fee_rate: float = 0.012
    loss_given_default: float = 0.55
    rwa_factor: float = 0.08
    lambda_default: float = 5.0
    lambda_rwa: float = 1.5
    max_pd_threshold: float = 0.12
    portfolio_pd_threshold: float = 0.10
