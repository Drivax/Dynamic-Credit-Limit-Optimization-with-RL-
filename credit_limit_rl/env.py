from __future__ import annotations

from typing import Mapping

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .config import PortfolioConfig
from .risk import RISK_FEATURES

ACTION_ADJUSTMENTS = np.array([-0.20, -0.10, 0.0, 0.10, 0.20], dtype=np.float32)
ACTION_LABELS = ["-20%", "-10%", "0%", "+10%", "+20%"]


def _to_series(record: pd.Series | Mapping[str, float]) -> pd.Series:
    if isinstance(record, pd.Series):
        return record
    return pd.Series(record)


def build_observation(record: pd.Series | Mapping[str, float]) -> np.ndarray:
    client = _to_series(record)
    observation = np.array(
        [
            client["internal_score"] / 1000.0,
            client["utilization"],
            client["delinquency_30d"] / 4.0,
            client["late_payments_6m"] / 5.0,
            min(client["monthly_spend"] / max(client["current_limit"], 1.0), 2.0),
            min(client["current_balance"] / max(client["income"], 1.0), 2.5),
            client["macro_unemployment"] * 10.0,
            client["macro_inflation"] * 20.0,
            min(client["income"] / 10000.0, 1.5),
            min(client["tenure_months"] / 180.0, 1.0),
        ],
        dtype=np.float32,
    )
    return observation


def _risk_feature_frame(client: pd.Series, proposed_limit: float, adjusted_spend: float) -> pd.DataFrame:
    utilization = np.clip(client["current_balance"] / max(proposed_limit, 1.0), 0.01, 1.5)
    debt_to_income = np.clip((client["current_balance"] + 0.35 * adjusted_spend) / max(client["income"], 1.0), 0.01, 3.5)
    row = {
        "income": client["income"],
        "tenure_months": client["tenure_months"],
        "internal_score": client["internal_score"],
        "utilization": utilization,
        "monthly_spend": adjusted_spend,
        "delinquency_30d": client["delinquency_30d"],
        "late_payments_6m": client["late_payments_6m"],
        "macro_unemployment": client["macro_unemployment"],
        "macro_inflation": client["macro_inflation"],
        "debt_to_income": debt_to_income,
    }
    return pd.DataFrame([row], columns=RISK_FEATURES)


def simulate_credit_decision(
    record: pd.Series | Mapping[str, float],
    adjustment: float,
    risk_model,
    config: PortfolioConfig,
    rng: np.random.Generator | None = None,
) -> dict[str, float | int]:
    client = _to_series(record)
    generator = rng or np.random.default_rng()

    current_limit = float(client["current_limit"])
    current_balance = float(client["current_balance"])
    proposed_limit = float(np.clip(current_limit * (1.0 + adjustment), config.min_limit, config.max_limit))
    spend_multiplier = float(np.clip(1.0 + 0.65 * adjustment, 0.75, 1.20))
    adjusted_spend = float(max(20.0, client["monthly_spend"] * spend_multiplier))

    risk_frame = _risk_feature_frame(client, proposed_limit, adjusted_spend)
    predicted_pd = float(risk_model.predict_proba(risk_frame)[0, 1])

    utilization_delta = float(risk_frame.iloc[0]["utilization"] - client["utilization"])
    realized_pd = float(np.clip(client["true_pd"] * (1.0 + 1.2 * utilization_delta), 0.002, 0.65))
    defaulted = int(generator.random() < realized_pd)

    monthly_interest = current_balance * config.annual_percentage_rate / 12.0
    fee_income = adjusted_spend * config.fee_rate
    expected_loss = predicted_pd * config.loss_given_default * current_balance
    rwa_cost = config.rwa_factor * predicted_pd * current_balance

    constraint_penalty = 0.0
    if predicted_pd > config.max_pd_threshold:
        constraint_penalty += 25.0 * (predicted_pd - config.max_pd_threshold)

    reward = monthly_interest + fee_income - config.lambda_default * expected_loss - config.lambda_rwa * rwa_cost - constraint_penalty
    if defaulted:
        reward -= config.loss_given_default * current_balance

    return {
        "new_limit": proposed_limit,
        "adjusted_spend": adjusted_spend,
        "predicted_pd": predicted_pd,
        "realized_pd": realized_pd,
        "defaulted": defaulted,
        "monthly_interest": monthly_interest,
        "fee_income": fee_income,
        "expected_loss": expected_loss,
        "rwa_cost": rwa_cost,
        "constraint_penalty": constraint_penalty,
        "reward": reward,
    }


class CreditLimitEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        portfolio: pd.DataFrame,
        risk_model,
        config: PortfolioConfig | None = None,
        episode_length: int = 256,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.portfolio = portfolio.reset_index(drop=True).copy()
        self.risk_model = risk_model
        self.config = config or PortfolioConfig()
        self.episode_length = min(episode_length, len(self.portfolio))
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(len(ACTION_ADJUSTMENTS))
        self.observation_space = spaces.Box(low=0.0, high=2.5, shape=(10,), dtype=np.float32)
        self.static_action_index = 2
        self._episode_indices: np.ndarray | None = None
        self._position = 0
        self._cumulative_pd = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        replace = len(self.portfolio) < self.episode_length
        self._episode_indices = self.rng.choice(len(self.portfolio), size=self.episode_length, replace=replace)
        self._position = 0
        self._cumulative_pd = 0.0
        first_client = self.portfolio.iloc[int(self._episode_indices[self._position])]
        return build_observation(first_client), {}

    def step(self, action: int):
        if self._episode_indices is None:
            raise RuntimeError("Environment must be reset before calling step().")

        client = self.portfolio.iloc[int(self._episode_indices[self._position])]
        outcome = simulate_credit_decision(client, float(ACTION_ADJUSTMENTS[action]), self.risk_model, self.config, self.rng)

        self._cumulative_pd += float(outcome["predicted_pd"])
        avg_pd = self._cumulative_pd / float(self._position + 1)
        portfolio_penalty = 0.0
        if avg_pd > self.config.portfolio_pd_threshold:
            portfolio_penalty = 20.0 * (avg_pd - self.config.portfolio_pd_threshold)

        reward = float(outcome["reward"]) - portfolio_penalty
        self._position += 1
        terminated = self._position >= self.episode_length

        if terminated:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            next_client = self.portfolio.iloc[int(self._episode_indices[self._position])]
            observation = build_observation(next_client)

        info = {
            "predicted_pd": float(outcome["predicted_pd"]),
            "defaulted": int(outcome["defaulted"]),
            "new_limit": float(outcome["new_limit"]),
            "reward": reward,
            "portfolio_penalty": portfolio_penalty,
            "avg_portfolio_pd": avg_pd,
        }
        return observation, reward, terminated, False, info
