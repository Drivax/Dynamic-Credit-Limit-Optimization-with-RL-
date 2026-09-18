"""One customer per episode, until default or a configurable monthly horizon."""

from dataclasses import asdict, replace
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from credit_rl.config import SimulationConfig
from credit_rl.reward import calculate_reward
from credit_rl.risk.pd_model import ObservedLogisticPD, ObservedRiskFeatures, PDModel
from credit_rl.simulation.customer import CustomerState, CustomerTraits, MacroState, initialize_customer
from credit_rl.simulation.dynamics import TransitionModel
from credit_rl.utils.seeding import episode_generators

# Dimension names are a public contract for deterministic baseline policies.
OBSERVATION_NAMES = (
    "month_fraction", "credit_limit_scaled", "balance_scaled", "utilization_bounded",
    "payment_ratio", "delinquency_status", "months_delinquent_bounded", "income_bounded",
    "behavioral_score_scaled", "macro_stress", "predicted_pd", "spend_bounded",
    "late_payments_fraction", "tenure_bounded", "defaulted",
)


class CreditLimitEnv(gym.Env[np.ndarray, int]):
    """Partially observed longitudinal research environment.

    Observations/info/history deliberately exclude latent traits and simulator
    probabilities. `options` accepts a customer_index or an explicit initial
    CustomerState + CustomerTraits pair for controlled simulation experiments.
    Snapshot rows are read only at reset; step never accesses the portfolio.
    """
    metadata = {"render_modes": []}

    def __init__(self, portfolio: pd.DataFrame | None = None, pd_model: PDModel | None = None,
                 config: SimulationConfig | None = None) -> None:
        super().__init__()
        self.config = config or SimulationConfig()
        if portfolio is not None and portfolio.empty:
            raise ValueError("portfolio must not be empty")
        # Only initialization fields survive ingestion, even if targets are supplied.
        required = ["income", "tenure_months", "internal_score", "current_limit",
                    "current_balance", "monthly_spend"]
        optional = ["customer_id", "delinquency_30d", "late_payments_6m"]
        self._portfolio = None if portfolio is None else portfolio[
            required + [name for name in optional if name in portfolio]].reset_index(drop=True).copy()
        self.pd_model = pd_model if pd_model is not None else ObservedLogisticPD(self.config.pd)
        self._transition = TransitionModel(self.config)
        self.action_space = spaces.Discrete(len(self.config.environment.action_multipliers))
        self.observation_space = spaces.Box(0.0, 1.0, (len(OBSERVATION_NAMES),), dtype=np.float32)
        self.static_action_index = self.config.environment.action_multipliers.index(1.0)
        self._state: CustomerState | None = None
        self._traits: CustomerTraits | None = None
        self._history: list[dict[str, Any]] = []
        self._done = True
        self._elapsed = 0
        self._predicted_pd = 0.0

    @property
    def state(self) -> CustomerState:
        """Immutable current observed state, safe to retain for before/after checks."""
        if self._state is None:
            raise RuntimeError("Call reset first")
        return self._state

    def _predict_pd(self, state: CustomerState) -> float:
        value = float(self.pd_model.predict(ObservedRiskFeatures.from_state(state)))
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("PD model must return a finite probability in [0, 1]")
        return value

    def _get_observation(self) -> np.ndarray:
        """Observable quantities only; x/(1+x) bounds ratios without clipping debt."""
        s = self.state
        cap = self.config.environment.max_limit
        d = self.config.dynamics
        bounded = lambda value: value / (1.0 + value)
        return np.array([
            self._elapsed / self.config.environment.horizon,
            s.credit_limit / cap, bounded(s.balance / cap), bounded(s.utilization),
            s.payment_ratio, s.delinquency_status, bounded(s.months_delinquent),
            bounded(s.income / cap),
            (s.behavioral_score - d.score_min) / (d.score_max - d.score_min),
            int(s.macro_state), self._predicted_pd, bounded(s.monthly_spend / cap),
            sum(s.late_history) / 6, bounded(s.tenure_months), int(s.defaulted),
        ], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        options = options or {}
        allowed = {"customer_index", "initial_state", "traits", "macro_state"}
        if set(options) - allowed:
            raise ValueError(f"Unknown reset options: {set(options) - allowed}")
        init_rng, self._behavior_rng, self._macro_rng = episode_generators(self.np_random)
        if "initial_state" in options:
            if "traits" not in options or "customer_index" in options:
                raise ValueError("initial_state requires traits and excludes customer_index")
            state, traits = options["initial_state"], options["traits"]
            if not isinstance(state, CustomerState) or not isinstance(traits, CustomerTraits):
                raise TypeError("Expected CustomerState and CustomerTraits")
        else:
            if "traits" in options:
                raise ValueError("traits requires initial_state")
            if self._portfolio is None:
                raise ValueError("Provide portfolio or explicit initial_state and traits")
            index = options.get("customer_index")
            if index is None:
                index = int(init_rng.integers(len(self._portfolio)))
            if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self._portfolio):
                raise ValueError("Invalid customer_index")
            row = self._portfolio.iloc[int(index)]
            state, traits = initialize_customer(row, str(row.get("customer_id", index)), self.config, init_rng)
        if state.defaulted:
            raise ValueError("Cannot initialize an active episode with a defaulted customer")
        e, d = self.config.environment, self.config.dynamics
        if not e.min_limit <= state.credit_limit <= e.max_limit:
            raise ValueError("Initial credit limit outside configured bounds")
        if not d.score_min <= state.behavioral_score <= d.score_max:
            raise ValueError("Initial score outside configured bounds")
        if "macro_state" in options:
            state = replace(state, macro_state=MacroState(options["macro_state"]))
        self._state, self._traits = state, traits
        self._elapsed, self._done = 0, False
        self._predicted_pd = self._predict_pd(state)
        self._history = [self._history_row(action=None, reward=0.0, terminated=False, truncated=False)]
        return self._get_observation(), self._info()

    def _info(self) -> dict[str, Any]:
        return {"customer_id": self.state.customer_id, "month": self.state.month,
                "predicted_pd": self._predicted_pd, "defaulted": self.state.defaulted,
                "credit_limit": self.state.credit_limit, "macro_state": int(self.state.macro_state)}

    def _history_row(self, *, action: int | None, reward: float, terminated: bool,
                     truncated: bool) -> dict[str, Any]:
        row = asdict(self.state)
        # Retain observable history counts, not initialization bookkeeping.
        row.pop("initial_score")
        row["late_payments_6m"] = sum(row.pop("late_history"))
        row.update(utilization=self.state.utilization, delinquency_status=self.state.delinquency_status,
                   predicted_pd=self._predicted_pd, action=action, reward=reward,
                   terminated=terminated, truncated=truncated, macro_state=int(self.state.macro_state))
        return row

    def step(self, action: int):
        if self._done:
            raise RuntimeError("Episode is inactive; call reset before step")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        action = int(action)
        previous = self.state
        decision_pd = self._predicted_pd
        multiplier = self.config.environment.action_multipliers[action]
        outcome = self._transition.step(previous, multiplier, self._traits,
                                        self._behavior_rng, self._macro_rng)
        breakdown = calculate_reward(previous, outcome, decision_pd, self.config.reward)
        self._state = outcome.state
        self._elapsed += 1
        self._predicted_pd = self._predict_pd(self.state)
        terminated = self.state.defaulted
        # Default takes precedence if it occurs on the last month.
        truncated = self._elapsed >= self.config.environment.horizon and not terminated
        self._done = terminated or truncated
        info = self._info()
        info.update(reward_components=breakdown.to_dict(), decision_pd=decision_pd,
                    payment=outcome.payment, spending=outcome.spending, exposure=outcome.exposure,
                    macro_state_used=int(previous.macro_state))
        row = self._history_row(action=action, reward=breakdown.total,
                                terminated=terminated, truncated=truncated)
        row.update({f"reward_{key}": value for key, value in breakdown.to_dict().items()})
        row.update(payment=outcome.payment, spending=outcome.spending, exposure=outcome.exposure,
                   decision_pd=decision_pd, action_multiplier=multiplier,
                   macro_state_used=int(previous.macro_state))
        self._history.append(row)
        return self._get_observation(), breakdown.total, terminated, truncated, info

    def get_history(self) -> pd.DataFrame:
        """Independent dataframe; row t+1 contains the action/reward from t to t+1."""
        return pd.DataFrame(self._history).copy(deep=True)
