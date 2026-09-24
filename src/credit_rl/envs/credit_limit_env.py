"""One customer per episode, until default or a configurable monthly horizon."""

from dataclasses import asdict, replace
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from credit_rl.config import SimulationConfig
from credit_rl.reward import calculate_reward
from credit_rl.risk.pd_model import ObservedLogisticPD, ObservedRiskFeatures, PDModel, HistoryPDModel
from credit_rl.risk.features import observable_row
from credit_rl.simulation.customer import CustomerState, CustomerTraits, initialize_customer
from credit_rl.simulation.dgp import CreditDGP
from credit_rl.simulation.macro import MacroPath, MacroProcess
from credit_rl.simulation.shocks import ShockPath
from credit_rl.utils.seeding import episode_generators
from .observation import OBSERVATION_NAMES, build_observation
from .constraints import effective_limit


class CreditLimitEnv(gym.Env[np.ndarray, int]):
    """Partially observed longitudinal research environment.

    Observations/info/history deliberately exclude latent traits and simulator
    probabilities. `options` accepts a customer_index or an explicit initial
    CustomerState + CustomerTraits pair for controlled simulation experiments.
    Snapshot rows are read only at reset; step never accesses the portfolio.
    """
    metadata = {"render_modes": []}

    def __init__(self, portfolio: pd.DataFrame | None = None, pd_model: PDModel | HistoryPDModel | None = None,
                 config: SimulationConfig | None = None, *, record_diagnostics: bool = False,
                 record_history: bool = True, severe_delinquency_months: int | None = None) -> None:
        super().__init__()
        self.config = config or SimulationConfig()
        if severe_delinquency_months is not None and (type(severe_delinquency_months) is not int or severe_delinquency_months < 1):
            raise ValueError("severe_delinquency_months must be a positive integer or None")
        self.severe_delinquency_months = severe_delinquency_months
        if portfolio is not None and portfolio.empty:
            raise ValueError("portfolio must not be empty")
        # Only initialization fields survive ingestion, even if targets are supplied.
        required = ["income", "tenure_months", "internal_score", "current_limit",
                    "current_balance", "monthly_spend"]
        optional = ["customer_id", "delinquency_30d", "late_payments_6m"]
        self._portfolio = None if portfolio is None else portfolio[
            required + [name for name in optional if name in portfolio]].reset_index(drop=True).copy()
        self.pd_model = pd_model if pd_model is not None else ObservedLogisticPD(self.config.pd)
        self._dgp = CreditDGP(self.config)
        self._record_diagnostics = record_diagnostics
        self._record_history = record_history
        self._diagnostics: list[dict[str, Any]] = []
        self.action_space = spaces.Discrete(len(self.config.environment.action_multipliers))
        self.observation_space = spaces.Box(0.0, 1.0, (len(OBSERVATION_NAMES),), dtype=np.float32)
        self.static_action_index = self.config.environment.action_multipliers.index(1.0)
        self._state: CustomerState | None = None
        self._traits: CustomerTraits | None = None
        self._history: list[dict[str, Any]] = []
        self._done = True
        self._elapsed = 0
        self._predicted_pd = 0.0
        self._risk_history = deque(maxlen=7)

    @property
    def state(self) -> CustomerState:
        """Immutable current observed state, safe to retain for before/after checks."""
        if self._state is None:
            raise RuntimeError("Call reset first")
        return self._state

    def _predict_pd(self, state: CustomerState) -> float:
        self._risk_history.append(observable_row(state))
        if state.defaulted and hasattr(self.pd_model, "predict_history"):
            return 1.0  # terminal sentinel, never used for an active decision
        if hasattr(self.pd_model, "predict_history"):
            value = float(self.pd_model.predict_history(tuple(self._risk_history)))
        else:
            value = float(self.pd_model.predict(ObservedRiskFeatures.from_state(state)))
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("PD model must return a finite probability in [0, 1]")
        return value

    def _get_observation(self) -> np.ndarray:
        """Observable quantities only; x/(1+x) bounds ratios without clipping debt."""
        return build_observation(self.state, self._predicted_pd, self._elapsed, self.config)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        options = options or {}
        allowed = {"customer_index", "initial_state", "traits", "macro_path", "shock_path"}
        if set(options) - allowed:
            raise ValueError(f"Unknown reset options: {set(options) - allowed}")
        init_rng, path_rng, macro_rng = episode_generators(self.np_random)
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
        if not d.income_min <= state.income <= d.income_max:
            raise ValueError("Initial income outside configured bounds")
        macro_path = options.get("macro_path")
        if macro_path is None:
            macro_path = MacroProcess(self.config.macro).generate(e.horizon, macro_rng, state.macro_state)
        if not isinstance(macro_path, MacroPath) or macro_path.horizon != e.horizon:
            raise ValueError("Macro path length must equal horizon + 1")
        shock_path = options.get("shock_path")
        if shock_path is None:
            shock_path = ShockPath.generate(state.customer_id, int(path_rng.integers(0, 2**32)), e.horizon)
        if not isinstance(shock_path, ShockPath) or len(shock_path.months) != e.horizon:
            raise ValueError("Shock path length must equal horizon")
        if shock_path.customer_id != state.customer_id:
            raise ValueError("Shock path customer identity mismatch")
        self._macro_path, self._shock_path = macro_path, shock_path
        state = replace(state, macro_state=macro_path.states[0])
        self._state, self._traits = state, traits
        self._elapsed, self._done = 0, False
        self._risk_history.clear()
        self._predicted_pd = self._predict_pd(state)
        self._history = [self._history_row(action=None, reward=0.0, terminated=False, truncated=False)] if self._record_history else []
        self._diagnostics = []
        return self._get_observation(), self._info()

    def _info(self) -> dict[str, Any]:
        return {"customer_id": self.state.customer_id, "month": self.state.month,
                "predicted_pd": self._predicted_pd, "defaulted": self.state.defaulted,
                "credit_limit": self.state.credit_limit, "macro_state": int(self.state.macro_state.regime)}

    def _history_row(self, *, action: int | None, reward: float, terminated: bool,
                     truncated: bool) -> dict[str, Any]:
        row = asdict(self.state)
        # Retain observable history counts, not initialization bookkeeping.
        row.pop("initial_score")
        row.pop("initial_income")
        row.pop("macro_state")
        row["late_payments_6m"] = sum(row.pop("late_history"))
        row.update(utilization=self.state.utilization, delinquency_status=self.state.delinquency_status,
                   predicted_pd=self._predicted_pd, action=action, reward=reward,
                   terminated=terminated, truncated=truncated, macro_state=int(self.state.macro_state.regime),
                   macro_income_growth=self.state.macro_state.income_growth,
                   macro_spending_growth=self.state.macro_state.spending_growth,
                   macro_credit_stress=self.state.macro_state.credit_stress,
                   delinquency_bucket=self.state.delinquency_bucket)
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
        admitted_limit, guardrail_blocked = effective_limit(previous.credit_limit, previous.months_delinquent,
            multiplier, self.config.environment, self.severe_delinquency_months)
        shocks = self._shock_path.months[self._elapsed]
        outcome = self._dgp.step(previous, admitted_limit/previous.credit_limit, self._traits,
                                 shocks, self._macro_path.states[self._elapsed+1])
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
                    macro_state_used=int(previous.macro_state.regime), requested_action=multiplier-1,
                    effective_action=outcome.effective_multiplier-1)
        info["guardrail_blocked"] = guardrail_blocked
        if self._record_history or self._record_diagnostics:
            row = self._history_row(action=action, reward=breakdown.total,
                                    terminated=terminated, truncated=truncated)
            row.update({f"reward_{key}": value for key, value in breakdown.to_dict().items()})
            row.update(payment=outcome.payment, spending=outcome.spending, exposure=outcome.exposure,
                       decision_pd=decision_pd, action_multiplier=multiplier,
                       macro_state_used=int(previous.macro_state.regime),
                       requested_action=multiplier-1, effective_action=outcome.effective_multiplier-1,
                       credit_limit_before_action=previous.credit_limit)
            row["guardrail_blocked"] = guardrail_blocked
            if self._record_history:
                self._history.append(row)
            if self._record_diagnostics:
                self._diagnostics.append({**row, "p_default_true": outcome.p_default_true,
                    "realized_default": self.state.defaulted, "adverse_income_event": outcome.adverse_income_event,
                    "spending_elasticity": outcome.spending_elasticity,
                    "macro_credit_stress_used": previous.macro_state.credit_stress,
                    "macro_income_growth_used": previous.macro_state.income_growth,
                    "macro_spending_growth_used": previous.macro_state.spending_growth,
                    **{f"shock_{name}": value for name, value in asdict(shocks).items()}})
        return self._get_observation(), breakdown.total, terminated, truncated, info

    def get_history(self) -> pd.DataFrame:
        """Independent dataframe; row t+1 contains the action/reward from t to t+1."""
        return pd.DataFrame(self._history).copy(deep=True)

    def get_diagnostics(self) -> pd.DataFrame:
        """Explicit research-only export. Never supplied via observations or info.

        Includes closing conditional hazard and shocks; do NOT use as ML features.
        Latent traits are deliberately kept in a separate export.
        """
        if not self._record_diagnostics:
            raise RuntimeError("Enable record_diagnostics explicitly")
        return pd.DataFrame(self._diagnostics).copy(deep=True)

    def get_latent_diagnostics(self) -> dict[str, Any]:
        if not self._record_diagnostics or self._traits is None:
            raise RuntimeError("Enable diagnostics and reset first")
        return {"customer_id": self.state.customer_id, **asdict(self._traits)}
