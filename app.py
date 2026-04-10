from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from stable_baselines3 import PPO

from credit_limit_rl.config import PortfolioConfig
from credit_limit_rl.env import ACTION_ADJUSTMENTS, ACTION_LABELS, build_observation, simulate_credit_decision

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "ppo_credit_limit.zip"
RISK_MODEL_PATH = ARTIFACT_DIR / "risk_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not RISK_MODEL_PATH.exists():
        return None, None
    return PPO.load(str(MODEL_PATH)), joblib.load(RISK_MODEL_PATH)


def infer_region_macro(region: str) -> tuple[float, float, float]:
    mapping = {
        "ile_de_france": (0.071, 0.022, 0.023),
        "nord": (0.089, 0.025, 0.026),
        "sud": (0.083, 0.024, 0.024),
        "est": (0.077, 0.023, 0.024),
        "ouest": (0.068, 0.021, 0.022),
    }
    return mapping[region]


def build_client_frame() -> pd.Series:
    region = st.sidebar.selectbox("Region", ["ile_de_france", "nord", "sud", "est", "ouest"], index=0)
    macro_unemployment, macro_inflation, macro_policy_rate = infer_region_macro(region)

    income = st.sidebar.slider("Monthly income (EUR)", 1200, 10000, 3200, step=100)
    internal_score = st.sidebar.slider("Internal score", 350, 900, 650, step=10)
    current_limit = st.sidebar.slider("Current limit (EUR)", 500, 15000, 3500, step=100)
    utilization = st.sidebar.slider("Utilization", 0.05, 0.99, 0.45, step=0.01)
    monthly_spend = st.sidebar.slider("Monthly spend (EUR)", 50, 8000, 1400, step=50)
    delinquency_30d = st.sidebar.slider("30d delinquency count", 0, 4, 0, step=1)
    late_payments_6m = st.sidebar.slider("Late payments over 6m", 0, 5, 1, step=1)
    tenure_months = st.sidebar.slider("Tenure (months)", 2, 180, 36, step=1)
    age = st.sidebar.slider("Age", 21, 78, 39, step=1)

    current_balance = current_limit * utilization
    debt_to_income = (current_balance + 0.35 * monthly_spend) / income
    true_pd = np.clip(
        1.0
        / (
            1.0
            + np.exp(
                -(
                    -4.35
                    + 3.05 * utilization
                    + 0.48 * delinquency_30d
                    + 0.35 * late_payments_6m
                    - 0.0052 * (internal_score - 650.0)
                    + 14.0 * (macro_unemployment - 0.07)
                    + 8.0 * (macro_inflation - 0.02)
                    + 6.5 * (macro_policy_rate - 0.022)
                    + 0.62 * debt_to_income
                    - 0.003 * min(tenure_months, 120)
                    + 0.015 * max(age - 60, 0)
                )
            )
        ),
        0.002,
        0.45,
    )

    return pd.Series(
        {
            "region": region,
            "age": age,
            "tenure_months": tenure_months,
            "income": float(income),
            "internal_score": float(internal_score),
            "delinquency_30d": float(delinquency_30d),
            "late_payments_6m": float(late_payments_6m),
            "macro_unemployment": float(macro_unemployment),
            "macro_inflation": float(macro_inflation),
            "macro_policy_rate": float(macro_policy_rate),
            "current_limit": float(current_limit),
            "current_balance": float(current_balance),
            "monthly_spend": float(monthly_spend),
            "utilization": float(utilization),
            "debt_to_income": float(debt_to_income),
            "true_pd": float(true_pd),
        }
    )


def main() -> None:
    st.set_page_config(page_title="Dynamic Credit Limit RL", layout="wide")
    st.title("Dynamic Credit Limit Optimization with Reinforcement Learning")
    st.caption("Retail banking demo: RL policy versus static credit limit management.")

    model, risk_model = load_artifacts()
    if model is None or risk_model is None:
        st.warning("Train the project first with `python train.py` so the dashboard can load the RL artifacts.")
        return

    client = build_client_frame()
    action_index, _ = model.predict(build_observation(client), deterministic=True)
    action_index = int(action_index)
    recommended_adjustment = float(ACTION_ADJUSTMENTS[action_index])

    config = PortfolioConfig()
    policy_result = simulate_credit_decision(client, recommended_adjustment, risk_model, config, np.random.default_rng(7))
    maintain_result = simulate_credit_decision(client, 0.0, risk_model, config, np.random.default_rng(7))

    col1, col2, col3 = st.columns(3)
    col1.metric("Recommended action", ACTION_LABELS[action_index])
    col2.metric("New limit", f"EUR {policy_result['new_limit']:.0f}", delta=f"{recommended_adjustment:+.0%}")
    col3.metric("Predicted PD", f"{policy_result['predicted_pd']:.2%}", delta=f"{policy_result['predicted_pd'] - maintain_result['predicted_pd']:+.2%}")

    comparison = pd.DataFrame(
        [
            {
                "policy": "Maintain",
                "reward": maintain_result["reward"],
                "expected_loss": maintain_result["expected_loss"],
                "rwa_cost": maintain_result["rwa_cost"],
                "new_limit": maintain_result["new_limit"],
            },
            {
                "policy": "RL recommendation",
                "reward": policy_result["reward"],
                "expected_loss": policy_result["expected_loss"],
                "rwa_cost": policy_result["rwa_cost"],
                "new_limit": policy_result["new_limit"],
            },
        ]
    )

    left, right = st.columns([1.2, 1.0])
    with left:
        chart = px.bar(
            comparison,
            x="policy",
            y=["reward", "expected_loss", "rwa_cost"],
            barmode="group",
            title="One-step financial impact",
        )
        st.plotly_chart(chart, use_container_width=True)
    with right:
        st.subheader("Client snapshot")
        st.dataframe(client.to_frame("value"), use_container_width=True)

    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.subheader("Offline backtest metrics")
        st.dataframe(pd.DataFrame(metrics["policy_comparison"]), use_container_width=True)


if __name__ == "__main__":
    main()
