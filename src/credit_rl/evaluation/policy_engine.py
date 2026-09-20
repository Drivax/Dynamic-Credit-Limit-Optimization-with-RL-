"""One controlled evaluator for every policy; raw economic units throughout."""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
import pandas as pd

from credit_rl import CreditLimitEnv
from credit_rl.policies.decision import transform_observation


@dataclass
class PolicySpec:
    name: str
    factory: object
    seed: int = -1
    information_set: str = "OBSERVABLE_ONLY"
    without_pd: bool = False
    pd_multiplier: float = 1.
    model_identifier: str = "rule"
    observation_transform_factory: object = None


def evaluate_policy(spec, scenarios, config, pd_model, settings, scenario_name, *, keep_history=True, transition_observer=None):
    started = perf_counter()
    episodes, histories = [], []
    inference_seconds = 0.
    for scenario in scenarios:
        env = CreditLimitEnv(pd_model=pd_model, config=config, record_history=True,
                             severe_delinquency_months=settings["guardrails"]["severe_delinquency_months"])
        policy = spec.factory(env, scenario)
        sensor = spec.observation_transform_factory(scenario) if spec.observation_transform_factory else None
        observation, _ = env.reset(seed=scenario.customer_seed, options=scenario.reset_options())
        actor_pd = [np.nan]
        while True:
            supplied = sensor(observation) if sensor else observation
            actor_observation = transform_observation(supplied, spec.without_pd, spec.pd_multiplier)
            tick = perf_counter()
            action = policy.act(actor_observation)
            inference_seconds += perf_counter()-tick
            if not env.action_space.contains(action):
                raise ValueError(f"Invalid action from {spec.name}: {action}")
            actor_pd.append(float(actor_observation[10]))
            observation, reward, term, trunc, info = env.step(action)
            if transition_observer is not None:
                components = info["reward_components"]
                transition_observer(dict(customer_id=scenario.customer_id, month=len(actor_pd)-2,
                    observation=actor_observation.copy(), action=int(action), reward=float(reward),
                    economic_value=float(components["interest_income"]+components["fee_income"]
                        -components["credit_loss"]-components["funding_cost"]),
                    next_observation=observation.copy(), terminated=bool(term), truncated=bool(trunc)))
            if not np.isfinite(reward) or not np.isfinite(observation).all():
                raise ValueError("Nonfinite environment result")
            if term or trunc:
                break
        frame = env.get_history()
        frame["actor_pd"] = actor_pd
        frame["opening_utilization"] = frame.utilization.shift(1)
        frame["opening_delinquency"] = frame.months_delinquent.shift(1)
        frame["opening_macro"] = frame.macro_state.shift(1)
        frame["scheduled_end"] = config.environment.horizon
        frame["cohort_start"] = 0
        tags = dict(policy=spec.name, policy_seed=spec.seed, scenario=scenario_name,
                    partition=scenario.partition, information_set=spec.information_set,
                    model_identifier=spec.model_identifier)
        frame = frame.assign(**tags)
        episodes.append({**episode_metrics(frame, settings), **tags})
        if keep_history:
            histories.append(frame)
        env.close()
    elapsed = perf_counter()-started
    episode_frame = pd.DataFrame(episodes)
    return episode_frame, pd.concat(histories, ignore_index=True) if histories else pd.DataFrame(), {
        "seconds": elapsed, "steps": int(episode_frame.steps.sum()),
        "steps_per_second": float(episode_frame.steps.sum()/elapsed),
        "policy_inference_ms": 1000*inference_seconds/episode_frame.steps.sum()}


def episode_metrics(history, settings):
    initial = history.iloc[0]
    s = history.iloc[1:]
    revenue = s.reward_interest_income.sum()+s.reward_fee_income.sum()
    effective = s.effective_action.to_numpy()
    changed = effective[np.abs(effective) > 1e-9]
    reversals = np.sum(np.sign(changed[1:]) != np.sign(changed[:-1]))
    return dict(customer_id=str(initial.customer_id), steps=len(s),
        cumulative_reward=float(s.reward.sum()),
        discounted_reward=float(np.dot(s.reward, settings["ppo"]["gamma"]**np.arange(len(s)))),
        revenue=float(revenue), interest_income=float(s.reward_interest_income.sum()),
        fee_income=float(s.reward_fee_income.sum()), credit_loss=float(s.reward_credit_loss.sum()),
        funding_cost=float(s.reward_funding_cost.sum()), capital_charge=float(s.reward_capital_cost.sum()),
        constraint_penalty=float(s.reward_constraint_penalty.sum()),
        net_economic_value=float(revenue-s.reward_credit_loss.sum()-s.reward_funding_cost.sum()),
        defaulted=int(s.defaulted.any()), ever_delinquent=int((s.months_delinquent > 0).any()),
        delinquent_months=int((s.months_delinquent > 0).sum()),
        initial_balance=float(initial.balance), initial_limit=float(initial.credit_limit),
        final_limit=float(s.credit_limit.iloc[-1]), final_balance=float(s.balance.iloc[-1]),
        exposure_sum=float(s.balance.sum()), mean_balance=float(s.balance.mean()),
        mean_limit=float(s.credit_limit.mean()), mean_utilization=float(s.utilization.mean()),
        predicted_pd_sum=float(s.decision_pd.sum()),
        high_risk_exposure=float(s.loc[s.decision_pd >= settings["risk_constraints"]["high_pd"], "balance"].sum()),
        exposure_at_default=float(s.loc[s.defaulted, "exposure"].sum()),
        increases=int((effective > 1e-9).sum()), decreases=int((effective < -1e-9).sum()),
        no_change=int((abs(effective) <= 1e-9).sum()), limit_changes=len(changed), reversals=int(reversals),
        requested_change_sum=float(s.requested_action.sum()), effective_change_sum=float(effective.sum()),
        absolute_change_sum=float(abs(effective).sum()), guardrail_blocks=int(s.guardrail_blocked.sum()),
        delinquent_increases=int(((s.opening_delinquency > 0) & (s.effective_action > 1e-9)).sum()),
        severe_delinquent_increases=int(((s.opening_delinquency >= settings["guardrails"]["severe_delinquency_months"]) & (s.effective_action > 1e-9)).sum()))


def aggregate_episodes(frame):
    count, months = len(frame), frame.steps.sum()
    return dict(episodes=count, customer_months=int(months),
        **{name: float(frame[name].mean()) for name in ("cumulative_reward", "discounted_reward", "net_economic_value",
            "revenue", "credit_loss", "funding_cost", "capital_charge", "constraint_penalty", "mean_balance",
            "mean_limit", "mean_utilization", "limit_changes", "reversals")},
        default_rate=float(frame.defaulted.mean()), ever_delinquent_rate=float(frame.ever_delinquent.mean()),
        delinquent_month_rate=float(frame.delinquent_months.sum()/months),
        credit_loss_rate=float(frame.credit_loss.sum()/max(frame.initial_balance.sum(), 1)),
        mean_exposure_at_default=float(frame.exposure_at_default.sum()/max(frame.defaulted.sum(), 1)),
        mean_predicted_pd=float(frame.predicted_pd_sum.sum()/months),
        high_risk_exposure_share=float(frame.high_risk_exposure.sum()/max(frame.exposure_sum.sum(), 1)),
        increase_fraction=float(frame.increases.sum()/months), decrease_fraction=float(frame.decreases.sum()/months),
        no_change_fraction=float(frame.no_change.sum()/months),
        requested_change=float(frame.requested_change_sum.sum()/months), effective_change=float(frame.effective_change_sum.sum()/months),
        mean_absolute_change=float(frame.absolute_change_sum.sum()/months),
        delinquent_increase_fraction=float(frame.delinquent_increases.sum()/months),
        severe_delinquent_increases=int(frame.severe_delinquent_increases.sum()),
        blocked_fraction=float(frame.guardrail_blocks.sum()/months),
        mean_limit_growth=float((frame.final_limit/frame.initial_limit-1).mean()))
