"""Aggregate trajectories without confusing monthly default hazard and incidence."""

import pandas as pd


def summarize_trajectories(history: pd.DataFrame) -> pd.DataFrame:
    """Report mean lifetime return and cumulative default incidence per policy.

    No extrapolation beyond truncation and no post-default padding. Requires
    unique episode_id per policy, which also supports repeated customer runs.
    """
    steps = history.loc[history.action.notna()]
    episodes = steps.groupby(["policy", "episode_id"]).agg(
        episode_return=("reward", "sum"), defaulted=("defaulted", "max"),
        months=("month", "size"), average_limit=("credit_limit", "mean"),
    )
    return episodes.groupby("policy").agg(
        episodes=("episode_return", "size"), mean_return=("episode_return", "mean"),
        return_std=("episode_return", "std"), default_incidence=("defaulted", "mean"),
        mean_months=("months", "mean"), mean_limit=("average_limit", "mean"),
    ).reset_index()
