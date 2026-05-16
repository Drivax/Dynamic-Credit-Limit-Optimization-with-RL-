"""
Quick fix: Implement hard per-client PD constraints to prevent action collapse.

This adds per-client risk-based limits that force conservative behavior on high-risk clients.
Expected: Action distribution becomes diverse (currently 100% +20%).
"""

def get_pd_threshold_by_score(internal_score: float) -> float:
    """
    Return maximum acceptable PD based on internal risk score.
    Higher score = lower risk = higher tolerance.
    
    Args:
        internal_score: Client's internal risk score (0-1000)
    
    Returns:
        Maximum acceptable predicted PD (0-1)
    """
    if internal_score < 500:
        return 0.04  # Very strict: high-risk clients
    elif internal_score < 600:
        return 0.06  # Strict
    elif internal_score < 700:
        return 0.10  # Moderate
    elif internal_score < 800:
        return 0.15  # Loose
    else:
        return 0.25  # Very loose: low-risk clients


# In credit_limit_rl/env.py, modify simulate_credit_decision() around line 88:
# Replace this section:
#
#   constraint_penalty = 0.0
#   if predicted_pd > config.max_pd_threshold:
#       constraint_penalty += 25.0 * (predicted_pd - config.max_pd_threshold)
#
# With this:

def simulate_credit_decision_with_constraints(
    record: pd.Series | Mapping[str, float],
    adjustment: float,
    risk_model,
    config: PortfolioConfig,
    rng: np.random.Generator | None = None,
) -> dict[str, float | int]:
    """Modified version with hard per-client PD constraints."""
    client = _to_series(record)
    generator = rng or np.random.default_rng()

    current_limit = float(client["current_limit"])
    current_balance = float(client["current_balance"])
    proposed_limit = float(np.clip(current_limit * (1.0 + adjustment), config.min_limit, config.max_limit))
    spend_multiplier = float(np.clip(1.0 + 0.65 * adjustment, 0.75, 1.20))
    adjusted_spend = float(max(20.0, client["monthly_spend"] * spend_multiplier))

    risk_frame = _risk_feature_frame(client, proposed_limit, adjusted_spend)
    predicted_pd = float(risk_model.predict_proba(risk_frame)[0, 1])

    # ===== NEW: HARD PD CONSTRAINT =====
    pd_threshold = get_pd_threshold_by_score(client["internal_score"])
    constraint_penalty = 0.0
    
    if predicted_pd > pd_threshold:
        # Large penalty for violating per-client risk limit
        # This makes the action unattractive
        excess_pd = predicted_pd - pd_threshold
        constraint_penalty = 1000.0 * excess_pd  # Very strong penalty
    # ===================================

    # Rest of the function remains the same...
    utilization_delta = float(risk_frame.iloc[0]["utilization"] - client["utilization"])
    realized_pd = float(np.clip(client["true_pd"] * (1.0 + 1.2 * utilization_delta), 0.002, 0.65))
    defaulted = int(generator.random() < realized_pd)

    monthly_interest = current_balance * config.annual_percentage_rate / 12.0
    fee_income = adjusted_spend * config.fee_rate
    expected_loss = predicted_pd * config.loss_given_default * current_balance
    rwa_cost = config.rwa_factor * predicted_pd * current_balance

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
        "constraint_penalty": constraint_penalty,  # Now includes hard PD constraint
        "reward": reward,
    }


# ============= INTEGRATION STEPS =============
#
# 1. Add get_pd_threshold_by_score() to credit_limit_rl/env.py (top of file)
#
# 2. Modify simulate_credit_decision() in credit_limit_rl/env.py:
#    - Keep the line: constraint_penalty = 0.0
#    - Replace the if block with:
#
#      pd_threshold = get_pd_threshold_by_score(client["internal_score"])
#      if predicted_pd > pd_threshold:
#          excess_pd = predicted_pd - pd_threshold
#          constraint_penalty = 1000.0 * excess_pd
#
# 3. Run test:
#    python retrain_fixed.py --clients 5000 --train-timesteps 50000 \
#      --ent-coef 0.10 --output-suffix "with_constraints"
#
# 4. Check action distribution in output:
#    Expected: Mix of actions (-20%, -10%, 0%, +10%, +20%)
#    vs Current: 100% +20%
#
# ============================================

if __name__ == "__main__":
    # Quick test: Show threshold by score
    print("PD Threshold Schedule by Internal Score:")
    print("-" * 50)
    for score in [400, 500, 600, 700, 800, 900]:
        threshold = get_pd_threshold_by_score(score)
        print(f"  Score {score}: Max PD = {threshold:.2%}")
