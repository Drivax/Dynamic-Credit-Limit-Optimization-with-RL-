# Action Collapse - Root Cause Analysis & Solution Roadmap

## Diagnosis Summary

The PPO policy has **completely collapsed to always selecting +20% (expand limit by 20%)**. This is NOT due to training failures—it's **correct behavior given the reward structure**.

### Evidence
- **Test 1**: Trained with entropy coef 0.10 (5x baseline) → 100% +20%
- **Test 2**: Trained with lambda_default 12.0 (2.4x stronger penalties) → 100% +20%  
- **Test 3**: Trained with lambda_default 15.0 + lambda_rwa 5.0 (3x stronger) → 100% +20%

### Root Cause
The reward function mathematically incentivizes expansion for **all clients, including high-risk ones**:

```
HIGH-RISK clients:
  -20%: -6305 | -10%: -5973 | 0%: -5308 | +10%: -4893 | +20%: -4527 ← BEST
  
LOW-RISK clients:  
  -20%: -5948 | -10%: -5197 | 0%: -4591 | +10%: -3755 | +20%: -3464 ← BEST
```

Even with triple the risk penalties, interest/fee income from higher utilization overwhelms default costs.

**The policy is correctly optimized—the problem is the optimization target.**

---

## Why This Matters

The current setup violates core banking principles:
1. **Risk-unaware**: Expansion increases default probability but is always rewarded
2. **Concentration**: All clients get same treatment regardless of risk
3. **Non-conservative**: No penalty for accumulating high-risk exposure

---

## Solution Roadmap

### ✅ Quick Fixes (Test First)

#### Option A: Hard Risk Constraints
Modify `env.py` to enforce per-client PD caps:

```python
# In simulate_credit_decision():
if predicted_pd > threshold_by_score(internal_score):
    constraint_penalty = LARGE_PENALTY  # e.g., 5000
```

**Expected outcome**: Force -20% or 0% for high-risk clients, +10% for low-risk.

**Implementation**: Add to `env.py` line ~85
```python
def get_pd_threshold(internal_score):
    if internal_score < 600:
        return 0.05  # Strict for risky
    elif internal_score < 700:
        return 0.08
    else:
        return 0.15  # Loose for safe
```

#### Option B: Action Diversity Bonus
Add penalty for always using the same action:

```python
# In env.py, track action counts per episode
if most_recent_100_actions.nunique() < 3:
    reward -= 100  # Penalize lack of diversity
```

**Expected outcome**: Forces mix of actions, even if suboptimal for some clients.

### ⚠️ Medium-Effort Fixes (Better Results)

#### Option C: Risk-Tiered Policies
Train separate models for risk segments:

```bash
# High-risk tier (internal_score < 600)
python train.py --clients 50000 --risk-segment high \
  --lambda-default 20.0 --lambda-rwa 8.0

# Low-risk tier (internal_score > 750)  
python train.py --clients 50000 --risk-segment low \
  --lambda-default 3.0 --lambda-rwa 0.5
```

**Expected outcome**: Specialized policies for each segment, respect local optima.

**Implementation**: Modify `train.py` to accept `--risk-segment` and filter portfolio before training.

#### Option D: Constraint-Based Formulation
Reframe as a **constrained optimization problem**:

```python
# Maximize: interest + fees
# Subject to:
#   - Portfolio PD <= 10%
#   - Per-client PD <= risk_appetite(score)
#   - Limit change in [-20%, +20%]
```

Use Lagrange multipliers or interior-point methods instead of RL.

**Implementation**: Create `train_constrained.py` using `scipy.optimize`.

### 🚀 Best Solution (Recommended)

#### Option E: Composite Reward with Fairness
Restructure reward to embed risk-aware constraints:

```python
reward = interest + fees - lambda_default * EL - lambda_rwa * RWA

# Add risk-interaction term: penalize expansion + high PD
if action_adjustment > 0 and predicted_pd > median_pd:
    risk_penalty += 500 * action_adjustment * (predicted_pd - median_pd)

# Add portfolio stability bonus
if portfolio_pd_std < target_std:
    reward += 50  # Bonus for conservative decisions

reward -= large_penalty if per_client_pd > threshold
```

**Expected outcome**: 
- Natural incentive to shrink high-risk clients
- Natural incentive to expand low-risk clients  
- Portfolio-level stability
- No collapse

**Implementation**:
1. Create `credit_limit_rl/config_v2.py` with new reward structure
2. Modify `env.py` to support both configs
3. Add `--reward-version 2` flag to `train.py`
4. Train and compare

---

## Implementation Checklist

### Phase 1: Validation (Complete Today)
- [x] Diagnose root cause: reward function structure
- [x] Test: High entropy doesn't help
- [x] Test: Stronger penalties don't help
- [ ] Test: Option A (hard PD constraints)
- [ ] Test: Option B (action diversity bonus)

### Phase 2: Core Fix (This Week)
- [ ] Implement Option E (composite reward)
- [ ] Add `--reward-version` flag to CLI
- [ ] Train with new reward structure
- [ ] Evaluate action distribution (expect diversity)
- [ ] Compare risk outcomes vs. original

### Phase 3: Production (Next Week)
- [ ] Validate on full portfolio (50K clients, 200K timesteps)
- [ ] Run k-fold cross-validation with Option E
- [ ] Create documentation of new reward structure
- [ ] Update README with new methodology

---

## Expected Improvements (Post-Fix)

| Metric | Current | Expected |
|--------|---------|----------|
| Action distribution | 100% +20% | 20-30% mixed |
| High-risk avg action | +20% (expand) | -10% to 0% (shrink/hold) |
| Low-risk avg action | +20% (expand) | +10% to +20% |
| Portfolio default rate | ~11% | ~8-9% |
| Risk stratification | 0 (all same) | High (risk-aware) |

---

## Recommended Next Step

**Start with Option A (Hard Constraints) this week:**

1. Add `get_pd_threshold()` function to `env.py`
2. Modify reward computation to add large penalty if PD exceeds threshold
3. Retrain with same parameters
4. Expect: Action distribution becomes 30-40% diverse

If that works, proceed to Option E for better balance.

---

## Code Locations to Modify

- `credit_limit_rl/env.py` (lines 60-100): `simulate_credit_decision()` reward calculation
- `credit_limit_rl/config.py`: Add new thresholds if using Option A
- `train.py`: Add CLI flags for reward weighting if using Option E

---

## Questions to Ask Before Implementation

1. **Business priority**: Maximize profit or minimize risk? (Affects lambda_default/lambda_rwa ratio)
2. **Risk appetite by segment**: What PD threshold for high-risk vs low-risk clients?
3. **Action diversity**: Is it OK to sometimes make "suboptimal" decisions for stabilization?
4. **Deployment**: Can we use multiple policies (Option C) or need one unified policy?

