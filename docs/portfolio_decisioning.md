# Risk-constrained sequential portfolio decisioning

## State, chronology and information

An episode is a closed portfolio of N customers sharing one macro path. Each calendar
month starts with public customer observations and frozen current PD forecasts. Active
customers are processed in a reproducible random permutation indexed by portfolio seed
and month. An accepted action updates the reserved portfolio capacity immediately.
Only after all active customers have received one decision do their existing customer
environments advance one month using their indexed exogenous shocks. No within-month
behavior or default of another customer is revealed before allocation is complete.

Default is absorbing. The defaulted loan leaves the next month's live book; its loss
is retained in cumulative accounting. No replacements or recoveries are introduced.
The episode stops on extinction or after 24 months. No debt is erased by reducing a
limit. There is no intercustomer contagion beyond shared macro and shared allocation
capacity; conditional independence is not assumed for inference on a coupled book.

Deployable rules receive copies of public monthly arrays. PPO receives the current
customer's 21-vector plus 13 bounded portfolio features: EL utilization, remaining
capacity, shortfall, EAD utilization, high-risk share, mean PD, active fraction,
allocation progress, mean EAD, allowed EL per initial client, cumulative loss,
dynamic-budget indicator and safety factor. This is a partial aggregate representation;
the batch baselines can inspect all current customers, never future states or hidden
traits. The PPO policy has no information advantage over those baselines.

## Risk stock and economic flows

Let B_i be current drawn balance and L_i the proposed limit. With configurable CCF c:

`undrawn_i = max(L_i - B_i, 0)`

`EAD_i = B_i + c * undrawn_i`.

Credit limit, balance, available credit and EAD are different quantities. The default
CCF 0.5 is a synthetic reservation assumption. It is not a regulatory conversion factor.
The frozen history model forecasts first default within 12 months. A flat-hazard
approximation converts it for this monthly capacity constraint:

`q_i = 1 - (1 - min(s * PD12_i, 1))^(1/12)`

`predicted_EL_i = q_i * LGD * EAD_i`.

Here s is the configurable safety factor (default 1); the buffered 12-month probability
is capped at 1; exact probability endpoints map to 0 and 1. Raw EL uses s=1. LGD is the
existing constant 0.55, checked against the customer environment's realized-loss LGD.
This horizon conversion is not a calibrated monthly model, and the reservation PD is
held fixed while candidate actions change EAD. It is not a causal action-specific PD.

The monthly risk budget is `R_t = N_initial * b * m_t`, where b is allowed EUR of
monthly predicted loss per initial customer. The optional stress rule sets m_t=0.75
in currently observed STRESS and 1 otherwise. Constant budget uses m_t=1. It never
uses future macro. This is a risk stock capacity recomputed each month, not a cash
account depleted forever by past EL. Future PD and balances alter available capacity.
The declared recovery phase retains the STRESS regime with declining severity, so
the 0.75 budget multiplier remains active until recovery ends.

Predicted constraints are:

* `sum(EL_i) <= R_t`;
* `sum(EAD_i) <= N_initial * e`, e=8,000 EUR by default;
* `sum(EAD_i * 1(PD12_i > 0.6)) <= 0.8 * sum(EAD_i)`.

High-risk classification uses the unbuffered, actor-visible PD; the buffer increases
the EL reservation. All budgets are synthetic, fixed in configuration, and distinct.

## Admission and inherited shortfalls

Structural guards always enforce limit bounds, monthly adjustment bounds, no active
decisions after default, and no increase after three delinquent months. For hard
portfolio admission, define linear excess vector

`g = (EL - R_t, EAD - exposure_budget, high_risk_EAD - share_cap * EAD)`.

A structurally projected proposal is accepted only if, componentwise,

`g_after <= max(0, g_before) + 1e-7`.

Otherwise it becomes no change with reason `portfolio_budget`. This means no new
immediate violation and no worsening of an existing violation. It does **not** promise
an absolutely feasible portfolio: inherited drawn balances, a deteriorating forecast
or a tighter macro budget can make the current state infeasible. Remaining capacity
is `max(0, R_t-EL)` and shortfall is separately `max(0, EL-R_t)`. Clamping remaining
capacity does not conceal a breach. Both initial and post-allocation deficits are saved.

The high-risk share is tested in linear form, accounting for denominator changes:
cutting a low-risk line can worsen concentration and may be rejected. A pending
release from a later customer cannot fund an earlier increase. Fixed and reversed
orders are separate diagnostics; primary pairing uses the same randomized order.

## Portfolio objective and PPO

Monthly economic contribution reuses the customer's accounting:

`V_t = sum(interest_t + purchase_fees_t - realized_credit_loss_t - funding_t)`.

Default loss is counted once as LGD times closing balance, not reservation EAD.
Interest is not recognized in a default month. The customer reward's synthetic capital
and individual PD penalties are not counted again in this portfolio objective.
They remain available in the individual-customer experiment. Portfolio penalties are
reported separately:

`penalty_t = lambda * max(EL_after_allocation - R_t, 0)` for soft mode, else 0.

PPO receives zero for intermediate allocation decisions and
`(V_t - penalty_t) / (1,000 * N_initial)` after the synchronized monthly transition.
The objective is the undiscounted finite-horizon sum (gamma=1), avoiding a changing
calendar discount when the number of active customers changes. GAE lambda remains
0.95 and is an optimization parameter, not the economic discount. No reward normalization
is estimated from test data; all scale factors are fixed.

The 24-month horizon is a terminal finite-MDP state, not an external time-limit
truncation: the critic must not bootstrap a continuation value beyond the stated
objective. Extinction is terminal as well.

PPO_unconstrained has structural guards only. PPO_penalty has an EL shortfall penalty
but does not penalize or guarantee the other two portfolio budgets. PPO_hard uses the
same explicit rejection rule as constrained baselines, with no soft penalty. These
are standard SB3 PPO variants, not a theoretical constrained-RL or general safe-RL solution.
Hyperparameters, budgets and seed lists are in `configs/constrained_policy.yaml`.

## Observable baselines

Static requests no change. Decrease20 requests the largest contraction and provides
an essential constant-policy control. RiskBased uses validation-frozen PD thresholds
0.10/0.50 and the shared hard admission rule. BufferedRiskBased uses a safety factor
selected on validation portfolios; the original score still drives its action rule.

The observable one-month value table reuses MyopicEconomic's explicit repayment,
spending and action-sensitive hazard approximations, with its individual capital and
PD penalty coefficients set to zero so the surrogate matches the portfolio economic
objective. It uses nominal coefficients even in shifted DGP worlds. Incremental
economic value is candidate minus unchanged value; incremental risk is candidate EL
reservation minus unchanged reservation.

Greedy sorts positive-gain opportunities by gain/max(incremental_EL,1e-6), choosing
at most one per customer under joint resource ceilings. Risk-releasing opportunities
thus rank before positive risk consumers. The final actions still pass common admission.

Myopic solves a multiple-choice mixed-integer linear problem: one legal candidate per
active customer, maximize the sum of surrogate immediate values, under EL, EAD and
linear high-risk resource ceilings. For an inherited breach, the ceiling is relaxed
only up to the current amount. SciPy/HiGHS reports solve status/gap with a one-second
time limit; absent a valid incumbent, it falls back to Greedy. The **proposal** is exact
only on optimal status. The **executed allocation** can be approximate because the
common sequential guard may reject a proposal before another customer's capacity
release. Reoptimization happens at the next month, not after every subdecision.
This limitation is measured through rejection and order diagnostics, not hidden.

## Simulator-only conditional-risk diagnostic

An independent diagnostic integrates one-step behavior after the effective allocation.
For each active client, 16 hypothetical indexed shocks independent of evaluation shocks
are applied to the true DGP with latent traits and current macro. The default Bernoulli
is integrated analytically conditional on each hypothetical closing state:

`true_EL_t_hat = (1/K) * sum_k sum_i(p_true_closing_i,k * LGD * closing_balance_i,k)`.

This estimates the next-month expected realized credit loss under simulator dynamics,
not a 12-month cumulative PD and not the future realized loss of the evaluated path.
It comes with a Monte Carlo standard error. It never enters observation, action
admission, deployable action values or training reward. Explicit diagnostic export is
separate from public step info. Comparing it to the same EUR budget gives four categories:
both satisfied, predicted-only satisfied, simulator-only satisfied and both violated.
Near-budget classifications remain Monte Carlo uncertain.

This comparison measures the full reservation-model mismatch, including horizon,
EAD and behavior assumptions, not PD calibration alone. It is an oracle **diagnostic**,
not an oracle allocation policy or an attainable optimal-value bound. The independent
customer true-risk oracle remains available in its separately labeled benchmark;
no portfolio perfect-foresight oracle is claimed.

## Experimental design and inference

Train, validation, final, tail and OPE portfolios have separate seed/ID namespaces.
Policies share initial customers, latent traits, macro paths, random decision orders
and indexed shocks within every paired evaluation. Final parameters remain frozen.
Lambda candidates are ranked on validation predicted EL violation, then economic
value; buffer candidates on validation simulator-diagnostic violation, then value.
The latter is privileged **validation evaluation**, never policy input or test tuning.
All final PPO seeds are reported, not just the best.

Primary cases cross three budgets and normal/stress macro. Separate diagnostics vary
actor PD intercept/slope/noise, buffers, constant/dynamic budgets, portfolio size,
decision order and prespecified DGP worlds. A PD distortion changes the common observable
risk service used both by policies and reservation accounting; hidden dynamics remain
unchanged. Buffer sweeps on test are prespecified sensitivities, not model selection.

Monthly state metrics describe the live book after allocation and before transition;
flows describe that month's transition. Monetary totals and default incidence divide
by initial N. EL utilization and violation rates average **observed portfolio-months
before extinction**, not customers or independent monthly samples. Exposure growth
includes live-book attrition from default; it is not just voluntary line changes.
PPO seeds are averaged within each portfolio before distribution summaries. Bootstrap
resamples whole paired portfolios jointly across seeds, and seeds; it never resamples
client-months independently. Size experiments are independent generated populations
at each N, not nested prefixes. Shared macro may generate dependence unconditionally;
this does not imply additional conditional latent correlation or a regulatory decomposition.

Independent tail portfolios report mean, median and SD. Tail quantiles require at least
20 expected tail observations: p90 at n>=200, p95 at n>=400, p99 at n>=2,000. Unsupported
quantiles remain missing. No CVaR constraint or unsupported expected shortfall is claimed.
The PPO seed-averaged distribution measures variation in the average outcome across
training seeds; it is not the loss distribution of one deployed policy. Separate
per-seed distributions retain that distinction.

## Portfolio OPE and reproducibility

Portfolio logs use known mixture propensities over requested commands, public 34-vectors,
remaining budget, shortfall and effective-action reasons. The stochastic behavior and
hard target policies share the same hard transition kernel. The OPE unit is the **whole
coupled portfolio**; importance ratios multiply across every customer's allocation and
month. Customer-level episodes are not independent here. IS/WIS and ESS reuse the
existing math; zero-support realizations remain undefined for WIS. These logs cannot
identify soft/unconstrained target kernels through policy ratios alone, so those targets
are excluded. The original independent-customer OPE remains functional.

Run `python -m experiments.portfolio_constraints --profile smoke --stage all` or choose
train/evaluate/tail/ope/report stages. Standard/full budgets are explicit in configuration.
Three spawned processes run independent standard evaluation jobs, one Torch/BLAS thread
each. Per-job completion files are written last. Registries store source/config/PD
identities; selected model hashes are verified before evaluation. Reporting is separate
from expensive raw simulation and preserves individual seed/portfolio tables.

This is synthetic risk-constrained research: simplified EAD/LGD, no Basel/IRB or IFRS 9
engine, no calibrated regulatory capital, no bank data or deployment validation. Budget
feasibility based on model estimates is not a guarantee on realized losses.
