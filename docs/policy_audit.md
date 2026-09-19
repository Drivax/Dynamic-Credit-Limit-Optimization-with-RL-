# Audit of the policy experiment surface

The active source, configuration, tests, experiment entry points, documentation and
output inventory were inspected before the policy work. Existing uncommitted PD
documentation/results were retained. Archived workflows are outside the active package.

1. **Policies:** static, constant-action, a simple PD threshold rule and an SB3 adapter
   existed. There was no observable economic optimizer or unified paired evaluation.
2. **PPO:** the script used Stable-Baselines3, a single seed, short rollouts and a
   synthetic train/test population generated with different seeds. It was explicitly
   an integration smoke test, not a tuned/multi-seed benchmark.
3. **Information:** PPO receives the 21 public observation coordinates. Hidden traits,
   closing hazard and future shocks/macro states are absent. Continuous quantities
   already use explicit bounded transforms; no empirical normalizer is needed.
4. **PD:** `CreditLimitEnv` really calls the saved longitudinal model using seven
   observed snapshots. The model/preprocessor/calibrator are bundled. The DGP does
   not receive that estimator. PD forecast horizon is 12 months by default.
5. **Reward:** interest and fee revenue minus realized credit loss, funding, a
   stylized PD-dependent capital charge and a threshold penalty. Every term is an
   EUR proxy per transition. Capital and penalties are not regulatory calculations.
6. **Actions:** the five limit multipliers are 0.8/0.9/1/1.1/1.2, constrained to
   EUR 500–15,000 and monthly ±20%. Requested and effective changes already existed.
   A delinquency guardrail was not present; the benchmark now explicitly configures it.
7. **Seeds:** pure DGP transitions consume supplied named shocks. `ShockPath` indexes
   customer identity, month and channel with stable hashes and independent generators.
8. **Pairing:** explicit initial state/traits, immutable macro paths and shock paths
   already support exactly matched exogenous inputs. Policies need no simulation RNG.
9. **Populations:** the existing smoke split was independent but had no durable
   RL_TRAIN/RL_VALIDATION/RL_TEST identity manifest or checkpoint selection protocol.
10. **Leakage:** no direct latent feature leak was found in the current environment.
    Privileged diagnostic exports and archived snapshot targets must stay separate.
    The benchmark introduces explicit split checks, validation-only selection and
    a privileged oracle type rather than giving latent data to deployable policies.

## Reward audit decision

The unchanged reward was measured over 12,822 training-cohort transitions under
Static, Random and MyopicEconomic, before the multi-seed experiment. The complete
mean/std/min/p01/p05/median/p95/p99/max tables are in `reward_audit.csv`.
Credit losses are rare within monthly rows and large when realized; the stylized
capital term is systematically larger than revenues. This is an economically
important feature of the specified objective, not a numerical NaN or a duplicated
loss. Its realism is not established by this audit.

PPO receives reward in kEUR (a fixed factor 0.001), while evaluation records the raw
EUR reward. This changes numerical units, not ordering or the discounted optimum.
There is no adaptive reward normalization or observation-statistics fit. No revenue,
loss, capital, PD, default, behavior or macro coefficient was changed to improve PPO.

The smoke training ran 1,024 transitions with finite observations/rewards, legal
actions and working resets. Its fixed validation objective improved, which justified
running the declared larger experiment; this is not evidence of general superiority.
