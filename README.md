# Dynamic Credit Limit Optimization with Reinforcement Learning

**A simulation lab for studying how to adjust credit limits over time, taking customer behavior, default risk, and economic conditions into account.**

Increasing a limit can enable more spending and generate additional revenue. It can also increase the amount lost if the customer stops repaying. Reducing a limit restricts available credit, but may leave an indebted customer above their new limit.

This project explores these trade-offs over several months, compares decision rules, and supports reinforcement learning experiments in a controlled, reproducible environment.

## The central question

> What credit limit should we offer today, knowing that this decision may change customer behavior and outcomes in the months ahead?

A decision cannot be evaluated solely by its immediate revenue. It changes available credit, potential spending, repayment, and risk exposure. These effects accumulate over time.

For example, a customer with EUR 2,500 in debt and a EUR 4,000 limit has a utilization rate of 62.5%. Raising the limit to EUR 4,800 immediately lowers that ratio to about 52.1%, without reducing the debt. The customer also has more credit available for future purchases. A lower ratio alone therefore does not establish that the decision will be profitable.

## How does the simulation work?

Each episode follows **the same customer**, month after month, for 24 months by default. At each step, a policy — a decision rule — chooses whether to decrease, maintain, or increase the customer's limit.

```text
     Customer state and economic conditions
                       ↓
                Risk estimation
                       ↓
             Credit limit decision
                       ↓
      Income, repayment, and new spending
                       ↓
    Balance, missed payments, and possible default
                       ↓
       Economic outcome and the next month
```

The default actions are −20%, −10%, no change, +10%, and +20%. They apply to the current limit, subject to configurable bounds. Default ends the episode.

The balance follows a simple accounting rule:

```text
Next balance = current balance − repayment + new purchases
```

A limit reduction never erases debt. Purchases are capped by the credit available after repayment. Interest and fees contribute to the economic outcome; they are not added to principal in this model.

## Core concepts

### Different customers, each with a history

Income, spending, payments, and delinquency evolve over time. Customers also have persistent characteristics: creditworthiness, spending propensity, repayment propensity, and income stability.

These characteristics are related without assigning customers to deterministic profiles. Two customers with similar observable circumstances can therefore follow different paths.

### Incomplete information

The policy can observe the limit, balance, income, recent payments, delinquency, score, and current economic conditions, among other features. It cannot see the customer's hidden characteristics or future events.

This distinction captures a central challenge in credit decisions: acting on information that is useful but imperfect.

### Estimating risk and generating default are separate processes

The estimated **probability of default**, or **PD**, is the risk assessment available to the policy.

The **data-generating process**, or **DGP**, is the internal mechanism that drives the simulated world. It determines a default probability from the customer's circumstances, hidden characteristics, and economic conditions, then makes a random draw.

This separation makes it possible to study the consequences of imperfect risk estimation. The PD model can be wrong; its prediction does not itself determine whether the customer defaults.

### Changing economic conditions

Three regimes are represented: expansion, normal conditions, and stress. They affect income, spending, repayments, and default risk.

Regimes can evolve randomly or follow a prescribed scenario: baseline, mild stress, severe stress, or recovery. This supports focused questions, such as whether a policy behaves consistently when incomes deteriorate.

### Reinforcement learning

**Reinforcement learning**, or **RL**, learns a decision rule by interacting with an environment and observing the consequences of its actions.

Here, the learning signal combines interest and fee revenue, credit losses, funding costs, a capital charge, and a risk penalty. The objective is to study cumulative outcomes over a customer trajectory.

The repository provides a Gymnasium environment and an integration with PPO, an RL algorithm, alongside simple policies: constant limits, systematic adjustments, and risk-threshold rules. These baselines are essential for assessing the value of a learned approach. Including PPO is not evidence that it outperforms them.

## What the project adds

| Capability | Why it matters |
|---|---|
| Follow the effects of a decision over several months | Observe delayed consequences for debt, repayment, and losses |
| Compare policies using the same customers and random shocks | Reduce random noise and better isolate the effects of decisions |
| Separate estimated risk from the default mechanism | Study risk model errors and their economic consequences |
| Replay identical economic scenarios | Examine how sensitive policies are to stress |
| Break down results into their components | Understand why a policy gains or loses, beyond an aggregate score |

Random shocks are associated with a customer, a month, and an event type. They remain aligned across policies, even if one policy leads to an earlier default. Parameters, seeds, and provenance information are recorded to make experiments reproducible.

The result is a **testbed for formulating and testing hypotheses**, before considering validation against real data.

## What we measure

Experiments produce individual trajectories, tables, and charts to examine:

- default and delinquency rates;
- credit limits, balances, spending, and repayments;
- revenue, credit losses, and cumulative economic outcomes;
- differences between policies and economic scenarios;
- whether observable information alone contains a predictive risk signal.

Exports containing hidden characteristics, realized shocks, or internal probabilities are reserved for diagnostics. They must remain separate from the data used by the policy or risk model.

## Running the project

Python 3.11 or later is required. From the repository root, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,experiments]"
python -m pytest
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.

To run diagnostics on 5,000 customers:

```shell
python -m experiments.dgp_sanity --customers 5000 --seed 42
```

Tables are saved to `outputs/results/dgp/` and charts to `outputs/figures/dgp/`. To retain multiple runs, choose a separate directory with `--output outputs/my_experiment`.

To check the PPO integration with a short training run:

```shell
python -m pip install -e ".[rl]"
python -m experiments.train_ppo --timesteps 256
```

This short run checks that the integration works; it is not sufficient to assess policy quality.

Simulator parameters are in `configs/simulation.yaml`. Economic regimes and scenarios are in `configs/macro_scenarios.yaml`.

## Repository structure

```text
configs/                 Simulator and experiment parameters
src/credit_rl/
  simulation/            Customers, behavior, defaults, macro conditions, and shocks
  envs/                  Decision environment and observable information
  risk/                  Probability of default estimation
  policies/              Decision rules and agent integration
  evaluation/            Metrics and diagnostics
  reward.py              Economic outcome components
experiments/             Experiment scripts
tests/                   Consistency and reproducibility tests
docs/                    Specifications and assumptions
outputs/                 Generated results, excluded from Git tracking
```

## Scope and limitations

The data and behavior are **synthetic and have not been calibrated against a real bank portfolio**. Some configurations produce high cumulative default rates. A consistent, reproducible mechanism is not necessarily a quantitatively realistic representation of credit.

Delinquency, costs, and losses are simplified. The model does not implement a complete billing and collections ledger, and the value of the remaining portfolio at the end of the horizon is not included. Averages calculated over active customers can also change because the most vulnerable customers have already defaulted.

The results therefore support understanding mechanisms and designing experiments. Operational use would require empirical calibration, independent risk model validation, and analysis across multiple populations and scenarios.

For more detail, see the [DGP equations and structure](docs/dgp.md), [environment interface](docs/environment.md), and [data provenance](data/README.md).
