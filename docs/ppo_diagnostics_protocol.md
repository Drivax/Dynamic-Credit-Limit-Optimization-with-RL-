# Phase C: controlled PPO learning diagnostics

Predeclared before intervention execution. Preserve Phase A/B result directories,
their scientific reports, all canonical models and simulator/reward source hashes.
Only local outputs are authorized. No next-phase economic or algorithm change.

## Hypotheses and design

H0 reproduces canonical seeds 101/202/303 and adds 404/505. H1 changes GAE lambda;
H2 critic capacity or critic-only optimization; H3 entropy and its schedule;
H4 budget; H5 actor capacity and supervised imitation; H6 discount; H7 scale;
H8 clipping; H9 learning rate; H10 rollout length. Exact values, seeds and budgets
are frozen in `configs/ppo_diagnostics.yaml`. Shared canonical cells are not rerun
as independent treatments. All negative results are retained. Smoke checks software.

The canonical actor and critic already have separate 64x64 Tanh MLPs. They share
only a parameter-free FlattenExtractor. C2 is therefore not an additional
intervention: there are no shared learned feature layers to separate. C1 changes
only vf to 128x128. C3 adds five ordered full-rollout critic-only passes after the
unchanged PPO update, with a separate Adam optimizer over vf parameters only.
Its effect includes additional critic optimization, not additional policy epochs.
C4 is a supervised current-observation value benchmark, diagnostic only.

The entropy schedule is fixed at .02 -> .01 linearly over 32768 transitions,
tested after constant-entropy experiments. No adaptive entropy algorithm is needed.
Actor-capacity experiments change actor layers only, leaving critic architecture
fixed. Budgets are exact powers of two (32768/65536/131072/262144), with constant
LR/clip/entropy; all intermediate validation trajectories and final actors remain
available. Reward scaling changes only numerical units entering PPO. Economics and
raw reported rewards remain unchanged. Gamma treatments report both their own
training discount and common .98-discounted endpoints; selection uses own gamma.

## Populations and selection

Reuse canonical Markov RL training and baseline validation customers. Fine
diagnostic validation uses the first eight canonical validation customers in
both fixed scenarios; these scores NEVER select checkpoints. Canonical selection
still uses all 100 baseline validation customers every 8192 steps and after final
update, with earliest ties. Fine probes and snapshots are read-only, RNG-neutral.
The original selected/final parameter tensors must match for seeds 101/202/303.

A new fixed population seed creates 100 untouched test customers, paired across
baseline and severe stress using identical customer identities and shock paths.
No test values enter intervention, confirmation, horizon, epoch or checkpoint
selection. Frozen Phase B F0 is the observable teacher; its original customer-
disjoint train/validation states may support imitation training/selection.
Imitation test metrics use the new Phase C population, never Phase B test labels
for selection. Architecture and epoch budgets are fixed in advance.

## Temporal and MC estimands

Log every actual PPO update: raw and actual minibatch-normalized advantages,
probabilities/logits, sampled actions, entropy, gradient norms and SB3 losses.
Evaluate a fixed observable validation-state panel at each update. Dense canonical
snapshots and diagnostic validation occur every 512 steps; interventions every
2048 steps (rollout boundaries take precedence). Save the actual model before
the next update with its checksum. Collapse time is the first strict crossing
of .5/.75/.90/.95, separately for mean stochastic contraction probability and
deterministic contraction fraction. Report censoring and recrossings, not only
first passage. Entropy alone is not state dependence; also report action diversity
and empirical mutual information with public PD/utilization/horizon/macro buckets.

Temporal ordering is descriptive, not identification of a causal chain. Controlled
one-factor interventions provide stronger evidence but may change visitation.
Actual single-trajectory GAE and independently simulated conditional MC advantages
are different noisy quantities; report uncertainty and sample sizes. For an equal
fixed full-state validation panel, counterfactual GAE is reconstructed from canonical
transitions under the frozen behavior actor, current critic and actual gamma/lambda;
compare to independent long-horizon stochastic actor returns using split banks.
Neither future shocks nor hidden traits enter any actor. MC conditioning on a
fixed future scenario is explicitly privileged diagnostic conditioning.

First-action Q comparisons use CRN across actions, and independent draw halves
when selecting a full-state reference. Report fitted ObservationPlanner regret
separately from independent simulator regret. Fixed panels allow maps to compare
actors at the same states. Sparse MC temporal probes are labelled missing between
measurements; do not interpolate observations into evidence. Dense rollout critic
errors versus GAE returns are explicitly in-sample, not independent MC accuracy.

## Statistical and confirmation gates

Primary endpoints: common .98 discounted reward, undiscounted net economic value,
contraction share, policy entropy, collapse time and ObservationPlanner regret.
Use paired customer/seed bootstrap separately per macro scenario, with treatment
x scenario differences reported. Exploratory intervals are descriptive and not
family-wise corrected. Five seeds support canonical/capacity/critic/imitation;
three support lighter OFAT diagnostics. Never discard a seed.

After OFAT completion, apply only the validation/mechanism-based eligibility and
ranking in YAML. Freeze at most two factors and a 2^k confirmation matrix before
running five fresh seeds. Include a fresh canonical control, all interactions and
negative cells. If no mechanism qualifies, record that confirmation was not
triggered; do not manufacture a winning combination. Test outcomes never determine
eligibility. No opaque BestPPO label.

Classification requires converging evidence across seeds and scenarios. Delayed
collapse alone does not prove exploration limitation; it must recover state
dependence and reduce held-out regret. Imitability by 2x64 weakens a representation
failure explanation but does not prove global approximation sufficiency. Critic
improvement must accompany advantage/action improvements to implicate the critic.
Scenario reversals prevent a claim of general improvement. Unresolved and mixed
interpretations remain admissible. Recommend the next experiment; do not implement it.
