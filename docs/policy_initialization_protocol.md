# Phase D preregistration

This protocol is frozen before Phase D data generation. Machine-readable choices
are in `configs/policy_initialization.yaml`. The Phase C decision gate is
ScenarioDependent + Unresolved: its partial imitation did not improve teacher
regret over canonical PPO. It therefore cannot itself qualify the new treatment.

The frozen F0 teacher is never refitted. New customer-disjoint train, validation
and test populations use a new population seed. Each customer contributes all
pre-action visits under one of three equally allocated source policies (canonical
PPO, F0, myopic). This defines the natural mixture estimand, rather than claiming
it is the stationary distribution of a single policy. The two macro versions of
a customer remain in the same partition. No balanced distribution is used.
Only the exact current 21D public observation enters imitation. Customer IDs,
source names, macro scenario labels, snapshots and future outcomes are metadata
for partitioning or diagnostics, never actor inputs. Test arrays are sealed in a
separate artifact and are not loaded for fitting, epoch selection or qualification.

Five actors are trained with ordinary cross entropy and a fixed 600-epoch budget;
minimum validation cross entropy selects the epoch. Qualification uses customer-
clustered, paired seed/customer bootstrap intervals against the corresponding
frozen canonical selected actors, separately in baseline and stress. Effective
agreement must improve significantly and fitted-teacher regret must decrease
significantly; diversity and action-support requirements are specified in YAML.
Both scenarios must pass. Only an architecture fallback from 64x64 to 128x128 is
allowed. Failure of both terminates the scientific preservation experiment with
Unresolved; no forced training of a scientifically unqualified treatment.

If qualified, only actor tensors are transferred. The critic is initialized by
the canonical protocol and copied identically between paired arms. Transfer
checks compare logits, probabilities and greedy actions before any PPO update.
PPO receives no teacher targets, Q values or imitation loss. Diagnostics have no
gradient or checkpoint-selection role. Canonical validation selection remains
unchanged. Temporal snapshots occur after updates; selected checkpoints may be
saved before the pending update at the same timestep. Their identities and
hashes are retained separately.

Independent short and extended runs share seeds, initial weights, training
populations and exogenous paths. Endogenous state evolution, default times and
therefore subsequent visited customers can diverge. Minibatch ordering uses the
same RNG protocol where compatible; identical trajectories are not assumed.
Stochastic evaluation uses local customer/month uniforms and does not consume
training RNG. Evaluation includes both greedy and stochastic policies. Test
results never select architectures, epochs, checkpoints or additional experiments.

Fitted teacher regret is a surrogate. Independent H=12 counterfactuals use shared
action draws, separate selection/evaluation banks and a frozen canonical
continuation; these are first-action comparisons, not lifetime policy values.
Teacher drift and economic rollout changes are interpreted separately. All
temporal intervals are descriptive; snapshots are correlated. Optional anchored
loss and critic warm-start experiments are not run in this phase.

Historical artifacts and scientific inputs are hashed before work, including
all canonical results and Phase A/B/C outputs. README and the technical paper
may receive append-only Phase D conclusions; their historical prefixes are
protected. No new phase is started and no DGP, reward or PPO tuning is permitted.
