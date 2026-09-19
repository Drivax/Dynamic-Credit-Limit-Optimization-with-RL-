"""Future-window labels and purged, customer-disjoint cohort validation."""
import numpy as np
import pandas as pd

from .features import build_features


def build_dataset(trajectories, horizon_months=12):
    if type(horizon_months) is not int or horizon_months < 1:
        raise ValueError("PD horizon must be a positive integer")
    if trajectories.empty or not trajectories.index.is_unique:
        raise ValueError("Nonempty trajectories with unique row index required")
    features = build_features(trajectories)
    outputs, audit = [], []
    for customer, group in trajectories.groupby("customer_id", sort=False):
        group = group.sort_values("month")
        if (group.month < 0).any() or not np.equal(group.month, np.floor(group.month)).all():
            raise ValueError("Observation months must be nonnegative integers")
        if not group.defaulted.isin([True, False]).all():
            raise ValueError("Invalid default indicator")
        defaults = group.loc[group.defaulted, "month"]
        if len(defaults) > 1 or (len(defaults) and defaults.iloc[0] != group.month.max()):
            raise ValueError("Default must be absorbing and terminal")
        if group.scheduled_end.nunique() != 1 or group.cohort_start.nunique() != 1:
            raise ValueError("Inconsistent cohort metadata")
        event = defaults.iloc[0] if len(defaults) else np.inf
        observed_end = group.month.max()
        scheduled_end = int(group.scheduled_end.iloc[0])
        if observed_end > scheduled_end:
            raise ValueError("Follow-up exceeds administrative end")
        end = group.month + horizon_months
        active = ~group.defaulted
        # Administrative eligibility does NOT depend on whether default occurs.
        administrative = end <= scheduled_end
        known = (end <= observed_end) | ((event > group.month) & (event <= end))
        keep = active & administrative & known
        chosen = group.loc[keep, ["customer_id", "month", "cohort_start", "macro_state"]].copy()
        chosen["observation_month"] = chosen.cohort_start + chosen.month
        chosen["label_end"] = chosen.observation_month + horizon_months
        chosen["target"] = ((event > chosen.month) & (event <= chosen.month + horizon_months)).astype(int)
        outputs.append(pd.concat([chosen, features.loc[chosen.index]], axis=1))
        audit.append(dict(customer_id=customer, rows=len(group), retained=int(keep.sum()),
            already_defaulted=int((~active).sum()), administrative_excluded=int((active & ~administrative).sum()),
            censored_unknown=int((active & administrative & ~known).sum()), event_observed=bool(len(defaults))))
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(audit)


def assert_split_integrity(samples):
    """All label windows finish before the next cohort's observation period."""
    previous_end = -np.inf
    identities = set()
    for name, frame in samples.items():
        if frame.empty:
            raise ValueError(f"Empty split {name}")
        ids = set(frame.customer_id)
        if ids & identities:
            raise ValueError("Customer overlap between splits")
        if frame.observation_month.min() <= previous_end:
            raise ValueError("Temporal overlap: previous labels not yet mature")
        identities.update(ids)
        previous_end = frame.label_end.max()


def sample_summary(frame):
    return dict(customers=int(frame.customer_id.nunique()), snapshots=len(frame),
        positive_snapshots=int(frame.target.sum()), default_prevalence=float(frame.target.mean()),
        customers_with_positive_snapshot=int(frame.loc[frame.target == 1, "customer_id"].nunique()),
        observation_start=int(frame.observation_month.min()), observation_end=int(frame.observation_month.max()),
        last_label_end=int(frame.label_end.max()))
