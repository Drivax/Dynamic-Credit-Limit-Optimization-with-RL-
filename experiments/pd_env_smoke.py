"""Verify a saved PD artifact inside the actual Gymnasium environment."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from credit_rl import CreditLimitEnv
from credit_rl.risk.features import build_features, feature_matrix
from credit_rl.risk.longitudinal import LongitudinalPDModel
from credit_rl.simulation.synthetic_snapshot import generate_synthetic_portfolio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("outputs/models/pd/logistic_calibrated.joblib"))
    parser.add_argument("--output", type=Path, default=Path("outputs/results/pd/env_smoke.json"))
    args = parser.parse_args()
    model = LongitudinalPDModel.load(args.model)
    population = generate_synthetic_portfolio(10, seed=91000)
    steps = 0
    started = perf_counter()
    with threadpool_limits(limits=1):
        for i in range(10):
            env = CreditLimitEnv(population, pd_model=model)
            env.reset(seed=91000+i, options={"customer_index": i})
            while True:
                opening_pd = env._predicted_pd
                obs, reward, terminal, truncated, info = env.step(env.static_action_index)
                assert info["decision_pd"] == opening_pd
                assert 0 <= obs[10] <= 1
                steps += 1
                if terminal or truncated:
                    break
            history = env.get_history()
            features = build_features(history)
            active = ~history.defaulted
            np.testing.assert_allclose(model.predict_proba(feature_matrix(features.loc[active], model.feature_names)),
                                       history.loc[active, "predicted_pd"], rtol=1e-12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(customers=10, steps=steps, seconds=perf_counter()-started,
                  offline_online_parity=True, model=str(args.model))
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
