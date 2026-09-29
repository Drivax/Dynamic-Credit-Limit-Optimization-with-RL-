"""Same-PPO-actor supervised imitation with public observations and validation-only epochs."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from threadpoolctl import threadpool_limits
import torch

from credit_rl import CreditLimitEnv
from credit_rl.experiments.ppo_diagnostics import prepare, protection, diagnostic_provenance
from credit_rl.experiments.ppo_measurements import diagnostic_panel, evaluate_actor, policy_measurements
from credit_rl.policies.information import choose


def fit_actor(observations, labels, weights, validation, validation_labels, config, risk,
              architecture, seed, settings, protocol, output):
    if observations.ndim != 2 or observations.shape[1] != 21 or validation.shape[1] != 21:
        raise ValueError('Imitation receives exactly 21 observable features')
    env = CreditLimitEnv(config=config, pd_model=risk)
    model = PPO('MlpPolicy', env, seed=seed, device='cpu', n_steps=64, batch_size=64,
        policy_kwargs=dict(net_arch=dict(pi=architecture, vf=[64, 64])))
    actor_parameters = list(model.policy.mlp_extractor.policy_net.parameters())+list(model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(actor_parameters, lr=settings['learning_rate'], eps=1e-5)
    x, y = torch.as_tensor(observations, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.long)
    w = torch.as_tensor(weights/np.mean(weights), dtype=torch.float32)
    vx = torch.as_tensor(validation, dtype=torch.float32)
    vy = torch.as_tensor(validation_labels, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed+9090)
    best, rows = float('inf'), []
    for epoch in range(protocol['imitation_epochs']+1):
        if epoch:
            order = torch.randperm(len(x), generator=generator)
            for start in range(0, len(x), settings['batch_size']):
                indices = order[start:start+settings['batch_size']]
                logits = model.policy.get_distribution(x[indices]).distribution.logits
                loss = (torch.nn.functional.cross_entropy(logits, y[indices], reduction='none')*w[indices]).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if epoch % protocol['imitation_check_every'] == 0 or epoch == protocol['imitation_epochs']:
            with torch.no_grad():
                logits = model.policy.get_distribution(vx).distribution.logits
                loss = float(torch.nn.functional.cross_entropy(logits, vy))
                accuracy = float((logits.argmax(1) == vy).float().mean())
            rows.append(dict(epoch=epoch, validation_cross_entropy=loss, validation_accuracy=accuracy))
            if loss < best:
                best = loss
                model.save(output/'selected.zip')
                selected = epoch
    pd.DataFrame(rows).to_csv(output/'learning_curve.csv', index=False)
    (output/'selection.json').write_text(json.dumps(dict(selected_epoch=selected,
        criterion='minimum validation cross entropy; earliest tie', features='public 21D only',
        architecture=architecture, seed=seed), indent=2))
    env.close()
    return PPO.load(output/'selected.zip', device='cpu')


def run(args):
    config, settings, matrix, protocol, risk = prepare(args)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        data = joblib.load(args.phase_b/'dataset.joblib')
        teacher = joblib.load(args.phase_b/'planners.joblib')['F0']['model']
        train = data['states'].role.eq('train').to_numpy()
        validation = data['states'].role.eq('validation').to_numpy()
        x = data['features']['F0'].astype(np.float32)
        labels = np.array([choose(q, o, config) for q, o in zip(teacher.predict(x), x)])
        test = diagnostic_panel(args, config, settings, protocol, risk)
        populations = joblib.load(args.output/'panels.joblib')['test']
        metrics, confusion, states, episodes = [], [], [], []
        for architecture in matrix['imitation']['architectures']:
            name = 'Imitation_'+'x'.join(map(str, architecture))
            for seed in protocol['important_seeds']:
                output = args.output/'imitation_models'/name/str(seed)
                output.mkdir(parents=True, exist_ok=True)
                if (output/'selection.json').exists():
                    model = PPO.load(output/'selected.zip', device='cpu')
                else:
                    model = fit_actor(x[train], labels[train], data['states'].weight[train].to_numpy(),
                        x[validation], labels[validation], config, risk, architecture, seed,
                        matrix['imitation'], protocol, output)
                frame, summary = policy_measurements(model, teacher, test, config, name, seed)
                states.append(frame)
                for scenario, group in frame.groupby('scenario'):
                    row = summary[summary.scenario == scenario].iloc[0].to_dict()
                    row.update(accuracy=np.average(group.agreement, weights=group.weight),
                        effective_accuracy=np.average(group.effective_agreement, weights=group.weight))
                    metrics.append(row)
                    counts = group.groupby(['teacher_action', 'action']).weight.sum().reset_index()
                    confusion.append(counts.assign(experiment_id=name, seed=seed, scenario=scenario))
                e, _ = evaluate_actor(model, config, settings, risk, populations, seed, name)
                episodes.append(e)
                diagnostic_provenance(args, output/'provenance.json',
                    dict(seed=seed, architecture=architecture, budget_epochs=protocol['imitation_epochs'],
                         imitation=matrix['imitation'], scenario='baseline / severe_stress',
                         selection=json.loads((output/'selection.json').read_text())), output/'selected.zip')
                print(f'IMITATION {name} seed={seed}', flush=True)
        pd.DataFrame(metrics).to_csv(args.output/'imitation.csv', index=False)
        pd.concat(confusion, ignore_index=True).to_csv(args.output/'imitation_confusion.csv', index=False)
        pd.concat(states, ignore_index=True).to_csv(args.output/'imitation_states.csv', index=False)
        pd.concat(episodes, ignore_index=True).to_csv(args.output/'imitation_episodes.csv', index=False)
    if json.loads((args.output/'protected_artifacts.json').read_text()) != protection(args.canonical, args.phase_b):
        raise AssertionError('Protected artifacts changed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=['standard', 'smoke'], default='standard')
    p.add_argument('--canonical', type=Path, default=Path('outputs/main/standard'))
    p.add_argument('--phase-b', type=Path, default=Path('outputs/main/information_gap'))
    p.add_argument('--output', type=Path, default=Path('outputs/main/ppo_diagnostics'))
    run(p.parse_args())


if __name__ == '__main__':
    main()
