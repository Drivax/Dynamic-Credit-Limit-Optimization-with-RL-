"""Read-only scientific integrity checks for completed portfolio artifacts."""
import argparse
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from experiments.study_common import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--profile',default='standard')
    args=parser.parse_args();root=Path('outputs/results/portfolio')/args.profile
    registry=Path('outputs/experiments/portfolio')/args.profile
    manifest=json.loads((registry/'manifest.json').read_text())
    for file,digest in manifest['sources'].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,file
    for file,digest in json.loads((root/'model_hashes.json').read_text()).items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,file
    assert hashlib.sha256(Path('outputs/models/pd/logistic_calibrated.joblib').read_bytes()).hexdigest()==manifest['pd_sha256']
    data=pd.read_csv(root/'portfolios.csv');monthly=pd.read_csv(root/'monthly.csv.gz');actions=pd.read_csv(root/'actions.csv.gz')
    keys=['case','policy','policy_seed','portfolio']
    assert not data.duplicated(keys).any()
    expected=0
    for case in json.loads((root/'cases.json').read_text()):
        part=data[data.case==case['name']]
        assert part.groupby(['policy','policy_seed']).size().eq(case['count']).all()
        expected+=case['count']*(6+3*len(manifest['settings']['profiles'][args.profile]['seeds']))
    assert len(data)==expected
    assert actions[actions['mode']=='hard'].hard_nonworsening.all()
    assert actions.requested_action.between(0,4).all() and actions.effective_action.between(0,4).all()
    assert actions.effective_change.between(-.20000001,.20000001).all()
    np.testing.assert_allclose(monthly.economic_value,monthly.revenue-monthly.credit_loss-monthly.funding_cost,atol=1e-8)
    np.testing.assert_allclose(monthly.objective,monthly.economic_value-monthly.penalty,atol=1e-8)
    np.testing.assert_allclose(monthly.risk_budget_utilization,monthly.expected_loss/monthly.risk_budget)
    np.testing.assert_allclose(monthly.remaining_risk_budget,np.maximum(0,monthly.risk_budget-monthly.expected_loss))
    np.testing.assert_allclose(monthly.risk_shortfall,np.maximum(0,monthly.expected_loss-monthly.risk_budget))
    cats=['both_safe','both_violated','predicted_safe_true_violated','predicted_violated_true_safe']
    assert monthly[cats].sum(axis=1).eq(1).all()
    counts=actions.groupby(keys+['month']).size().rename('actual_decisions').reset_index()
    check=monthly.merge(counts,on=keys+['month'],validate='one_to_one')
    assert check.active_customers.eq(check.actual_decisions).all()
    # Reconstruct risk stocks independently from every executed customer action.
    case_configs={c['name']:c['config'] for c in json.loads((root/'cases.json').read_text())}
    ccf=actions.case.map({k:v['ccf'] for k,v in case_configs.items()})
    lgd=actions.case.map({k:v['lgd'] for k,v in case_configs.items()})
    horizon=actions.case.map({k:v['pd_horizon'] for k,v in case_configs.items()})
    threshold=actions.case.map({k:v['high_risk_pd'] for k,v in case_configs.items()})
    actions['reconstructed_ead']=actions.balance_before+ccf*np.maximum(actions.limit_after-actions.balance_before,0)
    q=1-np.power(1-np.minimum(actions.safety_factor*actions.predicted_pd,1),1/horizon)
    raw_q=1-np.power(1-actions.predicted_pd,1/horizon)
    actions['reconstructed_el']=q*lgd*actions.reconstructed_ead
    actions['reconstructed_raw_el']=raw_q*lgd*actions.reconstructed_ead
    actions['reconstructed_high_risk']=actions.reconstructed_ead*(actions.predicted_pd>threshold)
    before_ead=actions.balance_before+ccf*np.maximum(actions.limit_before-actions.balance_before,0)
    actions['audit_initial_ead']=before_ead
    actions['audit_initial_el']=q*lgd*before_ead
    actions['audit_initial_high']=before_ead*(actions.predicted_pd>threshold)
    for suffix,after in [('ead','reconstructed_ead'),('el','reconstructed_el'),('high','reconstructed_high_risk')]:
        actions[f'audit_delta_{suffix}']=actions[after]-actions[f'audit_initial_{suffix}']
        recorded='delta_high_risk' if suffix=='high' else f'delta_{suffix}'
        np.testing.assert_allclose(actions[f'audit_delta_{suffix}'],actions[recorded],atol=1e-7)
        group=actions.groupby(keys+['month'],sort=False)
        actions[f'audit_before_{suffix}']=group[f'audit_initial_{suffix}'].transform('sum')+group[f'audit_delta_{suffix}'].cumsum()-actions[f'audit_delta_{suffix}']
    exposure_cap=actions['size']*actions.case.map({k:v['ead_budget_per_customer'] for k,v in case_configs.items()})
    high_cap=actions.case.map({k:v['high_risk_share'] for k,v in case_configs.items()})
    before_vectors=np.column_stack([actions.audit_before_el-actions.risk_budget,
        actions.audit_before_ead-exposure_cap,actions.audit_before_high-high_cap*actions.audit_before_ead])
    delta_vectors=np.column_stack([actions.audit_delta_el,actions.audit_delta_ead,
        actions.audit_delta_high-high_cap*actions.audit_delta_ead])
    assert (before_vectors[actions['mode']=='hard']+delta_vectors[actions['mode']=='hard']<=np.maximum(before_vectors[actions['mode']=='hard'],0)+1e-7).all()
    fields={'reconstructed_ead':'total_ead','reconstructed_el':'expected_loss',
        'reconstructed_raw_el':'raw_expected_loss','reconstructed_high_risk':'high_risk_exposure',
        'balance_before':'total_balance','limit_after':'total_credit_limit'}
    reconstructed=actions.groupby(keys+['month'])[list(fields)].sum().reset_index()
    check=monthly.merge(reconstructed,on=keys+['month'],validate='one_to_one')
    for observed,recorded in fields.items():
        np.testing.assert_allclose(check[observed],check[recorded],atol=1e-7)
    sums=monthly.groupby(keys)[['economic_value','credit_loss','defaults']].sum().reset_index()
    check=data.merge(sums,on=keys,validate='one_to_one')
    np.testing.assert_allclose(check.value,check.economic_value/check['size'])
    np.testing.assert_allclose(check.loss,check.credit_loss/check['size'])
    np.testing.assert_allclose(check.default_rate,check.defaults/check['size'])
    logs=pd.read_csv(root/'portfolio_ope_logs.csv.gz');mu=logs[[f'mu_{i}' for i in range(5)]].to_numpy()
    assert mu.min()>=.04-1e-12;np.testing.assert_allclose(mu.sum(axis=1),1.)
    np.testing.assert_allclose(mu[np.arange(len(logs)),logs.action],logs.behavior_probability)
    for _,g in logs.groupby('portfolio'):
        assert g.step.tolist()==list(range(len(g))) and (g.terminated|g.truncated).sum()==1
        assert not g.truncated.any() and bool(g.iloc[-1].terminated)
        np.testing.assert_array_equal(g[[f'next_observation_{i}' for i in range(34)]].to_numpy()[:-1],g[[f'observation_{i}' for i in range(34)]].to_numpy()[1:])
    assert not any('true' in c or 'latent' in c for c in logs.columns)
    record=dict(timestamp=datetime.now(timezone.utc).isoformat(),experiment_id=manifest['experiment_id'],
        portfolio_episodes=len(data),decisions=int(data.decisions.sum()),portfolio_months=len(monthly),
        cases=data.case.nunique(),complete_panel=True,frozen_sources_models_PD=True,
        accounting_reconciled=True,customer_risk_aggregates_reconstructed=True,
        hard_admission_nonworsening=True,hard_admission_recomputed=True,
        ope_complete_public_trajectories=True,finite_horizon_terminal=True,
        ope_portfolios=logs.portfolio.nunique(),ope_decisions=len(logs))
    test_file=Path('outputs/portfolio_pytest.xml')
    if test_file.exists():
        import xml.etree.ElementTree as ET
        suite=ET.parse(test_file).getroot().find('testsuite')
        record['test_suite']={k:int(suite.attrib[k]) for k in ('tests','failures','errors','skipped')}
        record['test_suite']['seconds']=float(suite.attrib['time'])
    replay_file=root/'parallel_verification.json'
    if replay_file.exists():record['serial_parallel_replay']=json.loads(replay_file.read_text())
    write_json(root/'verification.json',record)
    sources=['src/credit_rl/portfolio/reporting.py','experiments/portfolio_diagnostics.py','experiments/verify_portfolio.py']
    write_json(registry/'report_manifest.json',dict(timestamp=record['timestamp'],experiment_id=manifest['experiment_id'],
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}))
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
