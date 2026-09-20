from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from credit_rl import CreditLimitEnv, SimulationConfig
from credit_rl.evaluation.worlds import sample_evaluation_world, build_worlds, make_cohort, in_world, perturb
from credit_rl.evaluation.sensors import ObservationSensor
from credit_rl.evaluation.ope import trajectory_weights, estimates, MixturePolicy, SoftTarget
from credit_rl.evaluation.policy_engine import evaluate_policy
from credit_rl.policies.registry import baseline_specs
from credit_rl.policies.baselines import ConstantPolicy
from credit_rl.evaluation.robustness_statistics import constraint_violations, crossed_interval, panel


def settings(): return yaml.safe_load(Path("configs/robustness.yaml").read_text())


def test_worlds_reproducible_bounded_signed_and_nominal_unchanged():
    base=SimulationConfig(); before=asdict(base); s=settings()
    for seed in range(40):
        a=sample_evaluation_world(base,s,seed,str(seed))
        assert a==sample_evaluation_world(base,s,seed,str(seed))
        for name,value in a.parameters.items():
            assert s['parameters'][name][0]<=value<=s['parameters'][name][1]
        assert a.config.default.months_delinquent>0 and a.config.default.utilization>0
        assert set(a.metadata()) >= {'world_id','world_seed','kind','parameters','macro','environment'}
    assert asdict(base)==before
    with pytest.raises(ValueError,match='signs'): perturb(base,{'default.months_delinquent':-.1})


def test_shifted_world_shares_initial_traits_shocks_and_does_not_change_rules():
    base=SimulationConfig(); s=settings()
    cohort=make_cohort(base,count=3,seed=123,namespace='HOLDOUT')
    world=sample_evaluation_world(base,s,7,'shift')
    changed=in_world(cohort,world)
    for a,b in zip(cohort,changed):
        assert a.initial_state==b.initial_state and a.traits==b.traits and a.shock_path==b.shock_path
    benchmark=yaml.safe_load(Path('configs/policy_evaluation.yaml').read_text())
    specs=baseline_specs(base,benchmark,include_oracle=False)
    # A minimal risk handle suffices for factory construction; no hidden state is passed to the policy.
    class Risk:
        metadata={'horizon_months':12}
    env=CreditLimitEnv(config=world.config); env.pd_model=Risk()
    for spec in specs:
        policy=spec.factory(env,changed[0])
        assert policy.config==base
        assert not hasattr(policy,'env')


def test_sensor_identity_bounds_noise_and_causal_lag():
    obs=np.linspace(0,1,21,dtype=np.float32); obs[10]=.3
    np.testing.assert_array_equal(ObservationSensor('c',1)(obs),obs)
    a=ObservationSensor('c',3,intercept=-.7,slope=1.35,noise=.8)
    b=ObservationSensor('c',3,intercept=-.7,slope=1.35,noise=.8)
    for _ in range(30):
        x=a(obs); np.testing.assert_array_equal(x,b(obs)); assert 0<=x[10]<=1
        np.testing.assert_array_equal(np.delete(x,10),np.delete(obs,10))
    lag=ObservationSensor('c',1,lag=1)
    assert lag(obs)[10]==obs[10]
    current=obs.copy(); current[10]=.9
    assert lag(current)[10]==obs[10]
    assert lag(obs)[10]==current[10]
    measured=ObservationSensor('c',5,measurement_sigma=.05)(obs)
    assert np.isfinite(measured).all() and ((measured>=0)&(measured<=1)).all()


def test_actor_sensor_cannot_change_static_world_or_reward():
    base=SimulationConfig(); benchmark=yaml.safe_load(Path('configs/policy_evaluation.yaml').read_text())
    scenarios=make_cohort(base,count=3,seed=91,namespace='SENSOR_TEST')
    spec=baseline_specs(base,benchmark,include_oracle=False)[0]
    a,_,_=evaluate_policy(spec,scenarios,base,None,benchmark,'normal')
    biased=replace(spec,observation_transform_factory=lambda s:ObservationSensor(s.customer_id,9,noise=2,measurement_sigma=.1))
    b,_,_=evaluate_policy(biased,scenarios,base,None,benchmark,'normal')
    pd.testing.assert_frame_equal(a,b)


def test_known_episodic_weights_is_wis_and_ess():
    ids,w=trajectory_weights([.5,.5,.25],[1,1,.5],['a','a','b'])
    np.testing.assert_array_equal(ids,['a','b']); np.testing.assert_allclose(w,[4,2])
    v=estimates([10,20],w)
    assert v['IS']==40 and v['WIS']==pytest.approx(80/6) and v['ESS']==pytest.approx(36/20)
    clipped=estimates([10,20],w,clip=2)
    assert clipped['IS']==30 and clipped['WIS']==15 and clipped['ESS']==2


def test_behavior_identity_zero_weights_and_invalid_propensities():
    _,w=trajectory_weights([.2,.8,.4],[.2,.8,.4],['a','a','b'])
    np.testing.assert_array_equal(w,[1,1])
    stats=estimates([3,7],w); assert stats['IS']==stats['WIS']==5 and stats['ESS']==2
    _,zeros=trajectory_weights([.2,.3],[0,0],['a','b'])
    stats=estimates([3,7],zeros); assert stats['IS']==0 and np.isnan(stats['WIS']) and stats['ESS']==0
    with pytest.raises(ValueError): trajectory_weights([0],[1],['a'])


def test_stochastic_behavior_support_and_reproducibility():
    policies={'rule':ConstantPolicy(2)}; weights={'rule':.8,'uniform':.2}
    a=MixturePolicy(policies,weights,3); b=MixturePolicy(policies,weights,3)
    p=a.probabilities(np.zeros(21)); np.testing.assert_allclose(p,[.04,.04,.84,.04,.04])
    assert [a.act(np.zeros(21)) for _ in range(30)]==[b.act(np.zeros(21)) for _ in range(30)]
    soft=SoftTarget(a,ConstantPolicy(0),.8,1)
    assert soft.probabilities(np.zeros(21)).sum()==pytest.approx(1)


def test_public_logging_callback_accounting_and_done():
    base=SimulationConfig(); bench=yaml.safe_load(Path('configs/policy_evaluation.yaml').read_text())
    scenarios=make_cohort(base,count=2,seed=71,namespace='LOG_TEST')
    rows=[]; spec=baseline_specs(base,bench,include_oracle=False)[0]
    e,_,_=evaluate_policy(spec,scenarios,base,None,bench,'test',transition_observer=rows.append)
    assert len(rows)==e.steps.sum()
    assert all(set(r)=={'customer_id','month','observation','action','reward','economic_value','next_observation','terminated','truncated'} for r in rows)
    assert all(len(r['observation'])==21 for r in rows)
    assert sum(r['reward'] for r in rows)==pytest.approx(e.cumulative_reward.sum())
    assert sum(r['economic_value'] for r in rows)==pytest.approx(e.net_economic_value.sum())
    assert sum(r['terminated'] or r['truncated'] for r in rows)==2


def test_risk_violation_frequency_magnitude_and_boundary():
    f=pd.DataFrame(dict(policy=['A']*3,default_rate=[.5,.6,.8],credit_loss_rate=[.2,.2,.2],high_risk_exposure_share=[.1,.1,.1]))
    v=constraint_violations(f,dict(max_default_rate=.6,max_credit_loss_rate=.2,max_high_risk_exposure_share=.3))
    default=v[v.metric=='default_rate'].iloc[0]
    assert default.violation_frequency==pytest.approx(1/3)
    assert default.mean_excess==pytest.approx(.2/3) and default.worst_excess==pytest.approx(.2)
    assert v[v.metric=='credit_loss_rate'].violation_frequency.iloc[0]==0


def test_crossed_bootstrap_keeps_constant_paired_effect_and_rejects_missing_panel():
    values=np.full((4,3,6),2.)
    np.testing.assert_array_equal(crossed_interval(values,50,3),[2,2])
    f=pd.DataFrame(dict(world_id=['w1','w1','w2'],policy_seed=[1,1,1],customer_id=['a','b','a'],value=[1,2,3]))
    with pytest.raises(ValueError,match='Incomplete'): panel(f,'value')
