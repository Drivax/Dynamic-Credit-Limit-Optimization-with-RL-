from dataclasses import replace, asdict
import itertools
import numpy as np
import pytest
from credit_rl.config import SimulationConfig
from credit_rl.portfolio.accounting import PortfolioConfig, aggregate, ead, monthly_pd, admissible
from credit_rl.portfolio.environment import PortfolioEnv, PORTFOLIO_FEATURES
from credit_rl.portfolio.policies import solve_choices, plan_month


def test_accounting_and_horizon():
    c=PortfolioConfig(size=2,el_budget_per_customer=100)
    np.testing.assert_allclose(ead([100,200],[300,100],.5),[200,200])
    s=aggregate([100,200],[300,100],[.12,.24],c)
    assert s.total_credit_limit==400 and s.total_balance==300 and s.total_ead==400
    assert s.expected_loss==pytest.approx((monthly_pd(.12)+monthly_pd(.24))*.55*200)
    assert s.risk_budget_utilization==s.expected_loss/200
    assert s.remaining_risk_budget==pytest.approx(200-s.expected_loss)
    np.testing.assert_array_equal(monthly_pd([0.,1.]),[0.,1.])


def test_inherited_deficits_are_explicit_not_erased():
    c=PortfolioConfig(size=1,el_budget_per_customer=1,high_risk_share=1.)
    before=aggregate([200],[400],[.9],c); lower=aggregate([200],[200],[.9],c)
    higher=aggregate([200],[800],[.9],c)
    assert before.risk_shortfall>0 and before.remaining_risk_budget==0
    assert admissible(before,lower,c) and not admissible(before,higher,c)
    assert lower.total_balance==200 and lower.risk_shortfall>0


def test_budget_macro_buffer_and_empty_book():
    c=PortfolioConfig(size=2,safety_factor=1.25)
    s=aggregate([100],[500],[.4],c,stress=True)
    assert s.expected_loss>s.raw_expected_loss and s.risk_budget==225
    z=aggregate([],[],[],c);assert z.high_risk_share==0 and z.expected_loss==0


@pytest.mark.parametrize('kwargs',[{'ccf':2},{'lgd':-.1},{'mode':'fake'},{'size':0},{'penalty':-1},{'safety_factor':float('nan')}])
def test_config_rejects_invalid(kwargs):
    with pytest.raises(ValueError): PortfolioConfig(**kwargs)


def test_optimizer_matches_enumeration():
    values=np.array([[0.,5,7],[0,4,6]])
    costs=np.array([[[0.,2,4],[0,2,3]]])
    choices,info=solve_choices(values,costs,np.array([4.]))
    best=max(sum(values[i,a] for i,a in enumerate(pair)) for pair in itertools.product(range(3),repeat=2)
             if sum(costs[0,i,a] for i,a in enumerate(pair))<=4)
    assert sum(values[np.arange(2),choices])==best and info['exact']


def make_env(**kwargs):
    base=SimulationConfig();return PortfolioEnv(base,None,PortfolioConfig(size=3,**kwargs),fixed_seed=331,record=True)


def test_public_boundary_budget_and_step_order():
    env=make_env();obs,_=env.reset(seed=1)
    assert len(obs)==34 and 'remaining_el_fraction' in PORTFOLIO_FEATURES
    assert set(env.public_month())=={'ids','observations','balance','limits','pd','state'}
    assert not any('true' in k or 'latent' in k for k in asdict(env.state))
    initial=[e.state for e in env._envs]
    _,reward,_,_,info=env.step(4)
    assert reward==0 and [e.state for e in env._envs]==initial
    assert set(info)=={'decision'} and env.month==0
    env.step(2);_,reward,_,_,info=env.step(2)
    assert env.month==1 and 'month_result' in info
    assert all(row['hard_nonworsening'] for row in env.decisions)
    assert info['month_result']['economic_value']==pytest.approx(info['month_result']['revenue']-info['month_result']['credit_loss']-info['month_result']['funding_cost'])


def test_reproducibility_and_common_shocks():
    a=make_env();b=make_env();oa,_=a.reset();ob,_=b.reset()
    np.testing.assert_array_equal(oa,ob)
    for _ in range(12):
        xa=a.step(2);xb=b.step(2);np.testing.assert_array_equal(xa[0],xb[0]);assert xa[1:]==xb[1:]
    assert a.decisions==b.decisions


def test_hard_action_rejection_and_soft_admission():
    hard=make_env(el_budget_per_customer=.001,high_risk_share=1.)
    soft=make_env(el_budget_per_customer=.001,high_risk_share=1.,mode='soft',penalty=2.)
    hard.reset();soft.reset()
    for _ in range(3): hard.step(4);soft.step(4)
    assert any(r['constraint_reason']=='portfolio_budget' for r in hard.decisions)
    assert not any(r['constraint_reason']=='portfolio_budget' for r in soft.decisions)
    assert soft.months[0]['penalty']>0 and hard.months[0]['penalty']==0


def test_baselines_public_only_and_greedy_feasible():
    env=make_env();env.reset();p=env.public_month()
    for name in ['Static','RiskBased','Greedy','Myopic']:
        plan,stats=plan_month(name,p,env.base,env.config)
        assert set(plan)==set(p['ids']) and all(0<=a<5 for a in plan.values())


def test_ppo_smoke():
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    import torch
    torch.set_num_threads(1)
    env=make_env(mode='soft',penalty=2.)
    check_env(env,warn=True)
    agent=PPO('MlpPolicy',env,n_steps=32,batch_size=16,n_epochs=1,seed=5,verbose=0)
    agent.learn(64);obs,_=env.reset();action,_=agent.predict(obs,deterministic=True)
    assert np.isfinite(obs).all() and env.action_space.contains(int(action))


def test_shared_capacity_couples_customers():
    c=PortfolioConfig(size=2,el_budget_per_customer=1000,ead_budget_per_customer=160,high_risk_share=1.)
    before=aggregate([100,100],[200,200],[.1,.1],c)
    first=aggregate([100,100],[240,200],[.1,.1],c)
    second=aggregate([100,100],[240,240],[.1,.1],c)
    assert admissible(before,first,c)
    assert not admissible(first,second,c)
    assert first.total_ead==320 and second.total_ead==340


def test_diagnostic_has_no_effect_on_actual_path_or_public_info():
    a=make_env();b=make_env();b.diagnostics=True
    oa,_=a.reset();ob,_=b.reset()
    for _ in range(6):
        xa=a.step(2);xb=b.step(2)
        np.testing.assert_array_equal(xa[0],xb[0]);assert xa[1:]==xb[1:]
        assert not any('true' in k or 'latent' in k for k in xb[4])
    assert all(np.isfinite(r['true_expected_loss']) for r in b.diagnostic_export())


def test_future_macro_not_visible_before_arrival():
    base=SimulationConfig();c=PortfolioConfig(size=3)
    a=PortfolioEnv(base,None,c,fixed_seed=9)
    b=PortfolioEnv(base,None,c,fixed_seed=9,macro_spec={'onset':8,'duration':5,'intensity':1.5})
    oa,_=a.reset();ob,_=b.reset();np.testing.assert_array_equal(oa,ob)
    for _ in range(6):
        xa=a.step(2);xb=b.step(2);np.testing.assert_array_equal(xa[0],xb[0]);assert xa[1:]==xb[1:]


def test_greedy_known_public_opportunities(monkeypatch):
    import credit_rl.portfolio.policies as module
    c=PortfolioConfig(size=2,el_budget_per_customer=2,high_risk_share=1.)
    state=aggregate([0,0],[0,0],[0,0],c)
    values=np.array([[0,0,0,6,0],[0,0,0,5,0]],float)
    resources=np.zeros((3,2,5));resources[0,:,3]=[3,2]
    monkeypatch.setattr(module,'opportunity_tables',lambda *args:(values,np.zeros((2,5)),resources))
    plan,_=module.plan_month('Greedy',{'ids':np.array([0,1]),'state':state},SimulationConfig(),c)
    assert plan=={0:2,1:3}  # Higher 5/2 ratio first; the other no longer fits.


def test_finite_portfolio_horizon_is_terminal_without_bootstrap():
    base=SimulationConfig();base=replace(base,environment=replace(base.environment,horizon=1))
    env=PortfolioEnv(base,None,PortfolioConfig(size=3),fixed_seed=73)
    env.reset()
    for j in range(3):
        observation,_,terminated,truncated,_=env.step(2)
        assert terminated==(j==2) and not truncated
    np.testing.assert_array_equal(observation,np.zeros(34))
