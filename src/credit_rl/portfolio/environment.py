"""One episode is a synchronized closed portfolio; one step is an allocation."""
from dataclasses import asdict, replace
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from credit_rl.envs.credit_limit_env import CreditLimitEnv
from credit_rl.envs.constraints import effective_limit
from credit_rl.evaluation.sensors import ObservationSensor
from credit_rl.evaluation.worlds import make_cohort, stress_path
from .accounting import PortfolioConfig, aggregate, admissible, constraint_vector

PORTFOLIO_FEATURES=('el_utilization','remaining_el_fraction','el_shortfall_fraction','ead_utilization',
    'high_risk_share','average_pd','active_fraction','allocation_progress','mean_ead_scaled',
    'risk_budget_per_initial_scaled','cumulative_loss_scaled','dynamic_budget','safety_factor_scaled')


class PortfolioEnv(gym.Env):
    metadata={'render_modes':[]}

    def __init__(self, base, pd_model, portfolio=None, *, world_config=None, macro_spec=None,
                 sensor=None, namespace='PORTFOLIO_TRAIN', fixed_seed=None, record=False, diagnostics=False):
        self.base=base; self.world_config=world_config or base; self.pd_model=pd_model
        self.config=portfolio or PortfolioConfig(); self.macro_spec=macro_spec or {}
        if abs(self.config.lgd-self.world_config.reward.loss_given_default)>1e-12:
            raise ValueError('Portfolio LGD must equal the realized-loss LGD')
        self.sensor_config=sensor or {}; self.namespace=namespace; self.fixed_seed=fixed_seed
        self.record=record; self.diagnostics=diagnostics
        self.action_space=spaces.Discrete(5)
        self.observation_space=spaces.Box(0,1,(21+len(PORTFOLIO_FEATURES),),dtype=np.float32)
        self._done=True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options: raise ValueError('Portfolio reset accepts seed only')
        self.episode_seed=int(self.fixed_seed if self.fixed_seed is not None else self.np_random.integers(1,2**31))
        cohort=make_cohort(self.base,count=self.config.size,seed=self.episode_seed,namespace=f'{self.namespace}_{self.episode_seed}')
        path=stress_path(self.world_config,self.macro_spec)
        self._envs=[]; self._obs=[]; self._sensors=[]
        for scenario in cohort:
            env=CreditLimitEnv(config=self.world_config,pd_model=self.pd_model,record_history=False,severe_delinquency_months=3)
            observation,_=env.reset(seed=scenario.customer_seed,options=replace(scenario,macro_path=path).reset_options())
            self._envs.append(env); self._obs.append(observation)
            self._sensors.append(ObservationSensor(scenario.customer_id,self.episode_seed+414,**self.sensor_config))
        self.active=np.ones(self.config.size,dtype=bool)
        self.month=0; self.cumulative_loss=0.; self.months=[]; self.decisions=[]; self._diagnostics=[]
        self._done=False; self._start_month()
        return self._observation(),{}

    def _start_month(self):
        self.active_ids=np.flatnonzero(self.active)
        self.order=self.active_ids.copy()
        if self.config.order=='random': np.random.default_rng(np.random.SeedSequence([self.episode_seed,self.month,919])).shuffle(self.order)
        elif self.config.order=='reverse': self.order=self.order[::-1].copy()
        self.cursor=0; self.planned=np.full(self.config.size,2,dtype=int)
        self.limits=np.array([e.state.credit_limit for e in self._envs])
        self.balance=np.array([e.state.balance for e in self._envs])
        self.visible=np.array([self._sensors[i](self._obs[i]) if self.active[i] else self._obs[i] for i in range(self.config.size)])
        self.pd=self.visible[:,10].astype(float)
        self.start_state=self.state

    @property
    def state(self):
        ids=self.active_ids
        stress=bool(len(ids) and self.visible[ids[0],20])
        return aggregate(self.balance[ids],self.limits[ids],self.pd[ids],self.config,self.month,stress)

    def public_month(self):
        """Copies of public arrays only; no simulator object or mutable hidden reference."""
        return dict(ids=self.active_ids.copy(),observations=self.visible[self.active_ids].copy(),
            balance=self.balance[self.active_ids].copy(),limits=self.limits[self.active_ids].copy(),
            pd=self.pd[self.active_ids].copy(),state=self.state)

    def _observation(self):
        if self._done: return np.zeros(self.observation_space.shape,dtype=np.float32)
        s=self.state; n=self.config.size; cap=self.base.environment.max_limit
        bounded=lambda x: x/(1+x)
        features=[bounded(s.risk_budget_utilization),s.remaining_risk_budget/s.risk_budget,
            bounded(s.risk_shortfall/s.risk_budget),bounded(s.total_ead/s.exposure_budget),s.high_risk_share,
            s.average_pd,s.active_customers/n,self.cursor/max(len(self.order),1),bounded(s.total_ead/(n*cap)),
            bounded(s.risk_budget/(n*1000)),bounded(self.cumulative_loss/(n*cap)),float(self.config.dynamic_budget),
            bounded(self.config.safety_factor)]
        return np.r_[self.visible[self.order[self.cursor]],features].astype(np.float32)

    def step(self, action):
        if self._done: raise RuntimeError('Reset an inactive portfolio')
        if not self.action_space.contains(action): raise ValueError('Invalid action')
        i=int(self.order[self.cursor]); before=self.state; old=self.limits[i]
        limit,blocked=effective_limit(old,self._envs[i].state.months_delinquent,
            self.base.environment.action_multipliers[int(action)],self.base.environment,3)
        self.limits[i]=limit; candidate=self.state
        reason='severe_delinquency' if blocked else 'bounds' if abs(limit/old-self.base.environment.action_multipliers[int(action)])>1e-8 else 'accepted'
        rejected=self.config.mode=='hard' and not admissible(before,candidate,self.config)
        if rejected: self.limits[i]=old; effective=2; reason='portfolio_budget'
        else: effective=2 if blocked else int(action)
        self.planned[i]=effective; after=self.state
        record=dict(month=self.month,customer_index=i,requested_action=int(action),effective_action=effective,
            requested_change=self.base.environment.action_multipliers[int(action)]-1,effective_change=self.limits[i]/old-1,
            constraint_reason=reason,predicted_pd=float(self.pd[i]),income=self._envs[i].state.income,
            utilization=self._envs[i].state.utilization,score=self._envs[i].state.behavioral_score,
            remaining_budget_before=before.remaining_risk_budget,risk_shortfall_before=before.risk_shortfall,
            balance_before=float(self.balance[i]),limit_before=float(old),limit_after=float(self.limits[i]),
            proposed_delta_ead=candidate.total_ead-before.total_ead,
            proposed_delta_el=candidate.expected_loss-before.expected_loss,
            risk_budget=before.risk_budget,delta_ead=after.total_ead-before.total_ead,
            delta_el=after.expected_loss-before.expected_loss,
            delta_high_risk=after.high_risk_exposure-before.high_risk_exposure,
            hard_nonworsening=admissible(before,after,self.config))
        if self.record: self.decisions.append(record)
        self.cursor+=1; reward=0.; info={'decision':record}
        if self.cursor==len(self.order):
            allocation=self.state
            if self.diagnostics: self._diagnostics.append(self._true_risk_diagnostic())
            revenue=loss=funding=defaults=delinquent=0.
            for j in self.active_ids:
                obs,_,terminated,truncated,customer_info=self._envs[j].step(int(self.planned[j]))
                self._obs[j]=obs; c=customer_info['reward_components']
                revenue+=c['interest_income']+c['fee_income']; loss+=c['credit_loss']; funding+=c['funding_cost']
                defaults+=int(terminated); delinquent+=int(self._envs[j].state.months_delinquent>0)
                self.active[j]=not (terminated or truncated)
            value=revenue-loss-funding; self.cumulative_loss+=loss
            penalty=self.config.penalty*allocation.risk_shortfall if self.config.mode=='soft' else 0.
            reward=(value-penalty)/(1000*self.config.size)
            row=dict(**asdict(allocation),revenue=revenue,credit_loss=loss,funding_cost=funding,
                economic_value=value,penalty=penalty,objective=value-penalty,defaults=defaults,delinquent=delinquent,
                initial_el=self.start_state.expected_loss,initial_shortfall=self.start_state.risk_shortfall,
                el_violation=allocation.risk_shortfall>1e-7,
                any_violation=bool((constraint_vector(allocation,self.config)>1e-7).any()),
                pre_stress_ead=allocation.total_ead,
                macro_stress=float(self._envs[int(self.active_ids[0])].state.macro_state.credit_stress))
            # Row's macro describes the decision, not next month's observation.
            row['macro_stress']=float(self.visible[self.active_ids[0],9]/max(1-self.visible[self.active_ids[0],9],1e-8))
            self.months.append(row); info['month_result']=row
            self.month+=1; self._done=not self.active.any() or self.month>=self.base.environment.horizon
            if not self._done: self._start_month()
        # The stated objective ends at this finite, observed horizon. This is a
        # terminal MDP state, not an external TimeLimit requiring value bootstrap.
        return self._observation(),reward,bool(self._done),False,info

    def _true_risk_diagnostic(self):
        """Independent one-step MC integrates hidden behavior; never returned in step/info.

        Estimates actual next-month expected *realized loss*, not a 12-month PD.
        Uses independent hypothetical shocks and current macro, never realized futures.
        """
        from credit_rl.simulation.shocks import ShockPath
        draws=self.config.diagnostic_draws
        if not draws: return dict(month=self.month,true_expected_loss=np.nan,true_el_mc_se=np.nan)
        totals=np.zeros(draws)
        for i in self.active_ids:
            env=self._envs[i]
            path=ShockPath.generate(env.state.customer_id,self.episode_seed+170001+self.month*31,draws)
            for k,shock in enumerate(path.months):
                outcome=env._dgp.step(env.state,self.limits[i]/env.state.credit_limit,env._traits,shock,env.state.macro_state)
                totals[k]+=outcome.p_default_true*self.config.lgd*outcome.exposure
        return dict(month=self.month,true_expected_loss=float(totals.mean()),
                    true_el_mc_se=float(totals.std(ddof=1)/np.sqrt(draws)) if draws>1 else np.nan)

    def diagnostic_export(self):
        if not self.diagnostics: raise RuntimeError('Diagnostics disabled')
        return [dict(r) for r in self._diagnostics]

    def close(self):
        for env in getattr(self,'_envs',[]): env.close()
