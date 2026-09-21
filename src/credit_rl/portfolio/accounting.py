"""Current-information risk stocks, distinct from realized monthly loss flows."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PortfolioConfig:
    size: int = 12
    ccf: float = .5
    lgd: float = .55
    pd_horizon: int = 12
    el_budget_per_customer: float = 150.
    ead_budget_per_customer: float = 8000.
    high_risk_pd: float = .6
    high_risk_share: float = .8
    stress_budget_multiplier: float = .75
    dynamic_budget: bool = True
    safety_factor: float = 1.
    mode: str = 'hard'
    penalty: float = 0.
    order: str = 'random'
    diagnostic_draws: int = 16

    def __post_init__(self):
        if self.size < 1 or self.pd_horizon < 1 or self.diagnostic_draws < 0:
            raise ValueError('Invalid portfolio dimensions')
        for name in ('ccf','lgd','high_risk_pd','high_risk_share','stress_budget_multiplier'):
            if not np.isfinite(getattr(self,name)) or not 0 <= getattr(self,name) <= 1:
                raise ValueError(name)
        for name in ('el_budget_per_customer','ead_budget_per_customer','safety_factor'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<=0: raise ValueError(name)
        if self.mode not in ('hard','soft','unconstrained') or self.order not in ('random','fixed','reverse'):
            raise ValueError('Unknown constraint mode/order')
        if not np.isfinite(self.penalty) or self.penalty<0: raise ValueError('Invalid penalty')
        if self.stress_budget_multiplier<=0: raise ValueError('Stress budget must stay positive')


def ead(balance, limit, ccf):
    return np.asarray(balance)+ccf*np.maximum(np.asarray(limit)-np.asarray(balance),0.)


def monthly_pd(pd, horizon=12):
    """Flat-hazard approximation, not a calibrated monthly default forecast."""
    with np.errstate(divide='ignore'):
        return -np.expm1(np.log1p(-np.asarray(pd))/horizon)


@dataclass(frozen=True)
class PortfolioState:
    month: int
    active_customers: int
    total_credit_limit: float
    total_balance: float
    total_ead: float
    expected_loss: float
    raw_expected_loss: float
    average_pd: float
    high_risk_exposure: float
    risk_budget: float
    exposure_budget: float
    remaining_risk_budget: float
    risk_shortfall: float
    risk_budget_utilization: float
    high_risk_share: float
    top_decile_el_share: float


def aggregate(balance, limits, pd, config, month=0, stress=False):
    balance,limits,pd=map(lambda x:np.asarray(x,dtype=float),(balance,limits,pd))
    if not (balance.shape==limits.shape==pd.shape) or not all(np.isfinite(x).all() for x in (balance,limits,pd)) or (balance<0).any() or (limits<0).any() or ((pd<0)|(pd>1)).any():
        raise ValueError('Invalid public portfolio arrays')
    exposure=ead(balance,limits,config.ccf)
    buffered=np.minimum(1.,pd*config.safety_factor)
    loss=monthly_pd(buffered,config.pd_horizon)*config.lgd*exposure
    raw=monthly_pd(pd,config.pd_horizon)*config.lgd*exposure
    budget=config.size*config.el_budget_per_customer*(config.stress_budget_multiplier if stress and config.dynamic_budget else 1.)
    if budget<=0: raise ValueError('Risk budget must remain positive')
    total=float(exposure.sum()); expected=float(loss.sum())
    high=float(exposure[pd>config.high_risk_pd].sum())
    count=max(1,int(np.ceil(len(loss)*.1)))
    return PortfolioState(month,len(balance),float(limits.sum()),float(balance.sum()),total,expected,float(raw.sum()),
        float(pd.mean()) if len(pd) else 0.,high,budget,config.size*config.ead_budget_per_customer,
        max(0.,budget-expected),max(0.,expected-budget),expected/budget,high/total if total else 0.,
        float(np.sort(loss)[-count:].sum()/expected) if expected else 0.)


def constraint_vector(state, config):
    # Linear high-risk excess; denominator changes are accounted for.
    return np.array([state.expected_loss-state.risk_budget,state.total_ead-state.exposure_budget,
                     state.high_risk_exposure-config.high_risk_share*state.total_ead])


def admissible(before, after, config, tolerance=1e-7):
    """No new breach; inherited/exogenous deficits may not increase per constraint."""
    return bool(np.all(constraint_vector(after,config)<=np.maximum(0.,constraint_vector(before,config))+tolerance))
