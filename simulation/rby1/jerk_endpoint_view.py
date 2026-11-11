"""Equivalent solver view for cubic-velocity jerk; raw audit rows stay intact."""
import numpy as np
from scipy.interpolate import BSpline, PPoly


def endpoint_residuals(velocity_basis, coefficients, acceleration, duration):
    """Fixed-label, two-sided endpoint bounds for a J=5 Ruckig seed.

    On every merged interval jerk is affine, so bounding both ends bounds the
    entire interval. Coincident knots keep separate labels. At domain ends
    both labels use the interior limit; no constraint before/after the plan.
    """
    coefficients=np.asarray(coefficients,dtype=float)
    if velocity_basis.k!=3 or coefficients.shape!=(velocity_basis.c.shape[1],):
        raise ValueError('Expected a cubic velocity basis and matching coefficients')
    if not np.all(np.isfinite(coefficients)) or not np.isfinite(acceleration) or not np.isfinite(duration):
        raise ValueError('Nonfinite jerk-view input')
    if acceleration<=0 or duration<=0:
        raise ValueError('Positive acceleration/duration required')
    phase=acceleration/(5*duration)
    if not 0<phase<=.5:
        raise ValueError('Expected trapezoidal or triangular Ruckig braking phases')
    knots=np.unique(velocity_basis.t)
    if knots[0]!=0 or knots[-1]!=1:
        raise ValueError('Expected a basis normalized to [0,1]')
    pp=PPoly.from_spline(BSpline(velocity_basis.t,.1*(velocity_basis.c @ coefficients),3)).derivative(2)
    values=[]
    for u in np.r_[0.,phase,1-phase,1.,knots[1:-1]]:
        for requested in ['left','right']:
            side='right' if u==0 else ('left' if u==1 else requested)
            index=int(np.clip(np.searchsorted(pp.x,u,side=side)-1,0,len(pp.x)-2))
            correction=np.polyval(pp.c[:,index],u-pp.x[index])/duration**2
            initial=u<phase or (u==phase and side=='left')
            final=u>1-phase or (u==1-phase and side=='right')
            values.append((-5. if initial else (5. if final else 0.))+correction)
    values=np.asarray(values)
    return np.r_[(values+5)/5,(5-values)/5]


def replace_jerk_rows(packet, raw_constraints, velocity_basis):
    """Drop only two aggregated jerk rows; append equivalent endpoint rows."""
    raw=np.asarray(raw_constraints,dtype=float)
    block=next(b for b in packet['blocks'] if b['name']=='continuous')
    if block['stop']-block['start']!=12 or raw.ndim!=1 or not np.all(np.isfinite(raw)):
        raise ValueError('Unexpected raw continuous-constraint layout')
    x=np.asarray(packet['parameters'],dtype=float)
    if x.shape!=(53,): raise ValueError('Expected expanded 53-parameter packet')
    endpoints=endpoint_residuals(velocity_basis,x[3:7],.5+.5*x[0],packet['T_base_s'])
    keep=np.ones(len(raw),dtype=bool)
    keep[block['start']+np.array([2,8])]=False
    return np.r_[raw[keep],endpoints]


class EndpointConstraintView:
    """SLSQP-compatible callbacks over an existing cached f/g provider.

    The provider owns budgets, logging, original admission and parameter
    expansion. This class does not run an optimizer or replace its raw audit.
    """
    def __init__(self, value_provider, velocity_basis, lower, upper):
        self.value=value_provider
        self.basis=velocity_basis
        self.lower=np.asarray(lower,dtype=float)
        self.upper=np.asarray(upper,dtype=float)
        if self.lower.shape!=self.upper.shape or np.any(self.lower>=self.upper):
            raise ValueError('Expected matching nonempty parameter bounds')

    def constraints(self,x):
        packet,g=self.value(x)
        return replace_jerk_rows(packet,g,self.basis)

    def jacobian(self,x):
        x=np.asarray(x,dtype=float)
        if x.shape!=self.lower.shape or not np.all(np.isfinite(x)) or np.any(x<self.lower) or np.any(x>self.upper):
            raise ValueError('Parameters outside bounds')
        base=self.constraints(x); columns=[]
        for j in range(len(x)):
            step=1e-6 if x[j]+1e-6<=self.upper[j] else -1e-6
            if x[j]+step<self.lower[j]: raise ValueError('Bounds too narrow for declared difference step')
            trial=x.copy(); trial[j]+=step
            columns.append((self.constraints(trial)-base)/step)
        return np.column_stack(columns)
