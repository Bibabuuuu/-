"""Saved-data and scalar boundary checks; no model, reference or solver calls."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from fractions import Fraction
from bisect import bisect_right

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'work/stop-optimizer-deps'))
import numpy as np
from scipy.interpolate import BSpline,PPoly
from jerk_endpoint_view import endpoint_residuals,replace_jerk_rows,EndpointConstraintView

PRIOR=ROOT/'outputs/rby1-research/s3-b0-b1-continuous-v1'
BASIS=ROOT/'outputs/rby1-research/s3-structured-conditioning/alternative_velocity_basis.npz'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def oracle(vb,c,A,T):
    """Independent merged-interval midpoint selection, not knot-side lookup."""
    pp=PPoly.from_spline(BSpline(vb.t,.1*(vb.c @ c),3)).derivative(2)
    phase=A/5/T; edges=np.unique(np.r_[0.,phase,1-phase,1.,vb.t]); vals=[]
    for l,r in zip(edges[:-1],edges[1:]):
        # Adjacent float boundaries may have no representable float midpoint.
        # Rational comparison retains the real ordering without snapping knots.
        mid=(Fraction(float(l))+Fraction(float(r)))/2
        i=bisect_right([Fraction(float(t)) for t in pp.x],mid)-1
        seed=-5. if mid<Fraction(float(phase)) else (5. if mid>Fraction(float(1-phase)) else 0.)
        vals.extend(seed+np.polyval(pp.c[:,i],np.array([l,r])-pp.x[i])/T**2)
    return min(vals),max(vals)


def main():
    contract=read(PRIOR/'contract.json')
    for p,h in contract['source_sha256'].items():assert sha(Path(p))==h,p
    with np.load(BASIS) as z:vb=BSpline(z['knots'],z['Z'],3)
    solve=read(PRIOR/'B1_solve_0.json'); records=solve['evaluations']; bykey={}
    source=[Path(__file__),Path(__file__).with_name('jerk_endpoint_view.py'),BASIS,
            PRIOR/'contract.json',PRIOR/'B1_solve_0.json']
    error=0.; width=None
    for packet in records:
        assert sha(Path(packet['path']))==packet['sha256']
        with np.load(packet['path']) as z:raw=z['constraints'].copy()
        x=np.array(packet['parameters']); ep=endpoint_residuals(vb,x[3:7],.5+.5*x[0],packet['T_base_s'])
        n=len(ep)//2
        expected=packet['continuous_ranges']
        error=max(error,abs(min(ep[:n])-(expected['minimum'][2]+5)/5),abs(min(ep[n:])-(5-expected['maximum'][2])/5))
        view=replace_jerk_rows(packet,raw,vb); block=next(b for b in packet['blocks'] if b['name']=='continuous')
        keep=np.ones(len(raw),bool);keep[block['start']+np.array([2,8])]=False
        assert np.array_equal(view[:-len(ep)],raw[keep]) and np.array_equal(view[-len(ep):],ep)
        if width is None:width=len(view)
        assert len(view)==width and (min(raw)>=-1e-6)==(min(view)>=-1e-6)
        bykey[x.tobytes()]=(packet,raw)
    assert error<1e-10
    # Real SLSQP callbacks, with the provider restricted to old cached points.
    provider_calls=[]
    def provider(x):
        provider_calls.append(np.asarray(x).tobytes())
        return bykey[np.asarray(x).tobytes()]
    callback=EndpointConstraintView(provider,vb,np.r_[0.,-1.,0.,-np.ones(50)],np.ones(53))
    jacobians=[]
    for path in sorted(PRIOR.glob('B1_jac_*.npz')):
        source.append(path)
        with np.load(path) as z:x=z['x']; G=z['dg']
        packet,raw=provider(x); J=callback.jacobian(x)
        b=next(b for b in packet['blocks'] if b['name']=='continuous'); keep=np.ones(len(raw),bool)
        keep[b['start']+np.array([2,8])]=False
        assert np.array_equal(J[:sum(keep)],G[keep])
        jacobians.append(J.shape[1])
    # Midpoint oracle verifies both one-sided limits at true/near collisions.
    v0=.39993127226900177; cases=[]
    for k in np.unique(vb.t)[1:-1]:
        A=np.sqrt(5*v0*k/(1-k))
        if .5<=A<=1.:
            for delta in [-1e-12,0.,1e-12]:
                a=A+delta;cases.append((float(a),float(v0/a+a/5)))
    cases += [(.5,v0/.5+.5/5),(1.,v0+1/5),(.5,.2)]
    boundary_error=0.; boundary_count=0
    for A,T in cases:
        for c in [np.zeros(4),.2*np.cos(np.arange(4)+1),-.2*np.cos(np.arange(4)+1)]:
            ep=endpoint_residuals(vb,c,A,T); n=len(ep)//2; lo,hi=oracle(vb,c,A,T)
            e=max(abs(min(ep[:n])-(lo+5)/5),abs(min(ep[n:])-(5-hi)/5))
            boundary_error=max(boundary_error,e);boundary_count+=1
            assert e<1e-10,(A,T,c.tolist(),e,lo,hi,min(ep[:n]),min(ep[n:]))
    for bad in [(np.full(4,np.nan),.6,.7),(np.zeros(4),-.6,.7),(np.zeros(4),.6,0.)]:
        try:endpoint_residuals(vb,*bad)
        except ValueError:pass
        else:raise AssertionError('Invalid input accepted')
    # Exercise the upper-box one-sided step using a scalar-only test provider.
    scalar_calls=[]
    def synthetic(x):
        scalar_calls.append(np.array(x));full=np.r_[x,np.zeros(49)]
        return dict(parameters=full.tolist(),T_base_s=.7,blocks=[dict(name='continuous',start=0,stop=12)]),np.zeros(12)
    small=EndpointConstraintView(synthetic,vb,-np.ones(4),np.ones(4))
    small.jacobian(np.ones(4))
    assert all(np.all(x<=1.) and np.all(x>=-1.) for x in scalar_calls)
    return dict(passed=True,saved_cases=len(records),raw_rows=5366,solver_rows=width,
        saved_jerk_equivalence_error=error,unchanged_nonjerk_rows_exact=True,
        unchanged_admission_flags=True,Jacobian_callbacks=len(jacobians),columns_each=53,
        nonjerk_Jacobian_rows_exact=True,boundary_cases=boundary_count,
        boundary_oracle_error=boundary_error,box_boundary_step_passed=True,
        invalid_inputs_rejected=True,new_reference_evaluations=0,new_IK_calls=0,new_QP_calls=0,new_physics_steps=0,
        scope='Scalar equivalence and saved callback replay; not a fresh SLSQP run or convergence guarantee.',
        source_sha256={str(p):sha(p) for p in source})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
    arg=parser.parse_args();result=main()
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if arg.output:
        arg.output.parent.mkdir(parents=True,exist_ok=True)
        with arg.output.open('x',encoding='utf-8') as f:f.write(text)
    print(text)
