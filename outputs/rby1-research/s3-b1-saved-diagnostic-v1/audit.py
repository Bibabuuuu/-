"""Postprocess frozen parameters/Jacobians only; no reference factory or solver."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'work/stop-optimizer-deps'))
import numpy as np
from scipy.interpolate import BSpline,PPoly

SOURCE=ROOT/'outputs/rby1-research/s3-b0-b1-continuous-v1'
BASIS=ROOT/'outputs/rby1-research/s3-structured-conditioning/alternative_velocity_basis.npz'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def run():
    contract=read(SOURCE/'contract.json')
    for name,h in contract['source_sha256'].items(): assert sha(Path(name))==h,name
    solve=read(SOURCE/'B1_solve_0.json'); records=solve['evaluations']
    bykey={np.array(q['parameters']).tobytes():q for q in records}
    with np.load(BASIS) as z: vb=BSpline(z['knots'],z['Z'],3)
    # Cubic velocity => linear jerk in each merged piece. Both one-sided
    # endpoint bounds are sufficient, with no change to the +/-5 limit.
    knots=np.unique(vb.t); endpoint_cache={}
    def endpoints(q):
        if q['number'] in endpoint_cache:return endpoint_cache[q['number']]
        x=np.array(q['parameters']); T=q['T_base_s']; A=.5+.5*x[0]
        phase=A/5/T
        pp=PPoly.from_spline(BSpline(vb.t,.1*(vb.c @ x[3:7]),3)).derivative(2)
        values=[]
        for u in np.r_[0.,phase,1-phase,1.,knots[1:-1]]:
            for side in ['left','right']:
                side='right' if u==0 else ('left' if u==1 else side)
                i=int(np.clip(np.searchsorted(pp.x,u,side=side)-1,0,len(pp.x)-2))
                correction=np.polyval(pp.c[:,i],u-pp.x[i])/T**2
                initial=(u<phase or (u==phase and side=='left'))
                final=(u>1-phase or (u==1-phase and side=='right'))
                seed=-5. if initial else (5. if final else 0.)
                values.append(seed+correction)
        values=np.array(values)
        result=np.r_[(values+5)/5,(5-values)/5]
        expected=q['continuous_ranges']
        assert abs(min(values)-expected['minimum'][2])<1e-10
        assert abs(max(values)-expected['maximum'][2])<1e-10
        endpoint_cache[q['number']]=result
        return result
    for q in records:endpoints(q)
    derivative_numbers=set(); comparisons=[]; max_reproduction=0.; inputs=[SOURCE/'contract.json',SOURCE/'B1_solve_0.json',BASIS,Path(__file__)]
    for k,path in enumerate(sorted(SOURCE.glob('B1_jac_*.npz'))):
        inputs.append(path)
        with np.load(path) as z:x=z['x']; G=z['dg']; df=z['df']
        base=bykey[x.tobytes()]; b=endpoints(base)
        E=np.empty((len(b),53))
        with np.load(base['path']) as z:g=z['constraints']
        for j in range(53):
            step=1e-6 if x[j]+1e-6<=1 else -1e-6
            y=x.copy();y[j]+=step; q=bykey[y.tobytes()]
            derivative_numbers.add(q['number']); E[:,j]=(endpoints(q)-b)/step
            with np.load(q['path']) as z:gg=z['constraints']
            max_reproduction=max(max_reproduction,float(abs((gg-g)/step-G[:,j]).max()),abs((q['f']-base['f'])/step-df[j]))
        nxt=bykey[np.array(solve['iterations'][k]['parameters']).tobytes()]
        d=np.array(nxt['parameters'])-x
        aggregate_prediction=(g+G @ d)[[5355,5361]]
        actual=np.array([(nxt['continuous_ranges']['minimum'][2]+5)/5,
                         (5-nxt['continuous_ranges']['maximum'][2])/5])
        endpoint_prediction=b+E @ d
        # Signed rows keep every endpoint rather than taking an envelope
        # before finite differences. All vectors refer to stored points.
        aggregate_endpoint_prediction=np.array([min(endpoint_prediction[:len(b)//2]),min(endpoint_prediction[len(b)//2:])])
        comparisons.append(dict(iteration=k+1,step_norm=float(np.linalg.norm(d)),
            aggregate_predicted=aggregate_prediction.tolist(),actual=actual.tolist(),
            endpoint_predicted=aggregate_endpoint_prediction.tolist(),
            aggregate_prediction_error=float(abs(aggregate_prediction-actual).max()),
            endpoint_prediction_error=float(abs(aggregate_endpoint_prediction-actual).max())))
    # Include the already-saved columns of the unfinished final Jacobian.
    x=np.array(solve['iterations'][-1]['parameters'])
    for j in range(53):
        y=x.copy(); y[j]+=1e-6 if x[j]+1e-6<=1 else -1e-6
        if y.tobytes() in bykey:derivative_numbers.add(bykey[y.tobytes()]['number'])
    assert max_reproduction<1e-12
    times=[q['wall_s'] for q in records]
    return dict(scope='Frozen B1 data algebra only; endpoint constraints are an offline witness, not an implemented solver change.',
        complete_B1_evaluations=len(records),derivative_evaluations=len(derivative_numbers),
        derivative_evaluation_fraction=len(derivative_numbers)/len(records),
        derivative_recorded_wall_s=sum(q['wall_s'] for q in records if q['number'] in derivative_numbers),
        all_complete_evaluation_recorded_wall_s=sum(times),evaluation_wall_quantiles_s=np.quantile(times,[0,.5,.9,1]).tolist(),
        parameter_Jacobian_reproduction_error=max_reproduction,
        jerk_endpoint_equivalence_saved_cases=len(endpoint_cache),
        iteration_predictions=comparisons,
        notes=['Global min/max envelopes are nonsmooth when the active endpoint changes.',
               'Finite-difference arithmetic is reproducible; this does not prove it is a valid smooth gradient.',
               'A single saved step size cannot quantify roundoff versus truncation error.',
               'Endpoint equivalence here is checked on saved vectors, not a new optimizer or dynamics qualification.'],
        new_complete_reference_evaluations=0,new_IK_calls=0,new_QP_calls=0,new_physics_steps=0,
        source_sha256={str(p):sha(p) for p in inputs})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
    args=parser.parse_args(); result=run(); content=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        with args.output.open('x',encoding='utf-8') as f:f.write(content)
    else:print(content)
