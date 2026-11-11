"""One authorized endpoint-view B1 recheck; frozen evaluator, no physics."""
from pathlib import Path
import argparse
import sys
import time

START=time.perf_counter()
import compare_continuous_stop as base
from compare_continuous_stop import np,read,dump,sha,ROOT,ARCHIVE
from jerk_endpoint_view import replace_jerk_rows

PRIOR=base.DEST
DEST=ARCHIVE/'s3-b1-endpoint-recheck-v1'
# Isolated process configuration; no edits to the archived evaluator/source.
base.DEST=DEST
base.START=START
base.GLOBAL_LIMITS=dict(IK_calls=65000,QP_calls=200000,QP_internal_iterations=400000,
                       complete_started=1203,outer_calls=1)
base.PHASE_LIMITS=dict(preparepilot=60.,b1=840.,dense=120.)


class Budget(base.Budget):
    def check(self,**requests):
        super().check(**requests)
        if self.elapsed()>=1020.:raise base.LimitReached('1020-second total budget')
        if requests.get('qp') and self.stats['QP_internal_iterations']+80>400000:
            raise base.LimitReached('QP iteration reserve')


def guard():
    c=read(DEST/'contract.json')
    for path,h in c['source_sha256'].items():assert sha(Path(path))==h,path
    return c


def view_packet(prep,packet,g):
    view=replace_jerk_rows(packet,g,prep.vbasis)
    return view,dict(raw_min_g=packet['min_g'],solver_min_g=float(min(view)),
                     both_qualified=bool(packet['qualified'] and min(view)>=-1e-6))


class Problem(base.Problem):
    def value(self,x):
        packet,g=super().value(x)
        if 'solver_min_g' not in packet:
            view,flags=view_packet(self.prep,packet,g)
            path=Path(packet['path']).with_suffix('.solver.npz')
            np.savez_compressed(path,parameters=packet['parameters'],solver_constraints=view)
            packet.update(flags,solver_path=str(path),solver_sha256=sha(path))
            dump(Path(packet['path']).with_suffix('.json'),packet)
        return packet,g

    def constraint(self,x):
        packet,g=self.value(x)
        return replace_jerk_rows(packet,g,self.prep.vbasis)

    def derivatives(self,x):
        key=np.asarray(x).tobytes()
        if key==self.jac_key:return self.df,self.dg
        packet,raw=self.value(x); g=replace_jerk_rows(packet,raw,self.prep.vbasis)
        df=np.empty(53); dg=np.empty((len(g),53))
        for j in range(53):
            h=1e-6 if x[j]+1e-6<=self.upper[j] else -1e-6
            y=np.array(x,copy=True);y[j]+=h
            q,gg=self.value(y);vv=replace_jerk_rows(q,gg,self.prep.vbasis)
            df[j]=(q['f']-packet['f'])/h;dg[:,j]=(vv-g)/h
        assert np.all(np.isfinite(df)) and np.all(np.isfinite(dg))
        self.budget.stats['jacobians']+=1
        np.savez_compressed(DEST/f'B1_jac_{self.budget.stats["jacobians"]:03d}.npz',x=x,df=df,dg=dg)
        self.jac_key,self.df,self.dg=key,df,dg
        return df,dg

    def callback(self,x):
        packet,_=self.value(x)
        self.iterations.append(dict(f=packet['f'],min_g=packet['min_g'],
            solver_min_g=packet['solver_min_g'],both_qualified=packet['both_qualified'],parameters=packet['parameters']))


def prepare():
    assert not DEST.exists(),'Preserve existing run'
    old=read(PRIOR/'contract.json')
    for p,h in old['source_sha256'].items():assert sha(Path(p))==h,p
    B0=read(PRIOR/'B0_dense.json');assert B0['passed']
    DEST.mkdir()
    files=[Path(__file__),Path(__file__).with_name('jerk_endpoint_view.py'),
        Path(__file__).with_name('verify_jerk_endpoint_view.py'),PRIOR/'contract.json',
        PRIOR/'B0_dense.json',Path(B0['packet']['path']),PRIOR/'p2_gate.json',
        ARCHIVE/'s3-jerk-endpoint-view-v1/verification.json']
    pilot_old=read(PRIOR/'pilot.json')['evaluations'][4]
    files += [Path(pilot_old['path']),Path(pilot_old['path']).with_suffix('.json')]
    hashes=dict(old['source_sha256']);hashes.update({str(p.resolve()):sha(p) for p in files})
    c=dict(status='authorized_before_new_evaluation',authorization='User continued the proposed single B1 endpoint-view recheck, 840s solve, <=17min total, zero physics.',
        deadline_s=base.D,common_H_s=base.HORIZON,variables=53,start=B0['packet']['parameters'],
        B0_reference=str(PRIOR/'B0_dense.json'),pilot_nonzero=pilot_old,
        only_solver_change='Replace two aggregated jerk rows by fixed-label signed endpoint rows; retain original raw admission.',
        solver_rows=5388,raw_rows=5366,method='SLSQP',maxiter=25,ftol=1e-8,step=1e-6,
        maximum_complete_evaluations=1203,maximum_solver_evaluations=1200,maximum_pilot_evaluations=2,
        maximum_dense_audits=1,maximum_dense_nodes=900,limits=base.GLOBAL_LIMITS,
        phase_limits_s=base.PHASE_LIMITS,maximum_active_s=1020.,
        seed=0,new_physics_runs=0,new_physics_steps=0,new_dependencies=0,
        failure_policy='One solve, no restart. Normal convergence and both constraint views precede one dense audit. Preserve failures/partial output. Exclusive action markers prevent reruns.',
        timing='Cooperative active-time checks including imports/preparation/output; audit/gate waits excluded, stop/check tails disclosed.',
        scope='Same nominal kinematic task as prior; no drive/grasp dynamics, physical or realtime qualification.',source_sha256=hashes)
    dump(DEST/'contract.json',c);guard()
    state=dict(stage='prepared',failure=None,solves=[],pilot=[],B0=B0)
    budget=Budget(state,'preparepilot');base.Preparation(budget);budget.check()
    base.save_progress(state,budget)
    print({'prepared':True,'sources':len(hashes),'active_s':budget.elapsed()})


def pilot(state,budget,prep,c):
    assert state['stage']=='prepared'
    packets=[]
    for index,(x,expected) in enumerate([(c['start'],state['B0']['packet']),
                                       (c['pilot_nonzero']['parameters'],c['pilot_nonzero'])]):
        packet,g=base.evaluate(prep,budget,x,'P1_'+str(index))
        view,flags=view_packet(prep,packet,g)
        with np.load(expected['path']) as z:oldf=float(z['objective_weights'] @ np.sum(z['objective_acceleration']**2,axis=1))
        assert abs(oldf-packet['f'])<1e-12
        assert abs(min(view)-min(g))<1e-10
        assert flags['both_qualified']
        np.savez_compressed(DEST/f'pilot_view_{index}.npz',solver_constraints=view)
        packets.append(dict(packet=packet,**flags));state['pilot'].append(packet['number'])
        budget.check()
    dump(DEST/'pilot.json',dict(passed=True,cases=packets,new_physics_steps=0))
    state['stage']='pilot_done'


def solve(state,budget,prep,c):
    assert state['stage']=='pilot_done'
    gate=read(DEST/'p1_gate.json')
    assert gate['status']=='PASS' and gate['contract_sha256']==sha(DEST/'contract.json')
    assert budget.stats['outer_calls']==0
    budget.check(outer=True);budget.stats['outer_calls']+=1
    problem=Problem(prep,budget,c['start'],'B1',1200)
    result=None;final=None;first=None;failure=None;critical=None
    try:
        first,_=problem.value(np.array(c['start']))
        result=base.minimize(problem.fun,c['start'],method='SLSQP',jac=lambda x:problem.derivatives(x)[0],
            bounds=list(zip(problem.lower,problem.upper)),
            constraints=dict(type='ineq',fun=problem.constraint,jac=lambda x:problem.derivatives(x)[1]),
            callback=problem.callback,options=dict(maxiter=25,ftol=1e-8))
        final,_=problem.value(result.x)
    except Exception as e:
        failure=type(e).__name__+': '+str(e)
        if not isinstance(e,base.LimitReached) or str(e)!='B1 evaluation cap':critical=e
    record=dict(success=bool(result is not None and result.success),failure=failure,
        message=None if result is None else str(result.message),initial=first,final=final,
        best_raw_coarse=problem.best,evaluations=problem.records,iterations=problem.iterations)
    dump(DEST/'B1_solve_0.json',record);state['solves']=['B1_solve_0.json']
    if critical is not None:raise critical
    if record['success'] and final['both_qualified']:
        report=base.dense_audit(prep,budget,final,'B1')
        with np.load(report['packet']['path']) as z:g=z['constraints']
        vv,flags=view_packet(prep,report['packet'],g)
        report['endpoint_view']=flags;assert flags['both_qualified']
        dump(DEST/'B1_dense.json',report);state['B1']=report
        decrease=state['B0']['packet']['f']-report['packet']['f']
        threshold=max(1e-8,10*max(state['B0']['quadrature_12_24_error'],report['quadrature_12_24_error']))
        state['comparison']=dict(objective_decrease=decrease,numerical_threshold=threshold,
            decrease_resolved=bool(decrease>threshold),relative_RMS_change=np.sqrt(report['packet']['f']/state['B0']['packet']['f'])-1)
    else:state['stop_reason']='No normally converged reference satisfying both views; no dense or physical admission'
    state['stage']='complete'


def execute(action):
    c=guard();state=read(DEST/'progress.json');assert state['failure'] is None
    with (DEST/(action+'_started.flag')).open('x') as f:f.write('one authorized action\n')
    budget=Budget(state,'preparepilot' if action=='pilot' else 'b1')
    try:
        prep=base.Preparation(budget)
        (pilot if action=='pilot' else solve)(state,budget,prep,c)
    except Exception as e:state.update(stage='stopped',failure=type(e).__name__+': '+str(e))
    base.save_progress(state,budget)
    if state['stage'] in ['complete','stopped']:
        dump(DEST/'result.json',dict(**state,contract_sha256=sha(DEST/'contract.json'),
            comparison_completed='comparison' in state,new_physics_runs=0,new_physics_steps=0))
    guard();print({k:state.get(k) for k in ['stage','failure','active_s','stats','stop_reason']})
    return state['failure'] is None


def verify():
    guard();state=read(DEST/'progress.json')
    # The reused raw-data verifier never calls an optimizer or new IK.
    base.verify()
    with np.load(base.ARCHIVE/'s3-structured-conditioning/alternative_velocity_basis.npz') as z:
        vb=base.BSpline(z['knots'],z['Z'],3)
    count=0
    for path in sorted(DEST.glob('eval_*.json')):
        if path.stem.endswith(('_started','_failed')):continue
        packet=read(path)
        with np.load(packet['path']) as z:raw=z['constraints']
        view=replace_jerk_rows(packet,raw,vb)
        assert abs(min(view)-min(raw))<1e-9
        if 'solver_path' in packet:
            assert sha(Path(packet['solver_path']))==packet['solver_sha256']
            with np.load(packet['solver_path']) as z:assert np.array_equal(view,z['solver_constraints'])
            assert float(min(view))==packet['solver_min_g']
        count+=1
    assert state['active_s']<1021. and state['stats']['outer_calls']<=1
    print(dict(endpoint_views_verified=count,new_IK=0,new_QP=0,new_physics_steps=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','pilot','solve','verify'])
    action=parser.parse_args().action
    if action=='prepare':prepare()
    elif action=='verify':verify()
    else:sys.exit(0 if execute(action) else 1)
