"""One bounded mature SLSQP joint reference solve. No physical simulation."""
from pathlib import Path
from collections import OrderedDict
import argparse,hashlib,json,sys,time
ROOT=Path(r'C:\Users\14737\Documents\Codex\2026-10-20\new-chat')
PRIOR=ROOT/'outputs/rby1-research/s3-reference-jacobian-diagnostic'
CACHE=ROOT/'outputs/rby1-research/s3-cached-reference-evaluation'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(PRIOR))
from jacobian_probe import (Preparation,CachedAdapter,fixed_signature,assemble,
    guard as previous_guard,scalar_check,continuous_velocity,angular_check,read,sha,dump,np)
from whole_body_stop_reference import validate
from scipy.optimize import minimize
import scipy

class LimitReached(RuntimeError):pass
class Limits:
    def __init__(self,c):
        self.c=c;self.started=time.perf_counter();self.phase='prepare'
        self.stats=dict(IK_calls=0,QP_calls=0,QP_internal_iterations=0,unique_targets=0,line_search_trials=0)
    def elapsed(self):return time.perf_counter()-self.started
    def check(self,ik=False,qp=False):
        s=self.stats
        if self.elapsed()>=self.c['maximum_timed_run_wall_s']:raise LimitReached('180-second total run limit')
        if self.phase=='solve' and self.elapsed()>=self.c['optimizer_wall_cutoff_s']:raise LimitReached('160-second optimizer cutoff')
        if ik and s['IK_calls']>=self.c['maximum_IK_calls']:raise LimitReached('30000-IK total limit')
        if qp and (s['QP_calls']>=self.c['maximum_QP_calls'] or s['QP_internal_iterations']+80>self.c['maximum_QP_internal_iterations']):
            raise LimitReached('QP call/internal-iteration limit with 80-step reserve')

def solver_view(g,jerk,cv,angular,fixed):
    raw=np.r_[g[:4680],1-jerk/5,1+jerk/5,g[4725:]]
    assert len(raw)==4905
    keep=np.ones(4905,dtype=bool);keep[fixed]=False
    value=np.r_[raw[keep],cv,angular];assert len(value)==4928
    return value

def guard():
    c=read(HERE/'contract.json');previous_guard()
    for p,h in c['source_sha256'].items():assert sha(p)==h,p
    return c

def prepare():
    assert not (HERE/'contract.json').exists()
    old=previous_guard();proposal=read(PRIOR/'first_joint_solve_proposal.json')
    for p,h in proposal['source_sha256'].items():assert sha(p)==h,p
    with np.load(PRIOR/'jacobian.npz') as d:G=d['constraints'].copy()
    with np.load(PRIOR/'signed_jerk_review.npz') as d:expected=d['solver_draft_values'].copy()
    error=0.
    for i in range(52):
        with np.load(PRIOR/('case_%02d.npz'%i)) as d:j=d['jerk'].copy()
        cv=expected[i,-27:-3];ang=expected[i,-3:]
        value=solver_view(G[i],j,cv,ang,old['fixed_endpoint_rows'])
        error=max(error,float(abs(value-expected[i]).max()))
    assert error<1e-12
    preflight=dict(passed=True,old_probe_vectors_reused=52,new_IK_calls=0,new_outer_optimizer_calls=0,
        solver_view_max_error=error,old_rows_preserved=4860,solver_rows=4928,
        old_qualification_unchanged=True,scipy_version=scipy.__version__)
    dump(HERE/'preflight.json',preflight)
    b=proposal['proposed_budget']
    c={k:old[k] for k in ['times_s','objective_times_s','objective_normalized_integration_weights',
        'objective_scale','objective_scale_source','common_motion_window_s','T_s','meta','fixed_endpoint_rows']}
    c.update(date='2026-11-02',status='已授权实施',
        authorization='User explicitly selected 按推荐范围执行 for the one full-50 SLSQP stage.',
        free_variables=50,start_parameters=proposal['start_parameters'],optimizer_settings=proposal['optimizer_settings'],
        maximum_outer_optimizer_calls=1,maximum_complete_evaluations=600,optimizer_evaluation_cutoff=598,
        maximum_IK_calls=b['maximum_IK_calls'],maximum_QP_calls=b['maximum_QP_calls'],
        maximum_QP_internal_iterations=b['maximum_QP_internal_iterations'],
        maximum_timed_run_wall_s=b['maximum_run_wall_s'],optimizer_wall_cutoff_s=b['optimizer_wall_cutoff_s'],
        maximum_local_steps_per_IK=30,maximum_final_dense_audits=1,maximum_final_dense_IK_nodes=1400,
        final_dense_times_s=proposal['proposed_final_dense_times_s'],
        derivative_centers_s=read(ROOT/'outputs/rby1-research/s3-continuous-reference-qualification/contract.json')['derivative_centers_s'],
        difference_steps_s=[.001,.0005],objective_improvement_numerical_tolerance=1e-10,
        acceptance_min_scaled=-1e-6,numeric_domain='Finite infeasible values allowed; failed mapping stops. Old qualification and all 4860 audit rows retained.',
        final_gate='Normal solver success and objective decrease beyond 1e-10 precede at most one dense reference audit; no physical admission.',
        source_sha256={str(p):sha(p) for p in [
          Path(__file__),HERE/'preflight.json',PRIOR/'first_joint_solve_proposal.json',
          PRIOR/'jacobian_probe.py',PRIOR/'jacobian.npz',PRIOR/'signed_jerk_review.npz',
          CACHE/'cached_evaluation.py']})
    assert len(c['final_dense_times_s'])<=1400
    dump(HERE/'contract.json',c);guard()
    print(json.dumps({'prepared':True,'authorized':True,'preflight_error':error,'variables':50,
        'solver_rows':4928,'maximum_calls':1,'maxiter':40,'max_full_eval':600,'wall_s':180,
        'dense_nodes':len(c['final_dense_times_s'])},ensure_ascii=False))

def core(prep,budget,c,z,times,path):
    p=CachedAdapter(z,budget,0,prep);p.external.source_paths+=(Path(__file__),HERE/'contract.json')
    p.external.config.update(physical_execution_admitted=False)
    times=np.array(times);n=len(times);q=[];v=[];a=[];closures=[];gaps=[];pen=[];bases=[];objects=[];lift=[];mom=[];te=[];ce=[]
    try:
        for t in times:
            qq,vv,aa=p.joint(t);item,tt,cc=p.observe(t,qq,vv,aa,True)
            if max(tt,cc)>1+1e-6:raise RuntimeError('Implicit reference mapping violates old target/closure tolerance')
            q.append(qq);v.append(vv);a.append(aa);closures.append(item['closure']);gaps.append(item['gap']);pen.append(item['penetration'])
            bases.append(p.base_at(t));objects.append(p.object_at(t));lift.append(item['payload'][2]);mom.append(item['moments']);te.append(tt);ce.append(cc)
            budget.check()
        q,v,a,closures,gaps,pen=[np.array(x) for x in [q,v,a,closures,gaps,pen]]
        yaw=np.array([b[1] for b in bases]);w=np.array([b[4][2] for b in bases]);al=np.array([b[5][2] for b in bases])
        heading=np.column_stack([np.cos(yaw),np.sin(yaw),np.zeros(n)])
        speed=np.sum(np.array([b[2] for b in bases])*heading,axis=1)
        acc=np.sum(np.array([b[3] for b in bases])*heading,axis=1)
        seed=p.ctx.base_joint(p.x,p.ctx.table(times))
        jerk=seed[11]+p.ctx.scale_v*(p.vbasis(times/p.T,2)@p.z[:4])/p.T**2;jerk[times>=p.T]=0.
        original,blocks=assemble(closures,q,v,gaps,pen,speed,acc,jerk,yaw,w,al,c['meta'])
        pp=p.z[8:17];raw=float(prep.f0+prep.g@pp+.5*pp@prep.H@pp)
        sc=scalar_check(p.ctx,p.z[:4]);cv=continuous_velocity(p.ctx,p.z[:4]);ac=angular_check(p.ctx,p.x,p.z)
        angular=1-np.array([ac['maximum_abs_yaw_delta_rad'],ac['maximum_abs_omega_radps'],ac['maximum_abs_alpha_radps2']])/np.array(ac['original_caps'])
        if not np.all(np.isfinite(original)) or not np.isfinite(raw):raise RuntimeError('Nonfinite full reference values')
        initial=float(abs(q[0]-p.ctx.q0[p.qa]).max());terminal=float(max(abs(v[-1]).max(),abs(a[-1]).max()))
        if max(initial,terminal)>=1e-9:raise RuntimeError('Reference initial q or terminal v/a contract failed')
        arrays=dict(time=times,parameters=p.z,q=q,v=v,a=a,N=p.N,
            nominal=np.array([p.input(t)['nominal'] for t in times]),
            constraints=original,closure=closures,gap=gaps,penetration=pen,speed=speed,acceleration=acc,jerk=jerk,
            base_p=np.array([b[0] for b in bases]),base_yaw=yaw,base_v=np.array([b[2] for b in bases]),
            base_a=np.array([b[3] for b in bases]),base_w=np.array([b[4] for b in bases]),base_alpha=np.array([b[5] for b in bases]),
            object_p=np.array([x[0] for x in objects]),object_v=np.array([x[1] for x in objects]),object_a=np.array([x[2] for x in objects]),
            object_R=np.array([x[3] for x in objects]),object_w=np.array([x[4] for x in objects]),
            object_alpha=np.array([x[5] for x in objects]),lifted_object_a=np.array(lift),predicted_TCP_moment=np.array(mom))
        np.savez_compressed(path,**arrays)
        packet=dict(f=raw/c['objective_scale'],raw=raw,original=original,cv=cv,angular=angular,
            scalar=sc,angular_check=ac,max_target_scaled=float(max(te)),max_closure_scaled=float(max(ce)),
            qualified=bool(min(original)>=-1e-6 and sc['passed'] and min(cv)>=-1e-6 and ac['passed']),
            minimum_gap=float(np.min(gaps)),max_penetration=float(max(pen)),initial_q_error=initial,terminal_error=terminal,
            path=str(path),IK_history=p.history,blocks=blocks)
        if n==45:packet['solver']=solver_view(original,jerk,cv,angular,c['fixed_endpoint_rows'])
        return packet,p
    except Exception:
        if p.kinematics:
            ts=sorted(p.kinematics)
            np.savez_compressed(HERE/'failed_reference_partial.npz',parameters=p.z,time=ts,
                q=[p.kinematics[t][0] for t in ts],v=[p.kinematics[t][1] for t in ts],a=[p.kinematics[t][2] for t in ts])
        raise

class Problem:
    def __init__(self,c,budget,prep):
        self.c,self.budget,self.prep=c,budget,prep;self.cache=OrderedDict();self.history=[];self.iterations=[];self.jac_history=[]
        self.evaluations=0;self.fun_calls=0;self.grad_calls=0;self.constraints_calls=0;self.role='constraint'
        self.last_jac=None;self.last_jac_key=None;self.numerical_jacobians=0;self.last_started=None
    def objective(self,z):
        self.budget.check();self.fun_calls+=1;p=np.asarray(z)[8:17]
        return float((self.prep.f0+self.prep.g@p+.5*p@self.prep.H@p)/self.c['objective_scale'])
    def gradient(self,z):
        self.budget.check();self.grad_calls+=1;g=np.zeros(50)
        g[8:17]=(self.prep.g+self.prep.H@np.asarray(z)[8:17])/self.c['objective_scale'];return g
    def evaluate(self,z):
        self.budget.check();z=np.asarray(z,dtype=float)
        if z.shape!=(50,) or not np.all(np.isfinite(z)):raise RuntimeError('Invalid solver parameter vector')
        key=z.tobytes()
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        cutoff=self.c['optimizer_evaluation_cutoff'] if self.budget.phase=='solve' else self.c['maximum_complete_evaluations']-1
        if self.evaluations>=cutoff:raise LimitReached('Complete evaluation cutoff; final audit reserved')
        self.evaluations+=1;number=self.evaluations;self.last_started=dict(number=number,parameters=z.tolist(),role=self.role)
        start=self.budget.elapsed();packet,p=core(self.prep,self.budget,self.c,z,self.c['times_s'],HERE/('eval_%04d.npz'%number))
        elapsed=self.budget.elapsed()-start
        if number==1:
            with np.load(CACHE/'case_0.npz') as prior:assert np.max(abs(packet['original']-prior['constraints']))<1e-10
        record=dict(evaluation=number,role=self.role,parameters=z.tolist(),f=packet['f'],raw=packet['raw'],
            min_original=float(min(packet['original'])),min_solver=float(min(packet['solver'])),
            reference_sample_qualified=packet['qualified'],physical_execution_admitted=False,
            gap_m=packet['minimum_gap'],max_target=packet['max_target_scaled'],elapsed_s=elapsed,total_wall_s=self.budget.elapsed(),
            IK_calls=len(p.kinematics),QP_calls=sum(x['steps'] for x in p.history),path=packet['path'])
        self.history.append(record)
        packet={k:v for k,v in packet.items() if k not in ['IK_history','blocks']}
        self.cache[key]=packet
        if len(self.cache)>64:self.cache.popitem(last=False)
        # Break the adapter's bound-method cycle; no live reference is kept in the value cache.
        p.external=None
        if number%10==0:self.progress()
        return packet
    def constraints(self,z):
        self.constraints_calls+=1;return self.evaluate(z)['solver']
    def jacobian(self,z):
        z=np.asarray(z,dtype=float);key=z.tobytes()
        if key==self.last_jac_key:return self.last_jac
        started=self.budget.elapsed();before=self.evaluations;old_role=self.role;self.role='jacobian'
        J=np.full((4928,50),np.nan);completed=0
        try:
            base=self.evaluate(z)['solver']
            for j in range(50):
                point=z.copy();point[j]+=self.c['optimizer_settings']['constraint_forward_step']
                J[:,j]=(self.evaluate(point)['solver']-base)/self.c['optimizer_settings']['constraint_forward_step'];completed+=1
            if not np.all(np.isfinite(J)):raise RuntimeError('Nonfinite solver constraint Jacobian')
            self.numerical_jacobians+=1;self.last_jac=J;self.last_jac_key=key
            self.jac_history.append(dict(number=self.numerical_jacobians,parameters=z.tolist(),evaluations=self.evaluations-before,
                wall_s=self.budget.elapsed()-started,maximum_abs=float(abs(J).max())))
            return J
        except Exception:
            np.savez_compressed(HERE/'partial_jacobian.npz',parameters=z,J=J,completed_columns=completed)
            raise
        finally:self.role=old_role
    def callback(self,z):
        if len(self.iterations)>=40:raise LimitReached('40 major iterations')
        value=self.evaluate(z)
        self.iterations.append(dict(iteration=len(self.iterations)+1,parameters=np.asarray(z).tolist(),
            f=value['f'],min_original=float(min(value['original'])),min_solver=float(min(value['solver'])),
            reference_sample_qualified=value['qualified'],wall_s=self.budget.elapsed()))
        self.progress()
    def progress(self):
        dump(HERE/'progress.json',dict(phase=self.budget.phase,iterations=len(self.iterations),
            evaluations=self.evaluations,jacobians=self.numerical_jacobians,stats=self.budget.stats,
            last_evaluation=self.history[-1] if self.history else self.last_started,
            last_iteration=self.iterations[-1] if self.iterations else None,wall_s=self.budget.elapsed()))

def dense_audit(problem,z):
    c=problem.c;b=problem.budget
    if problem.evaluations>=600:raise LimitReached('No complete evaluation slot left for final dense audit')
    b.phase='audit';problem.evaluations+=1;start=b.elapsed()
    packet,p=core(problem.prep,b,c,z,c['final_dense_times_s'],HERE/'final_dense.npz')
    if len(p.kinematics)>1400:raise RuntimeError('Final dense node limit')
    count=b.stats['IK_calls'];schema=validate(p.external,p.m,p.ctx.q0);diffs=[]
    for t in c['derivative_centers_s']:
        q,v,a=p.joint(t)
        for h in c['difference_steps_s']:
            qm=p.joint(t-h)[0];qp=p.joint(t+h)[0];vf=(qp-qm)/(2*h);af=(qp-2*q+qm)/h**2
            _,target,closure=p.observe(t,q,vf,af,False)
            nm=p.input(t-h)['nominal'][0];nc=p.input(t)['nominal'][0];np_=p.input(t+h)['nominal'][0]
            nv=float(abs(p.N.T@(vf-(np_-nm)/(2*h))).max());na=float(abs(p.N.T@(af-(np_-2*nc+nm)/h**2)).max())
            scale=64*np.finfo(float).eps*(1+max(abs(q).max(),abs(nc).max()))*np.linalg.norm(p.N.T,ord=np.inf)
            vb=2e-13/h+scale/h;ab=4e-13/h**2+scale/h**2
            diffs.append(dict(time_s=t,h_s=h,qdot_error=float(abs(vf-v).max()),qddot_error=float(abs(af-a).max()),
                target_scaled=target,closure_scaled=closure,null_velocity=nv,null_acceleration=na,
                null_velocity_bound=float(vb),null_acceleration_bound=float(ab),
                passed=bool(max(target,closure)<=1+1e-6 and nv<=vb and na<=ab)))
            b.check()
    qt=np.array(c['objective_times_s']);qw=np.array(c['objective_normalized_integration_weights'])
    target_acc=np.array([p.object_at(t)[2] for t in qt]);lifted_acc=[]
    for t in qt:
        q,v,a=p.joint(t);item,_,_=p.observe(t,q,v,a,False);lifted_acc.append(item['payload'][2])
    target_value=float(qw@np.sum(target_acc**2,axis=1));lifted_value=float(qw@np.sum(np.array(lifted_acc)**2,axis=1))
    assert b.stats['IK_calls']==count,'Unexpected new IK outside frozen dense union'
    passed=packet['qualified'] and all(x['passed'] for x in diffs) and abs(target_value-lifted_value)<1e-10
    audit=dict(passed=bool(passed),node_count=len(p.kinematics),wall_s=b.elapsed()-start,
        min_original=float(min(packet['original'])),minimum_gap_m=packet['minimum_gap'],max_penetration=packet['max_penetration'],
        max_target_scaled=packet['max_target_scaled'],max_closure_scaled=packet['max_closure_scaled'],
        scalar_check=packet['scalar'],angular_check=packet['angular_check'],continuous_velocity_min=float(min(packet['cv'])),
        schema=schema,differences=diffs,Gauss_target_mean_acc2=target_value,Gauss_lifted_mean_acc2=lifted_value,
        Gauss_target_lifted_difference=abs(target_value-lifted_value),new_IK_in_schema_differences_Gauss=0,
        physical_execution_admitted=False)
    dump(HERE/'final_dense_audit.json',audit);p.external=None;b.check()
    return audit

def run():
    c=guard();assert not (HERE/'result.json').exists()
    assert not (HERE/'optimizer_started.json').exists(),'The only optimizer call has already been started'
    b=Limits(c);problem=None;prep=None;opt=None;failure=None;dense=None;solver_started=False;selected=None
    try:
        prep=Preparation();problem=Problem(c,b,prep);setup=b.elapsed();b.phase='solve';b.check()
        solver_started=True
        dump(HERE/'optimizer_started.json',dict(calls=1,method='SLSQP',wall_s=b.elapsed()))
        opt=minimize(problem.objective,np.array(c['start_parameters']),method='SLSQP',jac=problem.gradient,
            constraints=[dict(type='ineq',fun=problem.constraints,jac=problem.jacobian)],callback=problem.callback,
            options=dict(maxiter=40,ftol=1e-10,disp=False))
        solver_elapsed=b.elapsed()
        b.phase='postsolve'
        if opt.success:
            value=problem.evaluate(opt.x);start=problem.history[0]['f']
            if value['f']<start-c['objective_improvement_numerical_tolerance']:
                dense=dense_audit(problem,np.array(opt.x))
                if dense['passed']:selected=np.asarray(opt.x).tolist()
        assert fixed_signature(prep.ctx)==prep.signature
        b.check()
    except Exception as e:failure=type(e).__name__+': '+str(e)
    if problem is not None:
        dump(HERE/'evaluation_history.json',problem.history);dump(HERE/'iterations.json',problem.iterations);dump(HERE/'jacobians.json',problem.jac_history)
    solver=None if opt is None else dict(success=bool(opt.success),status=int(opt.status),message=str(opt.message),
        nit=int(opt.nit),nfev=int(opt.nfev),njev=int(opt.njev),x=np.asarray(opt.x).tolist(),fun=float(opt.fun))
    result=dict(solver_calls=int(solver_started),solver=solver,failure=failure,
        optimizer_normal_success=bool(opt is not None and opt.success),final_dense_audit=dense,selected=selected,
        new_reference_qualified=bool(selected is not None),physical_execution_admitted=False,
        complete_evaluations=problem.evaluations if problem else 0,completed_coarse_evaluations=len(problem.history) if problem else 0,
        numerical_jacobians=problem.numerical_jacobians if problem else 0,
        objective_calls=problem.fun_calls if problem else 0,objective_gradient_calls=problem.grad_calls if problem else 0,
        stats=b.stats,run_wall_s=b.elapsed(),last_started_evaluation=problem.last_started if problem else None,
        fixed_preparation_unchanged=bool(prep is not None and fixed_signature(prep.ctx)==prep.signature),
        contract_sha256=sha(HERE/'contract.json'),new_physics_runs=0,new_physics_steps=0,
        scope='One bounded mature full-50 reference solve and at most one final dense reference audit; no physical execution or global/novelty claim.')
    dump(HERE/'result.json',result);guard()
    print(json.dumps({k:result[k] for k in ['solver_calls','solver','failure','complete_evaluations','numerical_jacobians',
        'stats','run_wall_s','new_reference_qualified','physical_execution_admitted']},ensure_ascii=False))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run']);args=ap.parse_args()
    prepare() if args.action=='prepare' else run()

