"""Fixed preparation reuse with private per-reference data; two bounded complete evaluations."""
from pathlib import Path
import argparse,copy,hashlib,sys,time
ROOT=Path(r'C:\Users\14737\Documents\Codex\2026-10-20\new-chat')
PREV=ROOT/'outputs/rby1-research/s3-complete-reference-evaluation'
QUAL=ROOT/'outputs/rby1-research/s3-feasible-reference-directions'
STRUCT=ROOT/'outputs/rby1-research/s3-structured-conditioning'
CONT=ROOT/'outputs/rby1-research/s3-continuous-reference-qualification'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(PREV))
from evaluate_reference import (assemble,guard as previous_guard,Adapter,Budget,angular_check,scalar_check,
    continuous_velocity,context,read,sha,dump,np)
from whole_body_stop_reference import WholeBodyStopReference
import mujoco as mj

def fixed_signature(ctx):
    h=hashlib.sha256()
    for name in ['body_mass','body_inertia','body_ipos','geom_size','geom_friction','jnt_range','actuator_forcerange']:
        h.update(np.asarray(getattr(ctx.m,name)).tobytes())
    for a in [ctx.q0,ctx.v0,ctx.qa,ctx.va,ctx.vcaps,ctx.root_offset,ctx.Rroot0,ctx.closure_tol]:
        h.update(np.asarray(a).tobytes())
    for s in ['left','right']:
        for a in ctx.grasps[s]:h.update(np.asarray(a).tobytes())
    h.update(str((ctx.m.opt.timestep,ctx.m.opt.impratio,ctx.m.opt.noslip_iterations)).encode())
    return h.hexdigest()

class Preparation:
    def __init__(self):
        self.ctx=context()
        with np.load(STRUCT/'structure.npz') as d:self.x=d['x0'].copy()
        with np.load(ROOT/'outputs/rby1-research/s3-shared-object-prototype/reference_samples.npz') as d:self.N=d['N'].copy()
        from scipy.interpolate import BSpline
        with np.load(STRUCT/'alternative_velocity_basis.npz') as d:self.vbasis=BSpline(d['knots'],d['Z'],3,extrapolate=False)
        with np.load(QUAL/'objective_quadratic.npz') as d:self.H,self.g,self.f0=d['H'].copy(),d['g'].copy(),float(d['constant'])
        self.nodes,self.weights=np.polynomial.legendre.leggauss(8)
        self.W=np.diag(1/self.ctx.vcaps**2);self.box=np.vstack([np.eye(20),-np.eye(20)])
        self.rOC=self.ctx.m.body_ipos[self.ctx.ob].copy()
        for a in [self.x,self.N,self.H,self.g,self.nodes,self.weights,self.W,self.box,self.rOC]:a.flags.writeable=False
        self.signature=fixed_signature(self.ctx)
    def private_context(self):
        c=copy.copy(self.ctx)
        c.d=mj.MjData(c.m);c.plan_cache={}
        # Geometry has only fixed compiled geometry data; its distance calls use the supplied MjData.
        return c

class CachedAdapter(Adapter):
    def __init__(self,z,budget,case,prep):
        self.z=np.array(z,dtype=float,copy=True)
        if self.z.shape!=(50,) or not np.all(np.isfinite(self.z)):raise ValueError('Expected 50 finite coefficients')
        self.z.flags.writeable=False;self.budget=budget;self.case=case
        self.ctx=prep.private_context();self.m=self.ctx.m;self.d=mj.MjData(self.m)
        self.x=prep.x;self.ref=self.ctx.as_reference(self.x);self.T=self.ctx.T
        self.qa,self.va,self.sites=self.ctx.qa,self.ctx.va,self.ctx.sites
        self.fixedN=prep.N;self.N=self.fixedN;self.vbasis=prep.vbasis;self.rOC=prep.rOC
        self.nodes,self.weights=prep.nodes,prep.weights;self.W,self.box=prep.W,prep.box
        self.inputs={};self.solutions={};self.kinematics={};self.base_cache={}
        self.stats=budget.stats;self.history=[];self.started=None;self.c=None
        inp=self.input(0.);self.forward(inp['full_q'],np.zeros(20))
        self.identity=hashlib.sha256(self.z.tobytes()+self.N.tobytes()).hexdigest()
        self.external=WholeBodyStopReference(self.T,self.T,self.base_at,self.object_at,self.joint,self.qa,self.va,
            {'kind':'isolated_cached_shared_object_reference','identity':self.identity,
             'fixed_signature':prep.signature,'initial_policy':'unchanged old nominal boundary; no physical-state projection'},
            (Path(__file__),HERE/'contract.json',PREV/'contract.json'))

def guard():
    c=read(HERE/'contract.json');previous_guard()
    for p,h in c['source_sha256'].items():assert sha(p)==h,p
    return c

def prepare():
    assert not (HERE/'contract.json').exists()
    previous=previous_guard();proposal=read(PREV/'next_cached_evaluation_proposal.json')
    for p,h in proposal['source_sha256'].items():assert sha(p)==h,p
    parameters=[np.zeros(50).tolist(),read(QUAL/'contract.json')['parameters'][0]]
    b=proposal['proposed_budget']
    c={k:previous[k] for k in ['times_s','objective_times_s','objective_normalized_integration_weights',
        'objective_scale','objective_scale_source','common_motion_window_s','T_s','meta']}
    c.update(date='2026-11-02',authorization='User: 继续 after fixed-preparation reuse and two-evaluation proposal.',
        parameters=parameters,maximum_complete_candidates=2,
        maximum_IK_calls=b['maximum_IK_calls'],maximum_QP_calls=b['maximum_QP_calls'],
        maximum_QP_internal_iterations=b['maximum_QP_internal_iterations'],maximum_timed_run_wall_s=b['maximum_run_wall_s'],
        original_constraint_count_each=4860,maximum_local_steps_per_IK=30,
        acceptance_min_scaled=-1e-6,
        timing='Whole run includes fixed setup, both evaluations, direct objective audits, cache isolation and output. Per-evaluation time includes private context construction and f/g; direct objective audit/save separately reported.',
        scope='Original rules and only two previously qualified references; no numerical constraint Jacobian, optimizer or physics.',
        failure_policy='Stop at first failure/budget, keep results; no retuning, retries, model or threshold changes.',
        source_sha256={str(p):sha(p) for p in [
          Path(__file__),PREV/'next_cached_evaluation_proposal.json',PREV/'evaluate_reference.py',
          PREV/'evaluation.npz',PREV/'result.json',QUAL/'contract.json',QUAL/'case_0.npz',
          QUAL/'objective_quadratic.npz',CONT/'continuous_reference.py']})
    dump(HERE/'contract.json',c);guard()
    print(__import__('json').dumps({'prepared':True,'complete_candidates':2,'IK_budget':90,'wall_budget_s':30}))

def fingerprint(p):
    h=hashlib.sha256()
    for t in sorted(p.kinematics):
        for a in p.kinematics[t]:h.update(a.tobytes())
    return h.hexdigest()

def evaluate(p,c,prep,ci):
    begin=time.perf_counter();times=np.array(c['times_s']);rows=[];q=[];v=[];a=[];closures=[];gaps=[];pen=[];bases=[];target=[];lifted=[]
    for t in times:
        qq,vv,aa=p.joint(t);item,te,ce=p.observe(t,qq,vv,aa,True)
        q.append(qq);v.append(vv);a.append(aa);closures.append(item['closure']);gaps.append(item['gap']);pen.append(item['penetration'])
        bases.append(p.base_at(t));target.append(p.object_at(t));lifted.append(item['payload'][2])
        rows.append(dict(time_s=float(t),target_scaled_error=te,closure_scaled_error=ce))
        p.gate()
    q,v,a,closures,gaps,pen=[np.array(x) for x in [q,v,a,closures,gaps,pen]]
    seed=p.ctx.base_joint(p.x,p.ctx.table(times));u=times/p.T
    # The complete nonzero scalar reference uses the restored velocity basis and heading.
    heading=np.column_stack([np.cos([b[1] for b in bases]),np.sin([b[1] for b in bases]),np.zeros(len(times))])
    speed=np.sum(np.array([b[2] for b in bases])*heading,axis=1)
    acceleration=np.sum(np.array([b[3] for b in bases])*heading,axis=1)
    jerk=seed[11]+p.ctx.scale_v*(p.vbasis(u,2)@p.z[:4])/p.T**2
    jerk[times>=p.T]=0.
    yaw=np.array([b[1] for b in bases]);w=np.array([b[4][2] for b in bases]);al=np.array([b[5][2] for b in bases])
    cons,blocks=assemble(closures,q,v,gaps,pen,speed,acceleration,jerk,yaw,w,al,c['meta'])
    pp=p.z[8:17];mean=float(prep.f0+prep.g@pp+.5*pp@prep.H@pp)
    grad=np.zeros(50);grad[8:17]=(prep.g+prep.H@pp)/c['objective_scale']
    sc=scalar_check(p.ctx,p.z[:4]);cv=continuous_velocity(p.ctx,p.z[:4]);ac=angular_check(p.ctx,p.x,p.z)
    elapsed=time.perf_counter()-begin
    if len(cons)!=4860 or min(cons)<-1e-6 or not sc['passed'] or min(cv)<-1e-6 or not ac['passed']:
        raise RuntimeError('Original full constraints/base qualification failed')
    if max(x['target_scaled_error'] for x in rows)>1+1e-6:raise RuntimeError('Target mapping qualification failed')
    if np.max(abs(q[0]-p.ctx.q0[p.qa]))>=1e-9 or max(abs(v[-1]).max(),abs(a[-1]).max())>=1e-9:
        raise RuntimeError('Initial q or terminal v/a qualification failed')
    audit_start=time.perf_counter();qt=np.array(c['objective_times_s']);qw=np.array(c['objective_normalized_integration_weights'])
    direct_acc=np.array([p.object_at(t)[2] for t in qt]);direct=float(qw@np.sum(direct_acc*direct_acc,axis=1))
    if abs(direct-mean)>1e-10:raise RuntimeError('Audited quadratic/direct reference objective mismatch')
    direct_time=time.perf_counter()-audit_start
    ta=np.array([z[2] for z in target]);la=np.array(lifted)
    arrays=dict(time=times,q=q,v=v,a=a,N=p.N,nominal=np.array([p.input(t)['nominal'] for t in times]),
        closure=closures,gap=gaps,penetration=pen,constraints=cons,speed=speed,acceleration=acceleration,jerk=jerk,
        base_p=np.array([b[0] for b in bases]),base_yaw=yaw,base_v=np.array([b[2] for b in bases]),
        base_a=np.array([b[3] for b in bases]),base_w=np.array([b[4] for b in bases]),base_alpha=np.array([b[5] for b in bases]),
        object_p=np.array([z[0] for z in target]),object_v=np.array([z[1] for z in target]),object_a=ta,
        object_R=np.array([z[3] for z in target]),object_w=np.array([z[4] for z in target]),object_alpha=np.array([z[5] for z in target]),
        lifted_object_a=la,objective_times=qt,objective_weights=qw,objective_acceleration=direct_acc)
    np.savez_compressed(HERE/('case_'+str(ci)+'.npz'),**arrays)
    return dict(case=ci,passed=True,parameters=p.z.tolist(),identity=p.identity,blocks=blocks,kinematic_rows=rows,
        objective=dict(mean_acceleration_squared=mean,scaled=mean/c['objective_scale'],analytic_gradient_50=grad.tolist(),
            direct_reference_error=abs(mean-direct)),
        complete_f_g_wall_s=elapsed,direct_objective_audit_wall_s=direct_time,IK_history=p.history,
        continuous_checks=dict(velocity=sc,velocity_min_scaled=float(min(cv)),angular=ac),
        target_lifted_acceleration_max_error=float(abs(ta-la).max()),
        min_constraint=float(min(cons)),minimum_gap_m=float(np.min(gaps)))

def run():
    c=guard();assert not (HERE/'result.json').exists()
    budget=Budget(c);prep=None;cases=[];refs=[];failure=None;isolation=None;setup_wall=None;zero_snapshot=None;p=None
    try:
        prep=Preparation();setup_wall=time.perf_counter()-budget.started;budget.check()
        for ci,z in enumerate(c['parameters']):
            started=time.perf_counter();p=CachedAdapter(z,budget,ci,prep)
            init=time.perf_counter()-started
            case=evaluate(p,c,prep,ci);case['private_reference_initialization_wall_s']=init
            case['complete_f_g_with_private_init_wall_s']=init+case['complete_f_g_wall_s']
            refs.append(p);cases.append(case);budget.check()
            if ci==0:zero_snapshot=fingerprint(p)
            dump(HERE/'progress.json',dict(stats=budget.stats,cases=cases,setup_wall_s=setup_wall,run_wall_s=time.perf_counter()-budget.started))
        a,b=refs;before=budget.stats['IK_calls'];sig=zero_snapshot
        for t in c['times_s']:a.joint(t)
        isolation=dict(shared_model=a.m is b.m is prep.ctx.m,
            private_IK_data=a.d is not b.d,private_reference_data=a.ctx.d is not b.ctx.d,
            independent_MjData_arrays=not np.shares_memory(a.d.qpos,b.d.qpos) and not np.shares_memory(a.ctx.d.qpos,b.ctx.d.qpos),
            private_contexts=a.ctx is not b.ctx,private_plan_cache=a.ctx.plan_cache is not b.ctx.plan_cache,
            private_inputs=a.inputs is not b.inputs,private_joint_cache=a.kinematics is not b.kinematics,
            private_base_cache=a.base_cache is not b.base_cache,
            distinct_candidate_identity=a.identity!=b.identity,
            zero_candidate_cached_requery_unchanged=fingerprint(a)==sig,
            cached_requery_new_IK=budget.stats['IK_calls']-before,
            fixed_preparation_unchanged=fixed_signature(prep.ctx)==prep.signature)
        assert all(value for key,value in isolation.items() if key!='cached_requery_new_IK')
        assert isolation['cached_requery_new_IK']==0
        budget.check()
    except Exception as e:
        failure=type(e).__name__+': '+str(e)
        if p is not None and p.kinematics:
            ts=sorted(p.kinematics)
            np.savez_compressed(HERE/'failed_partial.npz',time=ts,q=[p.kinematics[t][0] for t in ts],
                v=[p.kinematics[t][1] for t in ts],a=[p.kinematics[t][2] for t in ts],N=p.N)
    result=dict(passed=failure is None and len(cases)==2,failure=failure,cases=cases,stats=budget.stats,
        setup_wall_s=setup_wall,run_wall_s=time.perf_counter()-budget.started,cache_isolation=isolation,
        fixed_signature=prep.signature if prep is not None else None,zero_cache_snapshot=zero_snapshot,
        zero_cache_final=fingerprint(refs[0]) if refs else None,contract_sha256=sha(HERE/'contract.json'),
        new_numerical_constraint_Jacobians=0,new_outer_optimizer_calls=0,new_physics_runs=0,new_physics_steps=0,
        scope='Two previously-qualified complete evaluations with fixed preparation reuse; no Jacobian, optimization, online or physical claim.')
    dump(HERE/'result.json',result);guard()
    print(__import__('json').dumps({'passed':result['passed'],'failure':failure,'stats':budget.stats,
        'setup_s':setup_wall,'per_evaluation_s':[x['complete_f_g_with_private_init_wall_s'] for x in cases],
        'run_s':result['run_wall_s'],'cache_isolation':isolation},ensure_ascii=False))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run']);args=ap.parse_args()
    prepare() if args.action=='prepare' else run()

