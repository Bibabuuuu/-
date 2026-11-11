"""One complete reference f/g cost sample; no Jacobian, outer solve, or physical step."""
from pathlib import Path
import argparse,json,sys,time
ROOT=Path(r'C:\Users\14737\Documents\Codex\2026-10-20\new-chat')
PREV=ROOT/'outputs/rby1-research/s3-feasible-reference-directions'
STRUCT=ROOT/'outputs/rby1-research/s3-structured-conditioning'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(PREV))
from qualify_directions import (Adapter,Budget,guard as previous_guard,angular_check,scalar_check,
    continuous_velocity,context,read,sha,dump,np)
from build_payload import rotvec

def assemble(closure,q,v,gap,penetration,speed,acc,jerk,yaw,w,alpha,meta):
    residual=np.sqrt(3)*closure/np.array(meta['closure_tolerance'])[None,:,None]
    groups=[
      ('closure_minus',1-residual),('closure_plus',1+residual),
      ('joint_lower',(q-np.array(meta['joint_lower']))/.1),
      ('joint_upper',(np.array(meta['joint_upper'])-q)/.1),
      ('joint_speed',1-abs(v)/np.array(meta['joint_speed_caps'])),
      ('geometry_gap',(gap-meta['gap_m'])/.05),
      ('penetration',((1e-5-penetration)/1e-4)[:,None]),
      ('base_speed_lower',(speed/.4)[:,None]),
      ('base_speed_upper',((meta['initial_speed_mps']-speed)/.4)[:,None]),
      ('base_acceleration_lower',(acc+1)[:,None]),('base_acceleration_upper',(-acc)[:,None]),
      ('base_jerk',(1-abs(jerk)/5)[:,None]),
      ('base_heading',(1-abs(yaw-meta['yaw0_rad'])/np.deg2rad(10))[:,None]),
      ('base_omega',(1-abs(w)/meta['max_omega_radps'])[:,None]),
      ('base_alpha',(1-abs(alpha)/meta['max_alpha_radps2'])[:,None])]
    blocks=[];chunks=[];start=0
    for name,a in groups:
        val=np.ravel(a);chunks.append(val)
        blocks.append(dict(name=name,start=start,stop=start+len(val),length=len(val),minimum=float(min(val))))
        start+=len(val)
    return np.concatenate(chunks),blocks

def guard():
    c=read(HERE/'contract.json');previous_guard()
    for p,h in c['source_sha256'].items():assert sha(p)==h,p
    return c

def prepare():
    assert not (HERE/'contract.json').exists()
    previous_guard();proposal=read(PREV/'next_evaluation_proposal.json')
    for p,h in proposal['source_sha256'].items():assert sha(p)==h,p
    old=read(STRUCT/'presolved_contract.json');ctx=context();times=old['coarse_times'];assert len(times)==45
    nodes,weights=np.polynomial.legendre.leggauss(12);l,h=ctx.breaks[:-1],np.diff(ctx.breaks)
    qt=(l[:,None]+h[:,None]*(nodes+1)/2).ravel()
    qw=(h[:,None]*np.tile(weights,(len(h),1))/2/ctx.horizon).ravel()
    assert len(qt)==60
    b=proposal['proposed_budget']
    meta=dict(closure_tolerance=ctx.closure_tol.tolist(),joint_lower=ctx.m.jnt_range[ctx.ids,0].tolist(),
        joint_upper=ctx.m.jnt_range[ctx.ids,1].tolist(),joint_speed_caps=ctx.vcaps.tolist(),gap_m=ctx.gap,
        initial_speed_mps=float(ctx.seed.values(0)[1]@np.array([np.cos(ctx.seed.yaw0),np.sin(ctx.seed.yaw0),0.])),
        yaw0_rad=ctx.seed.yaw0,max_omega_radps=ctx.maxw,max_alpha_radps2=ctx.maxalpha)
    c=dict(date='2026-11-02',authorization='User: 继续 after one complete f/g sample proposal.',
        parameters=np.zeros(50).tolist(),maximum_complete_candidates=1,times_s=times,
        objective_times_s=qt.tolist(),objective_normalized_integration_weights=qw.tolist(),
        objective_scale=old['scale'],objective_scale_source=str(STRUCT/'presolved_contract.json'),
        common_motion_window_s=ctx.horizon,T_s=ctx.T,meta=meta,
        maximum_IK_calls=b['maximum_IK_calls'],maximum_QP_calls=b['maximum_QP_calls'],
        maximum_QP_internal_iterations=b['maximum_QP_internal_iterations'],maximum_timed_run_wall_s=b['maximum_run_wall_s'],
        maximum_local_steps_per_IK=30,original_constraint_count=4860,acceptance_min_scaled=-1e-6,
        evaluation='One fresh zero candidate. Explicit shared COM objective on 60 Gauss nodes; lift and all original constraints on 45 coarse nodes. Target-vs-lift consistency on those 45 only.',
        not_in_scope=['No lifted joint verification at the 60 objective-only Gauss times.',
            'No numerical constraint Jacobian, optimizer, physical run, or actual initial-state projection.'],
        failure_policy='Keep first result and stop; no rerun, threshold, budget, model or controller changes.',
        source_sha256={str(p):sha(p) for p in [
          Path(__file__),PREV/'next_evaluation_proposal.json',PREV/'qualify_directions.py',
          PREV/'objective_structure.json',PREV/'objective_quadratic.npz',STRUCT/'presolved_contract.json']})
    dump(HERE/'contract.json',c);guard()
    print(json.dumps({'prepared':True,'coarse_nodes':45,'objective_only_nodes':60,'constraints':4860,'maximum_IK':45,'wall_s':30}))

def run():
    c=guard();assert not (HERE/'result.json').exists()
    budget=Budget(c);failure=None;passed=False;p=None;rows=[];blocks=[];objective=None;continuity=None;schema=None
    fields={};profile={}
    try:
        p=Adapter(c['parameters'],budget,0);p.external.source_paths+=(Path(__file__),HERE/'contract.json')
        profile['context_ready_s']=time.perf_counter()-budget.started
        sc=scalar_check(p.ctx,np.zeros(4));cv=continuous_velocity(p.ctx,np.zeros(4));ac=angular_check(p.ctx,p.x,np.zeros(50))
        continuity=dict(velocity=sc,velocity_min_scaled=float(min(cv)),angular=ac)
        times=np.array(c['times_s']);q=[];v=[];a=[];closure=[];gaps=[];pen=[];target=[];lifted=[];bases=[]
        for t in times:
            qq,vv,aa=p.joint(t);item,te,ce=p.observe(t,qq,vv,aa,True)
            q.append(qq);v.append(vv);a.append(aa);closure.append(item['closure']);gaps.append(item['gap']);pen.append(item['penetration'])
            target.append(p.object_at(t));lifted.append(item['payload']);bases.append(p.base_at(t))
            diff=np.array([item['payload'][0]-target[-1][0],rotvec(item['payload'][3]@target[-1][3].T),
                item['payload'][1]-target[-1][1],item['payload'][4]-target[-1][4],
                item['payload'][2]-target[-1][2],item['payload'][5]-target[-1][5]])
            rows.append(dict(time_s=float(t),target_scaled_error=te,closure_scaled_error=ce,
                target_component_max_abs=np.max(abs(diff),axis=1).tolist()))
            budget.check()
        q,v,a,closure,gaps,pen=[np.array(x) for x in [q,v,a,closure,gaps,pen]]
        profile['mapping_and_geometry_ready_s']=time.perf_counter()-budget.started
        # Reuse original scalar seed values; explicit corrections are zero in this one candidate.
        seed=p.ctx.base_joint(p.x,p.ctx.table(times))
        speed=seed[9];acc=seed[10];jerk=seed[11]
        yaw=np.array([b[1] for b in bases]);omega=np.array([b[4][2] for b in bases]);alpha=np.array([b[5][2] for b in bases])
        constraints,blocks=assemble(closure,q,v,gaps,pen,speed,acc,jerk,yaw,omega,alpha,c['meta'])
        assert len(constraints)==4860
        profile['constraints_ready_s']=time.perf_counter()-budget.started
        qt=np.array(c['objective_times_s']);qw=np.array(c['objective_normalized_integration_weights'])
        qa=np.array([p.object_at(t)[2] for t in qt]);mean=float(qw@np.sum(qa*qa,axis=1))
        with np.load(PREV/'objective_quadratic.npz') as model:
            exact=float(model['constant']);grad=np.zeros(50);grad[8:17]=model['g']/c['objective_scale']
        assert abs(mean-exact)<1e-10
        ta=np.array([z[2] for z in target]);la=np.array([z[2] for z in lifted])
        coarse_target=float(np.trapezoid(np.sum(ta*ta,axis=1),times)/c['common_motion_window_s'])
        coarse_lifted=float(np.trapezoid(np.sum(la*la,axis=1),times)/c['common_motion_window_s'])
        objective=dict(mean_acceleration_squared=mean,rms_acceleration=mean**.5,scaled=mean/c['objective_scale'],
            analytic_gradient_50=grad.tolist(),quadratic_value_error=abs(mean-exact),
            objective_only_Gauss_nodes=60,IK_at_objective_only_nodes=0,
            coarse_target_trapezoid_mean_acc2=coarse_target,coarse_lifted_trapezoid_mean_acc2=coarse_lifted,
            coarse_target_lifted_objective_difference=abs(coarse_target-coarse_lifted),
            comparison_scope='45 lifted constraint nodes only; coarse trapezoid is not the Gauss objective.')
        profile['objective_ready_s']=time.perf_counter()-budget.started
        before=budget.stats['IK_calls']
        q0_error=float(abs(q[0]-p.ctx.q0[p.qa]).max())
        terminal=float(max(abs(v[-1]).max(),abs(a[-1]).max()))
        assert all(np.array_equal(z,y) for z,y in zip(p.joint(p.T+1),p.joint(p.T)))
        assert budget.stats['IK_calls']==before
        schema=dict(initial_q_error=q0_error,terminal_joint_v_a_error=terminal,
            checked_query_times=45,terminal_hold_checked=True,
            scope='Only existing nodes; previous 31-node validate interface qualification retained, not rerun with extra IK.')
        fields=dict(time=times,q=q,v=v,a=a,N=p.N,nominal=np.array([p.input(t)['nominal'] for t in times]),
            closure=closure,gap=gaps,penetration=pen,constraints=constraints,speed=speed,acceleration=acc,jerk=jerk,
            base_p=np.array([b[0] for b in bases]),base_yaw=yaw,base_v=np.array([b[2] for b in bases]),
            base_a=np.array([b[3] for b in bases]),base_w=np.array([b[4] for b in bases]),base_alpha=np.array([b[5] for b in bases]),
            object_p=np.array([z[0] for z in target]),object_v=np.array([z[1] for z in target]),
            object_a=ta,object_R=np.array([z[3] for z in target]),object_w=np.array([z[4] for z in target]),
            object_alpha=np.array([z[5] for z in target]),lifted_object_a=la,
            objective_times=qt,objective_weights=qw,objective_acceleration=qa)
        np.savez_compressed(HERE/'evaluation.npz',**fields)
        passed=(float(min(constraints))>=-1e-6 and sc['passed'] and ac['passed'] and min(cv)>=-1e-6
            and max(row['target_scaled_error'] for row in rows)<=1+1e-6 and q0_error<1e-9 and terminal<1e-9
            and objective['coarse_target_lifted_objective_difference']<1e-10)
        budget.check()
    except Exception as e:
        failure=type(e).__name__+': '+str(e)
        if p is not None and p.kinematics and not fields:
            ts=sorted(p.kinematics)
            np.savez_compressed(HERE/'failed_partial.npz',time=ts,q=[p.kinematics[t][0] for t in ts],
                v=[p.kinematics[t][1] for t in ts],a=[p.kinematics[t][2] for t in ts],N=p.N)
    result=dict(passed=bool(passed and failure is None),failure=failure,complete_candidates_evaluated=1 if fields else 0,
        stats=budget.stats,run_wall_s=time.perf_counter()-budget.started,profile_cumulative_s=profile,
        blocks=blocks,kinematic_rows=rows,continuous_checks=continuity,objective=objective,schema=schema,
        IK_history=p.history if p is not None else [],contract_sha256=sha(HERE/'contract.json'),
        new_numerical_constraint_Jacobians=0,new_outer_optimizer_calls=0,new_physics_runs=0,new_physics_steps=0,
        scope='One zero-candidate complete sampled f/g evaluation cost; not optimization, physical or continuous-safety qualification.')
    dump(HERE/'result.json',result);guard()
    print(json.dumps({'passed':result['passed'],'failure':failure,'stats':budget.stats,'wall_s':result['run_wall_s'],
        'constraints':sum(x['length'] for x in blocks),'objective':objective['scaled'] if objective else None},ensure_ascii=False))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run']);args=ap.parse_args()
    prepare() if args.action=='prepare' else run()

