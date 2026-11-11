"""Bounded continuous shared-object adapter qualification; no outer solve or physics."""
from pathlib import Path
import argparse,hashlib,json,sys,time
ROOT=Path(r'C:\Users\14737\Documents\Codex\2026-10-20\new-chat')
PRIOR=ROOT/'outputs/rby1-research/s3-shared-object-prototype'
REVIEW=ROOT/'outputs/rby1-research/s3-joint-reference-contract'
VEL=ROOT/'outputs/rby1-research/s3-velocity-planning/continuous-limits'
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(PRIOR),str(VEL),str(ROOT/'simulation/rby1'),str(ROOT/'work/stop-optimizer-deps')]
from shared_object_prototype import Prototype,canonical,guard as parent_guard
from longitudinal_planner_complete import scalar_check,continuous_velocity
from whole_body_stop_reference import WholeBodyStopReference,validate
from load_sharing import feasible_qp
from qualify_planar_stop import rz
from build_payload import rotvec
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation
import numpy as np
import mujoco as mj
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    def native(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(str(type(x)))
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=native)+'\n',encoding='utf-8')
def skew(x):
    a,b,c=x
    return np.array([[0.,-c,b],[c,0.,-a],[-b,a,0.]])
def vee(K):return np.array([K[2,1]-K[1,2],K[0,2]-K[2,0],K[1,0]-K[0,1]])*.5
def rotation_motion(eta,ed,edd,R0,w0,a0):
    q=float(eta@eta);K=skew(eta);Kd=skew(ed);qd=2*float(eta@ed)
    if q<1e-8:
        A=.5-q/24+q*q/720-q**3/40320
        B=1/6-q/120+q*q/5040-q**3/362880
        Ad=(-1/24+q/360-q*q/13440)*qd
        Bd=(-1/120+q/2520-q*q/120960)*qd
    else:
        th=np.sqrt(q);td=float(eta@ed)/th
        A=(1-np.cos(th))/q;B=(th-np.sin(th))/th**3
        Ad=(np.sin(th)/q-2*(1-np.cos(th))/th**3)*td
        Bd=((1-np.cos(th))/th**3-3*(th-np.sin(th))/th**4)*td
    E=Rotation.from_rotvec(eta).as_matrix()
    J=np.eye(3)+A*K+B*K@K
    Jd=Ad*K+A*Kd+Bd*K@K+B*(Kd@K+K@Kd)
    we=J@ed;ae=J@edd+Jd@ed
    return E@R0,we+E@w0,ae+np.cross(we,E@w0)+E@a0

class Budget:
    def __init__(self,c):
        self.c=c;self.started=time.perf_counter()
        self.stats=dict(IK_calls=0,QP_calls=0,QP_internal_iterations=0,unique_targets=0,line_search_trials=0)
    def check(self,ik=False,qp=False):
        s=self.stats
        if time.perf_counter()-self.started>=self.c['maximum_timed_run_wall_s']:raise RuntimeError('120-second total run budget')
        if ik and s['IK_calls']>=self.c['maximum_IK_calls']:raise RuntimeError('404-IK total budget')
        if qp and (s['QP_calls']>=self.c['maximum_QP_calls'] or
                   s['QP_internal_iterations']+80>self.c['maximum_QP_internal_iterations']):
            raise RuntimeError('Global QP call/internal-iteration budget with 80-step reserve')

class Adapter(Prototype):
    def __init__(self,z,budget,case):
        self.z=np.array(z,dtype=float,copy=True)
        if self.z.shape!=(50,) or not np.all(np.isfinite(self.z)):raise ValueError('Expected 50 finite coefficients')
        self.z.flags.writeable=False;self.budget=budget;self.case=case;self.base_cache={}
        with np.load(PRIOR/'reference_samples.npz') as data:self.fixedN=data['N'].copy()
        self.fixedN.flags.writeable=False;self.N=self.fixedN
        with np.load(ROOT/'outputs/rby1-research/s3-structured-conditioning/alternative_velocity_basis.npz') as data:
            self.vbasis=BSpline(data['knots'],data['Z'],3,extrapolate=False)
        self.nodes,self.weights=np.polynomial.legendre.leggauss(8)
        super().__init__()
        self.N=self.fixedN;self.stats=budget.stats;self.x.flags.writeable=False
        assert np.max(abs(self.x[:4]))<1e-12
        self.identity=hashlib.sha256(self.z.tobytes()+self.N.tobytes()).hexdigest()
        self.external=WholeBodyStopReference(self.T,self.T,self.base_at,self.object_at,self.joint,self.qa,self.va,
            {'kind':'isolated_continuous_shared_object_reference','identity':self.identity,
             'initial_policy':'old nominal reference; no physical-state projection','force_distribution':'unchanged equal split'},
            (Path(__file__),HERE/'contract.json',PRIOR/'contract.json'))
    def key(self,t):
        if not np.isfinite(t):raise ValueError('Reference time must be finite')
        return canonical(float(t),self.T)
    def gate(self,qp=False):self.budget.check(qp=qp)
    def base_at(self,t):
        t=self.key(t)
        if t not in self.base_cache:
            old=self.ref.at(t)
            if np.max(abs(self.z[:8]))==0.:
                val=old
            else:
                u=t/self.T;c=self.ctx
                delta_yaw=c.scale_w*self.T*((c.bui(u)-c.bui(0.))@self.z[4:8])
                yaw=float(old[1]+delta_yaw)
                oldspeed=float(old[2]@np.array([np.cos(old[1]),np.sin(old[1]),0.]))
                oldacc=float(old[3]@np.array([np.cos(old[1]),np.sin(old[1]),0.]))
                v=oldspeed+c.scale_v*(self.vbasis(u)@self.z[:4])
                a=oldacc+c.scale_v*(self.vbasis(u,1)@self.z[:4])/self.T
                w=float(old[4][2]+c.scale_w*(c.bu(u)@self.z[4:8]))
                al=float(old[5][2]+c.scale_w*(c.bu(u,1)@self.z[4:8])/self.T)
                edges=np.unique(np.r_[0.,c.breaks[(c.breaks>0)&(c.breaks<t)],t])
                xy=c.seed.p0[:2].copy()
                if len(edges)>1:
                    left,width=edges[:-1],np.diff(edges)
                    gt=(left[:,None]+width[:,None]*(self.nodes+1)/2).ravel();ug=gt/self.T
                    legacy=c.base_joint(self.x,c.table(gt))
                    vg=legacy[9]+c.scale_v*(self.vbasis(ug)@self.z[:4])
                    yg=legacy[1]+c.scale_w*self.T*((c.bui(ug)-c.bui(0.))@self.z[4:8])
                    f=np.column_stack([vg*np.cos(yg),vg*np.sin(yg)]).reshape(-1,8,2)
                    xy+=np.einsum('ijk,j,i->k',f,self.weights,width/2)
                if t>=self.T:v=a=w=al=0.
                heading=np.array([np.cos(yaw),np.sin(yaw),0.]);tangent=np.array([-np.sin(yaw),np.cos(yaw),0.])
                val=(np.r_[xy,c.seed.p0[2]],yaw,v*heading,a*heading+v*w*tangent,np.array([0.,0.,w]),np.array([0.,0.,al]))
            self.base_cache[t]=tuple(x.copy() if isinstance(x,np.ndarray) else x for x in val)
        return tuple(x.copy() if isinstance(x,np.ndarray) else x for x in self.base_cache[t])
    def object_at(self,t):
        t=self.key(t);u=t/self.T
        b=[self.ctx.bq(u,k)/self.T**k for k in range(3)]
        old=self.ref.payload(t);trans=self.z[8:17].reshape(3,3);ori=self.z[17:26].reshape(3,3)
        dp,dv,da=[.01*(trans@x) for x in b]
        eta,ed,edd=[.1*(ori@x) for x in b]
        R,w,al=rotation_motion(eta,ed,edd,old[3],old[4],old[5])
        return old[0]+dp,old[1]+dv,old[2]+da,R,w,al
    def input(self,t):
        t=self.key(t)
        if t in self.inputs:return self.inputs[t]
        self.budget.check()
        c=self.ctx;p,yaw,v,a,w,al=self.base_at(t);obj=self.object_at(t)
        seed=tuple(x.copy() for x in self.ref.joint(t))
        b=[c.bq(t/self.T,k)/self.T**k for k in range(3)]
        zr=self.z[26:].reshape(8,3);zeta=tuple(.1*(zr@x) for x in b)
        nominal=tuple(seed[k]+self.fixedN@zeta[k] for k in range(3))
        delta=rz(yaw-c.seed.yaw0);Rroot=delta@c.Rroot0;r=delta@c.root_offset
        q=c.q0.copy();q[:3]=p+r;mj.mju_mat2Quat(q[3:7],Rroot.reshape(-1));q[self.qa]=seed[0]
        com,vo,ao,Ro,wo,alo=obj;q[c.pq:c.pq+3]=com-Ro@self.rOC
        mj.mju_mat2Quat(q[c.pq+3:c.pq+7],Ro.reshape(-1))
        vf=np.zeros(self.m.nv);af=vf.copy()
        vf[:3]=v+np.cross(w,r);vf[3:6]=Rroot.T@w
        af[:3]=a+np.cross(al,r)+np.cross(w,np.cross(w,r));af[3:6]=Rroot.T@al
        targets={};motion={}
        for side,(gp,gR) in c.grasps.items():
            offset=Ro@(gp-self.rOC);targets[side]=(com+offset,Ro@gR)
            motion[side]=(vo+np.cross(wo,offset),wo,ao+np.cross(alo,offset)+np.cross(wo,np.cross(wo,offset)),alo)
        out=dict(t=t,full_q=q,vf=vf,af=af,payload=obj,nominal=nominal,seed=seed,
                 zeta=zeta,targets=targets,hand_motion=motion)
        self.inputs[t]=out
        return out
    def solve_pose(self,t,seed,repeat=False):
        self.budget.check(ik=True);self.stats['IK_calls']+=1;inp=self.input(t);q=np.array(seed).copy()
        for step in range(31):
            self.gate();M,error=self.task(inp,q)
            pos=max(np.linalg.norm(error[k:k+3]) for k in [0,6])
            ang=max(np.linalg.norm(error[k+3:k+6])/.25 for k in [0,6]);anchor=float(abs(error[12:]).max())
            if pos<=1e-13 and ang<=1e-12 and anchor<=1e-13:
                row=dict(case=self.case,time_s=inp['t'],steps=step,position_residual_m=float(pos),
                    angle_residual_rad=float(ang),anchor_residual_rad=anchor,matrix_condition=float(np.linalg.cond(M)))
                self.history.append(row);return q,row
            if step==30:raise RuntimeError('30 local steps per target')
            self.gate(qp=True)
            lo=np.maximum(self.m.jnt_range[self.ctx.ids,0]-q,-.12);hi=np.minimum(self.m.jnt_range[self.ctx.ids,1]-q,.12)
            dq,info=feasible_qp(M.T@M+1e-5*self.W,-M.T@error,self.box,np.r_[hi,-lo],tolerance=1e-14)
            self.stats['QP_calls']+=1;self.stats['QP_internal_iterations']+=int(info.get('iterations',80))
            if not info['accepted']:raise RuntimeError('Existing box QP failed: '+str(info))
            accepted=False;norm=float(np.linalg.norm(error))
            for power in range(12):
                self.gate();self.stats['line_search_trials']+=1
                trial=q+dq*2.**(-power);_,e=self.task(inp,trial)
                if np.linalg.norm(e)<norm:q=trial;accepted=True;break
            if not accepted:raise RuntimeError('No decreasing local IK step')
    def joint(self,t):
        t=self.key(t)
        if t not in self.kinematics:
            inp=self.input(t);q,_=self.solve_pose(t,inp['seed'][0]);v,a,_=self.lift_derivatives(t,q)
            self.solutions[t]=q;self.kinematics[t]=(q,v,a);self.stats['unique_targets']+=1
        return tuple(x.copy() for x in self.kinematics[t])
    def observe(self,t,q,v,acc,geometry=False):
        base=list(self.ctx.base_joint(self.x,self.ctx.table([t])))
        bp,yaw,bv,ba,bw,bal=self.base_at(t)
        base[:6]=[bp[None,:],np.array([yaw]),bv[None,:],ba[None,:],np.array([bw[2]]),np.array([bal[2]])]
        base[6:9]=[q[None,:],v[None,:],acc[None,:]]
        item=self.ctx.sample(tuple(base),0,geometry);actual=item['payload'];desired=self.input(t)['payload']
        diff=np.array([actual[0]-desired[0],rotvec(actual[3]@desired[3].T),actual[1]-desired[1],
                       actual[4]-desired[4],actual[2]-desired[2],actual[5]-desired[5]])
        target=float(np.max(np.sqrt(3)*abs(diff)/self.ctx.closure_tol[:,None]))
        closure=float(np.max(np.sqrt(3)*abs(item['closure'])/self.ctx.closure_tol[:,None]))
        return item,target,closure

def rotation_self_check():
    eta=np.array([.2,-.1,.15]);ed=np.array([.13,.07,-.11]);edd=np.array([-.08,.04,.03]);ws=np.array([.03,-.05,.09])
    R0,w,a=rotation_motion(eta,ed,edd,np.eye(3),ws,np.zeros(3));rows=[]
    for h in [.001,.0005]:
        def at(s):return Rotation.from_rotvec(eta+s*ed+.5*s*s*edd).as_matrix()@Rotation.from_rotvec(s*ws).as_matrix()
        Rm,Rp=at(-h),at(h);Rd=(Rp-Rm)/(2*h);Rdd=(Rp-2*R0+Rm)/h**2
        we=vee(Rd@R0.T);ae=vee(Rdd@R0.T-skew(w)@skew(w))
        row=dict(h_s=h,omega_error=float(abs(we-w).max()),alpha_error=float(abs(ae-a).max()))
        assert max(row['omega_error'],row['alpha_error'])<1e-6,row
        rows.append(row)
    return rows

def guard():
    c=read(HERE/'contract.json');parent_guard()
    for path,h in c['source_sha256'].items():assert sha(path)==h,path
    return c

def prepare():
    assert not (HERE/'contract.json').exists()
    old=parent_guard();proposal=read(REVIEW/'proposal.json')
    for path,h in proposal['source_sha256'].items():assert sha(path)==h,path
    c=dict(proposal['next_qualification_proposed'])
    c.update(date='2026-11-02',authorization='User: 继续, after the concrete four-reference proposal.',
        scope='Isolated continuous adapter, old nominal initial reference, no physical state projection.',
        source_proposal=str(REVIEW/'proposal.json'),
        cases=['zero ascending','zero fixed permutation','positive mixed perturbation','negative mixed perturbation'],
        derivative_centers_s=old['derivative_centers_s'],difference_steps_s=old['difference_steps_s'],
        geometry_times_s=old['main_times_s'],coefficient_order=proposal['coefficient_order_proposed'],
        IK_tolerance=old['IK_numerical_tolerance'],numeric_repeat_tolerance_rad=1e-8,
        acceptance_min_scaled=-1e-6,
        run_order='Per case: scalar extrema preflight, then mapping/schema/FD/geometry. Stop at first failed case. No later case or retries.',
        progress='No implementation of an outer optimizer; actual source/controller/physics unchanged.',
        source_sha256={str(path):sha(path) for path in [
          Path(__file__),REVIEW/'proposal.json',PRIOR/'contract.json',PRIOR/'shared_object_prototype.py',
          PRIOR/'reference_samples.npz',VEL/'longitudinal_planner_complete.py']})
    dump(HERE/'contract.json',c);guard()
    print(json.dumps({'prepared':True,'maximum_IK':404,'wall_s':120,'physics':0}))

def run():
    c=guard();assert not (HERE/'result.json').exists()
    budget=Budget(c);cases=[];failure=None;completed=[];mathcheck=[];order_error=None
    direction=1e-4*np.cos(np.arange(50)+1);parameters=[np.zeros(50),np.zeros(50),direction,-direction]
    try:
        mathcheck=rotation_self_check()
        for ci,z in enumerate(parameters):
            case_start=time.perf_counter();p=Adapter(z,budget,ci);budget.check()
            scal=scalar_check(p.ctx,z[:4]);cons=continuous_velocity(p.ctx,z[:4])
            case=dict(case=ci,label=c['cases'][ci],parameters=z.tolist(),scalar_check=scal,
                minimum_continuous_scalar_constraint=float(min(cons)),mapping_started=False,passed=False)
            cases.append(case);dump(HERE/'progress.json',dict(stats=budget.stats,cases=cases,wall_s=time.perf_counter()-budget.started))
            if not scal['passed'] or min(cons)<c['acceptance_min_scaled']:
                raise RuntimeError('Case '+str(ci)+' fails unchanged continuous v/a/jerk limits before IK')
            times=c['times_s'];query=times if ci!=1 else [times[i] for i in sorted(range(len(times)),key=lambda i:(37*i)%101)]
            case['mapping_started']=True;rows=[];fd=[]
            case.update(kinematic_rows=rows,differences=fd,IK_history=p.history)
            for t in query:
                q,v,a=p.joint(t);isgeom=t in c['geometry_times_s']
                item,target,closure=p.observe(t,q,v,a,isgeom)
                margin=float(np.min(np.minimum(q-p.m.jnt_range[p.ctx.ids,0],p.m.jnt_range[p.ctx.ids,1]-q)))
                speed=float(max(abs(v)/p.ctx.vcaps))
                row=dict(time_s=t,target_scaled_error=target,closure_scaled_error=closure,joint_margin_rad=margin,speed_fraction=speed)
                if isgeom:row['geometry']=dict(gap_m=float(min(item['gap'])),penetration_m=float(item['penetration']))
                rows.append(row)
                if target>1+1e-6 or closure>1+1e-6 or margin<-.1e-6 or speed>1+1e-6:
                    raise RuntimeError('Case '+str(ci)+' original kinematic target/joint qualification failed')
                if isgeom and (min(item['gap'])<p.ctx.gap-.05e-6 or item['penetration']>1e-5):
                    raise RuntimeError('Case '+str(ci)+' original reference geometry qualification failed')
                budget.check()
            for t in c['derivative_centers_s']:
                q,v,a=p.joint(t)
                for h in c['difference_steps_s']:
                    qm=p.joint(t-h)[0];qp=p.joint(t+h)[0];vf=(qp-qm)/(2*h);af=(qp-2*q+qm)/h**2
                    _,target,closure=p.observe(t,q,vf,af)
                    nm=p.input(t-h)['nominal'][0];nc=p.input(t)['nominal'][0];np_=p.input(t+h)['nominal'][0]
                    nv=float(abs(p.N.T@(vf-(np_-nm)/(2*h))).max())
                    na=float(abs(p.N.T@(af-(np_-2*nc+nm)/h**2)).max())
                    scale=64*np.finfo(float).eps*(1+max(abs(q).max(),abs(nc).max()))*np.linalg.norm(p.N.T,ord=np.inf)
                    vb=2e-13/h+scale/h;ab=4e-13/h**2+scale/h**2
                    b0=p.base_at(t);bm=p.base_at(t-h);bp=p.base_at(t+h)
                    bv=(bp[0]-bm[0])/(2*h);ba=(bp[0]-2*b0[0]+bm[0])/h**2
                    o0=p.object_at(t);om=p.object_at(t-h);op=p.object_at(t+h)
                    ov=(op[0]-om[0])/(2*h);oa=(op[0]-2*o0[0]+om[0])/h**2
                    ow=vee((op[3]-om[3])/(2*h)@o0[3].T)
                    oal=vee((op[3]-2*o0[3]+om[3])/h**2@o0[3].T-skew(o0[4])@skew(o0[4]))
                    ratios=np.sqrt(3)*np.array([max(abs(ov-o0[1])),max(abs(ow-o0[4])),max(abs(oa-o0[2])),max(abs(oal-o0[5]))])/p.ctx.closure_tol[[2,3,4,5]]
                    row=dict(time_s=t,h_s=h,qdot_error=float(abs(vf-v).max()),qddot_error=float(abs(af-a).max()),
                         target_scaled_error=target,closure_scaled_error=closure,null_velocity=nv,null_acceleration=na,
                         null_velocity_bound=float(vb),null_acceleration_bound=float(ab),
                         base_velocity_error=float(abs(bv-b0[2]).max()),base_acceleration_error=float(abs(ba-b0[3]).max()),
                         payload_matrix_FD_scaled=float(max(ratios)))
                    fd.append(row)
                    if max(target,closure,float(max(ratios)))>1+1e-6 or nv>vb or na>ab:
                        raise RuntimeError('Case '+str(ci)+' original FD qualification failed')
                    if (row['base_velocity_error']>p.ctx.closure_tol[2]/np.sqrt(3) or
                        row['base_acceleration_error']>p.ctx.closure_tol[4]/np.sqrt(3)):
                        raise RuntimeError('Case '+str(ci)+' base position derivatives inconsistent')
            schema=validate(p.external,p.m,p.ctx.q0)
            before=budget.stats['IK_calls']
            assert all(np.array_equal(a,b) for a,b in zip(p.joint(-1.),p.joint(0.)))
            assert all(np.array_equal(a,b) for a,b in zip(p.joint(p.T+1.),p.joint(p.T)))
            for bad in [float('nan'),float('inf')]:
                try:p.joint(bad)
                except ValueError:pass
                else:raise AssertionError('Nonfinite time accepted')
            assert budget.stats['IK_calls']==before
            qs=np.array([p.kinematics[t][0] for t in times]);vs=np.array([p.kinematics[t][1] for t in times]);aas=np.array([p.kinematics[t][2] for t in times])
            nominal=np.array([p.input(t)['nominal'] for t in times])
            if ci==0:
                with np.load(PRIOR/'reference_samples.npz') as old:
                    index={t:i for i,t in enumerate(times)};ix=[index[t] for t in old['time']]
                    regression=[float(abs(new[ix]-old[key]).max()) for new,key in [(qs,'q'),(vs,'v'),(aas,'a')]]
                case['old_finite_replay_max_q_v_a_difference']=regression
                if max(regression)>1e-8:raise RuntimeError('Zero candidate old finite replay regression failed')
            if ci==1:
                with np.load(HERE/'case_0.npz') as old:
                    order_error=max(float(abs(qs-old['q']).max()),float(abs(vs-old['v']).max()),float(abs(aas-old['a']).max()))
                if order_error>1e-8:raise RuntimeError('Query-order repeatability failed')
            base=[p.base_at(t) for t in times];obj=[p.object_at(t) for t in times]
            np.savez_compressed(HERE/('case_'+str(ci)+'.npz'),time=times,q=qs,v=vs,a=aas,N=p.N,nominal=nominal,
                base_p=[x[0] for x in base],base_yaw=[x[1] for x in base],base_v=[x[2] for x in base],base_a=[x[3] for x in base],
                base_w=[x[4] for x in base],base_alpha=[x[5] for x in base],
                object_p=[x[0] for x in obj],object_v=[x[1] for x in obj],object_a=[x[2] for x in obj],
                object_R=[x[3] for x in obj],object_w=[x[4] for x in obj],object_alpha=[x[5] for x in obj])
            case.update(passed=True,kinematic_rows=rows,differences=fd,schema=schema,IK_history=p.history,
                wall_s=time.perf_counter()-case_start,queries_solved=len(p.kinematics))
            completed.append(ci);budget.check()
            dump(HERE/'progress.json',dict(stats=budget.stats,cases=cases,wall_s=time.perf_counter()-budget.started))
    except Exception as e:failure=type(e).__name__+': '+str(e)
    result=dict(passed=failure is None and len(completed)==4,failure=failure,
        completed_cases=completed,cases=cases,stats=budget.stats,rotation_math_check=mathcheck,
        zero_query_order_max_q_v_a_difference=order_error,run_wall_s=time.perf_counter()-budget.started,
        new_outer_optimizer_calls=0,new_physics_runs=0,new_physics_steps=0,
        contract_sha256=sha(HERE/'contract.json'),scope='Isolated numerical reference qualification only; not dynamics, physical handoff or planning success')
    dump(HERE/'result.json',result);guard()
    print(json.dumps({k:result[k] for k in ['passed','failure','completed_cases','stats','run_wall_s','zero_query_order_max_q_v_a_difference']},ensure_ascii=False))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.action=='prepare' else run()

