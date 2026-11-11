"""Authorized bounded B0/B1 kinematic reference comparison; no physics."""
from pathlib import Path
from collections import OrderedDict
import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET

START = time.perf_counter()
import qualify_explicit_stop_mapping as old
from qualify_explicit_stop_mapping import (np, mj, ExplicitMapping, WholeBodyStopReference,
    _make_reference, joint_indices, canonical, read, dump, sha, INITIAL, PATH_FILE, PATH_AUDIT,
    ARCHIVE, ROOT, SCREEN, PRIOR, QUAL, rz, validate)
from continuous_reference import rotation_motion, skew, vee
from scipy.interpolate import BSpline, PPoly
from scipy.optimize import minimize
from numpy.polynomial import Polynomial as Poly
from payload_wrench import split_wrench

DEST = ARCHIVE/'s3-b0-b1-continuous-v1'
D = .7865521204483363
HORIZON = .8998625445380035
SEEDS = [[.2, 0., 0.], [.5, 0., .5], [.66, -.37, .41]]
GLOBAL_LIMITS = dict(IK_calls=90000, QP_calls=240000, QP_internal_iterations=480000,
                     complete_started=1568, outer_calls=4)
PHASE_LIMITS = dict(preparepilot=60., b0=180., b1=840., dense=120.)


class LimitReached(RuntimeError): pass


class Budget:
    def __init__(self, state, phase):
        self.used = state.get('active_s', 0.)
        self.phases = state.get('phases', {}).copy()
        self.phase, self.phase_started = phase, START
        self.stats = state.get('stats', dict(IK_calls=0, QP_calls=0, QP_internal_iterations=0,
            unique_targets=0, line_search_trials=0, complete_started=0, complete_completed=0,
            outer_calls=0, jacobians=0, cache_hits=0, requests=0))

    def elapsed(self): return self.used+time.perf_counter()-START

    def phase_elapsed(self): return self.phases.get(self.phase, 0.)+time.perf_counter()-self.phase_started

    def switch(self, phase):
        self.phases[self.phase] = self.phase_elapsed()
        self.phase, self.phase_started = phase, time.perf_counter()
        self.check()

    def check(self, ik=False, qp=False, complete=False, outer=False):
        if self.elapsed() >= 1200. or self.phase_elapsed() >= PHASE_LIMITS[self.phase]:
            raise LimitReached('Active-time budget: '+self.phase)
        for name, requested in [('IK_calls', ik), ('QP_calls', qp),
                                ('complete_started', complete), ('outer_calls', outer)]:
            if requested and self.stats[name] >= GLOBAL_LIMITS[name]: raise LimitReached(name)
        if qp and self.stats['QP_internal_iterations']+80 > 480000: raise LimitReached('QP internal reserve')

    def snapshot(self):
        return dict(active_s=self.elapsed(), stats=self.stats,
                    phases={**self.phases, self.phase:self.phase_elapsed()})


def basis():
    with np.load(ARCHIVE/'s3-structured-conditioning/alternative_velocity_basis.npz') as z:
        velocity = BSpline(z['knots'], z['Z'], 3, extrapolate=False)
    knots = np.r_[np.zeros(6), 1/3, 2/3, np.ones(6)]
    b = BSpline(knots, np.eye(8), 5, extrapolate=False)
    eq = np.vstack([b(0), b(0,1), b(0,2), b(1,1), b(1,2)])
    eu = np.vstack([b(0), b(0,1), b(1), b(1,1)])
    position = BSpline(knots, np.linalg.svd(eq, full_matrices=True)[2][5:].T, 5, extrapolate=False)
    angular = BSpline(knots, np.linalg.svd(eu, full_matrices=True)[2][4:].T, 5, extrapolate=False)
    return velocity, angular, position


class Preparation:
    def __init__(self, budget):
        # The old factory performs no IK and preserves the current actual state.
        p = old.make_case(('prepare','upper_bar',.6,0.,16), budget)
        self.m, self.q0, self.v0 = p.m, p.ctx.q0, p.ctx.v0
        self.axle, self.yaw, self.speed = p.base.p0, p.ctx.seed.yaw0, p.speed
        self.grasps, self.N = p.ctx.grasps, p.N
        self.qa, self.va = p.qa, p.va
        self.vbasis, self.wbasis, self.pbasis = basis()
        self.wintegral = self.wbasis.antiderivative()
        self.gnodes, self.gweights = np.polynomial.legendre.leggauss(12)
        fastest = self.speed+1/5
        self.maxw = np.deg2rad(10)*1.875/fastest
        self.maxalpha = np.deg2rad(10)*(10/np.sqrt(3))/fastest**2
        p.external = None


class GeneralMapping(ExplicitMapping):
    def __init__(self, x, prep, budget, label):
        self.x = np.asarray(x, dtype=float).copy()
        assert self.x.shape == (53,) and np.all(np.isfinite(self.x))
        A, angle, f = .5+.5*self.x[0], 10*self.x[1], self.x[2]
        assert .5 <= A <= 1. and abs(angle) <= 10. and 0. <= f <= 1.
        base, upper, *_ = _make_reference(prep.m, prep.q0, prep.v0,
            dict(label=label, A_mps2=A, angle_deg=angle, fraction=f if A>.5 else 0.))
        self.seed = WholeBodyStopReference.from_legacy(prep.m, prep.q0, base, upper)
        if self.seed.joint_at is None:
            self.seed.qadr, self.seed.vadr = prep.qa, prep.va
            self.seed.joint_at = lambda t: (prep.q0[prep.qa].copy(), np.zeros(20), np.zeros(20))
        self.base, self.upper, self.prep, self.A = base, upper, prep, A
        self.object_A = A-f*(A-.5)
        self.z = self.x[3:]
        self.base_cache = {}
        self.Tb, self.Ta = self.seed.T, self.seed.T_all
        self.edges = np.unique(np.r_[0., A/5., self.Tb-A/5., self.Tb,
            prep.vbasis.t*self.Tb, prep.wbasis.t*self.Tb,
            self.object_A/5., self.Ta-self.object_A/5., self.Ta, prep.pbasis.t*self.Ta])
        nominal = WholeBodyStopReference(self.Tb, self.Ta, self.base_at, self.object_at,
            self.nominal_joint, prep.qa, prep.va, dict(kind='bounded_53_parameter_reference'), ())
        super().__init__(prep.m, prep.q0, prep.v0, nominal, prep.axle, prep.yaw,
                         prep.grasps, prep.N, budget, label)
        self.speed = prep.speed
        self.external.config.update(physical_execution_admitted=False, coefficients=self.x.tolist())
        self.external.source_paths = (Path(__file__), DEST/'contract.json')

    def scalar(self, t):
        t = np.clip(t, 0., self.Tb); u = t/self.Tb; p = self.prep
        _, v, a = self.base.otg.at_time(float(t))
        speed = v[0]+.1*(p.vbasis(u) @ self.z[:4])
        acc = a[0]+.1*(p.vbasis(u,1) @ self.z[:4])/self.Tb
        seed = self.base.at(float(t))
        yaw = seed[1]+.3*self.Tb*((p.wintegral(u)-p.wintegral(0.)) @ self.z[4:8])
        w = seed[4][2]+.3*(p.wbasis(u) @ self.z[4:8])
        alpha = seed[5][2]+.3*(p.wbasis(u,1) @ self.z[4:8])/self.Tb
        if t >= self.Tb: speed=acc=w=alpha=0.
        return float(speed), float(acc), float(yaw), float(w), float(alpha)

    def base_at(self, t):
        t = canonical(t, self.Tb)
        if t in self.base_cache: return tuple(np.copy(v) if isinstance(v,np.ndarray) else v for v in self.base_cache[t])
        seed = self.base.at(t)
        if not np.any(self.z[:8]): value = seed
        else:
            v, a, yaw, w, alpha = self.scalar(t)
            edges = np.unique(np.r_[0., self.edges[(self.edges>0)&(self.edges<t)], t])
            delta = np.zeros(3)
            for l,r in zip(edges[:-1], edges[1:]):
                ts = l+(r-l)*(self.prep.gnodes+1)/2
                difference = []
                for s in ts:
                    vv, _, yy, _, _ = self.scalar(float(s))
                    difference.append(vv*np.array([np.cos(yy),np.sin(yy),0.])-self.base.at(float(s))[2])
                delta += (r-l)/2*(self.prep.gweights @ np.array(difference))
            heading = np.array([np.cos(yaw),np.sin(yaw),0.]); tangent = np.array([-heading[1],heading[0],0.])
            value = (seed[0]+delta, yaw, v*heading, a*heading+v*w*tangent,
                     np.array([0.,0.,w]), np.array([0.,0.,alpha]))
        self.base_cache[t] = value
        return tuple(np.copy(v) if isinstance(v,np.ndarray) else v for v in value)

    def object_at(self, t):
        t = canonical(t,self.Ta); b = [self.prep.pbasis(t/self.Ta,k)/self.Ta**k for k in range(3)]
        o = self.seed.payload(t)
        dp,dv,da = [.01*(self.z[8:17].reshape(3,3) @ v) for v in b]
        eta,ed,edd = [.1*(self.z[17:26].reshape(3,3) @ v) for v in b]
        R,w,alpha = rotation_motion(eta,ed,edd,o[3],o[4],o[5])
        return o[0]+dp,o[1]+dv,o[2]+da,R,w,alpha

    def nominal_joint(self, t):
        t=canonical(t,self.Ta); seed=self.seed.joint(t)
        return tuple(seed[k]+self.prep.N @ (.1*(self.z[26:].reshape(8,3) @
            (self.prep.pbasis(t/self.Ta,k)/self.Ta**k))) for k in range(3))

    def input(self,t):
        out = ExplicitMapping.input(self,t)
        out['seed'] = self.seed.joint(canonical(t,self.Ta))
        return out


def local_poly(spline, coefficients, left, duration, scale):
    pp=PPoly.from_spline(BSpline(spline.t, scale*(spline.c @ coefficients), spline.k))
    u=left/duration; i=np.searchsorted(pp.x,u,side='right')-1
    i=min(max(i,0),len(pp.x)-2)
    return Poly(pp.c[:,i][::-1])(Poly([u-pp.x[i],1/duration]))


def extrema(poly,width):
    points=[0.,width]
    for root in poly.deriv().roots():
        if abs(root.imag)<1e-10 and 0.<root.real<width: points.append(float(root.real))
    values=poly(points)
    return float(min(values)),float(max(values))


def continuous_limits(p):
    # Merge spline knots with Ruckig phases; they need not coincide.
    edges=np.unique(np.r_[0.,p.A/5.,p.Tb-p.A/5.,p.Tb,
        p.prep.vbasis.t*p.Tb,p.prep.wbasis.t*p.Tb])
    ranges=[[] for _ in range(6)]
    for l,r in zip(edges[:-1],edges[1:]):
        mid=(l+r)/2; _,v,a=p.base.otg.at_time(float(l))
        jerk=-5. if mid<p.A/5. else (0. if mid<p.Tb-p.A/5. else 5.)
        velocity=Poly([v[0],a[0],jerk/2])+local_poly(p.prep.vbasis,p.z[:4],l,p.Tb,.1)
        u=Poly([l/p.Tb,1/p.Tb]); yaw=p.prep.yaw+np.deg2rad(10*p.x[1])*(10*u**3-15*u**4+6*u**5)
        correction=local_poly(p.prep.wbasis,p.z[4:8],l,p.Tb,.3)
        integral=correction.integ()
        yaw += integral+(.3*p.Tb*((p.prep.wintegral(l/p.Tb)-p.prep.wintegral(0.)) @ p.z[4:8]))
        for j,poly in enumerate([velocity,velocity.deriv(),velocity.deriv(2),yaw,yaw.deriv(),yaw.deriv(2)]):
            ranges[j].append(extrema(poly,r-l))
    lo=np.array([min(x[0] for x in q) for q in ranges]); hi=np.array([max(x[1] for x in q) for q in ranges])
    lower=np.array([0.,-1.,-5.,p.prep.yaw-np.deg2rad(10),-p.prep.maxw,-p.prep.maxalpha])
    upper=np.array([p.speed,0.,5.,p.prep.yaw+np.deg2rad(10),p.prep.maxw,p.prep.maxalpha])
    scale=np.array([.4,1.,5.,np.deg2rad(10),p.prep.maxw,p.prep.maxalpha])
    return np.r_[(lo-lower)/scale,(upper-hi)/scale],dict(minimum=lo,maximum=hi)


def coarse_times(p):
    return np.r_[np.linspace(0.,p.Ta,45),0.,p.A/5.,p.Tb-p.A/5.,p.Tb,
                 p.object_A/5.,p.Ta-p.object_A/5.,p.Ta,HORIZON]


def objective(p,order=12):
    nodes,weights=np.polynomial.legendre.leggauss(order)
    edges=np.unique(np.r_[p.edges,HORIZON]); ts=[]; ws=[]
    for l,r in zip(edges[:-1],edges[1:]):
        ts.extend(l+(r-l)*(nodes+1)/2); ws.extend(weights*(r-l)/2/HORIZON)
    acc=np.array([p.ref.payload(float(t))[2] for t in ts])
    return float(np.asarray(ws) @ np.sum(acc*acc,axis=1)),np.array(ts),np.array(ws),acc


def evaluate(prep,budget,x,role,times=None):
    budget.check(complete=True); budget.stats['complete_started']+=1
    number=budget.stats['complete_started']; label=f'eval_{number:04d}'
    dump(DEST/(label+'_started.json'),dict(role=role,parameters=np.asarray(x).tolist()))
    begun=time.perf_counter(); p=GeneralMapping(x,prep,budget,label)
    ts=coarse_times(p) if times is None else np.asarray(times)
    try:
        qs=[]; vs=[]; aa=[]; closures=[]; gaps=[]; penetrations=[]; targets=[]; moments=[]; fronts=[]
        for t in ts:
            q,v,a=p.joint(float(t)); item,te,ce=p.observe(float(t),q,v,a,True)
            qs.append(q); vs.append(v); aa.append(a); closures.append(item['closure'])
            gaps.append(item['gap']); penetrations.append(item['penetration']); targets.append(te)
            moments.append(item['moments'])
            fr=p.ctx.geom.fronts(p.ctx.d); fronts.append([fr[k]['x_m'] for k in ['base','upper','payload']])
            budget.check()
        q,v,a,closure,gap,pen=[np.array(z) for z in [qs,vs,aa,closures,gaps,penetrations]]
        residual=np.sqrt(3)*closure/p.ctx.closure_tol[None,:,None]
        continuous,ranges=continuous_limits(p)
        groups=[('closure_minus',1-residual),('closure_plus',1+residual),
            ('joint_lower',(q-p.m.jnt_range[p.ctx.ids,0])/.1),
            ('joint_upper',(p.m.jnt_range[p.ctx.ids,1]-q)/.1),
            ('joint_speed',1-abs(v)/p.ctx.vcaps),('geometry',gap/.05),
            ('penetration',(1e-5-pen)/1e-4),('target',1-np.array(targets)),
            ('continuous',continuous),('deadline',np.array([(D-p.Ta)/D]))]
        chunks=[]; blocks=[]; offset=0
        for name,value in groups:
            value=np.ravel(value); chunks.append(value)
            blocks.append(dict(name=name,start=offset,stop=offset+len(value),minimum=float(min(value))))
            offset+=len(value)
        g=np.concatenate(chunks); f,ot,ow,oa=objective(p)
        if not np.all(np.isfinite(g)) or not np.isfinite(f): raise RuntimeError('Nonfinite f/g')
        initial=float(abs(p.joint(0.)[0]-prep.q0[prep.qa]).max())
        terminal=float(max(abs(p.joint(p.Ta)[1]).max(),abs(p.joint(p.Ta)[2]).max()))
        if max(initial,terminal)>1e-9: raise RuntimeError('Initial command/terminal boundary')
        unique=sorted(p.kinematics); kin=np.array([p.kinematics[t] for t in unique])
        b=[p.ref.at(t) for t in unique]; o=[p.ref.payload(t) for t in unique]
        arrays=dict(parameters=np.array(x),constraint_times=ts,constraints=g,closure=closure,gap=gap,
            penetration=pen,target_scaled=np.array(targets),predicted_moment=np.array(moments),fronts=np.array(fronts),
            sampled_q=q,sampled_v=v,sampled_a=a,time=np.array(unique),q=kin[:,0],v=kin[:,1],a=kin[:,2],
            nominal=np.array([p.input(t)['nominal'] for t in unique]),N=p.N,
            objective_time=ot,objective_weights=ow,objective_acceleration=oa)
        for name,values,fields in [('base',b,['p','yaw','v','a','w','alpha']),
                                   ('object',o,['p','v','a','R','w','alpha'])]:
            for j,field in enumerate(fields): arrays[name+'_'+field]=np.array([value[j] for value in values])
        path=DEST/(label+'.npz'); np.savez_compressed(path,**arrays)
        packet=dict(number=number,role=role,parameters=np.asarray(x).tolist(),f=f,min_g=float(min(g)),
            qualified=bool(min(g)>=-1e-6),path=str(path),sha256=sha(path),
            T_base_s=p.Tb,T_all_s=p.Ta,min_gap_m=float(np.min(gap)),
            max_penetration_m=float(max(pen)),max_target_scaled=float(max(targets)),
            max_joint_speed_fraction=float(np.max(abs(v)/p.ctx.vcaps)),
            maximum_matrix_condition=max(r['matrix_condition'] for r in p.history),
            maximum_IK_steps=max(r['steps'] for r in p.history),queries=len(unique),
            wall_s=time.perf_counter()-begun,blocks=blocks,continuous_ranges=ranges,
            initial_q_error_rad=initial,terminal_v_a_error=terminal)
        dump(DEST/(label+'.json'),packet)
        budget.stats['complete_completed']+=1; budget.check()
        p.external=None; p.ref=None
        return packet,g
    except Exception as error:
        unique=sorted(p.kinematics)
        if unique:
            np.savez_compressed(DEST/(label+'_partial.npz'),parameters=x,time=unique,
                q=[p.kinematics[t][0] for t in unique],v=[p.kinematics[t][1] for t in unique],
                a=[p.kinematics[t][2] for t in unique])
        dump(DEST/(label+'_failed.json'),dict(role=role,parameters=np.asarray(x).tolist(),
            error=type(error).__name__+': '+str(error),IK_history=p.history,stats=budget.stats))
        raise


def guard():
    c=read(DEST/'contract.json')
    for path,expected in c['source_sha256'].items(): assert sha(Path(path))==expected,path
    return c


def save_progress(state,budget):
    state.update(budget.snapshot()); dump(DEST/'progress.json',state)


def prepare():
    assert not DEST.exists(),'Preserve prior runs'
    DEST.mkdir()
    prior_gate=read(old.DEST/'p2_gate.json'); assert prior_gate['status']=='PASS'
    files=list((ROOT/'simulation/rby1').glob('*.py'))
    for folder in [QUAL,PRIOR,ARCHIVE/'s3-structured-conditioning']:
        files.extend(folder.glob('*.py'))
    files += [INITIAL,PATH_FILE,PATH_AUDIT,ARCHIVE/'model/parameters.json',
        PRIOR/'reference_samples.npz',PRIOR/'result.json',
        ARCHIVE/'s3-structured-conditioning/alternative_velocity_basis.npz',old.DEST/'p2_gate.json',
        SCREEN/'upper_bar/scene.xml']
    for index in [16,1,130]: files += [SCREEN/'upper_bar'/f'row_{index:03d}.json',SCREEN/'upper_bar'/f'reference_{index:03d}.npz']
    for mesh in ET.parse(SCREEN/'upper_bar/scene.xml').getroot().findall('asset/mesh'):
        path=Path(mesh.attrib['file']); files.append(path if path.is_absolute() else SCREEN/'upper_bar'/path)
    c=dict(status='authorized_before_numerical_work',authorization='User selected 按推荐方案实施 for the reviewed formal contract.',
        seed=0,deadline_s=D,common_objective_window_s=HORIZON,variables_B0=3,variables_B1=53,
        coefficient_order='normalized_A, angle_over_10deg, fraction, v4, omega4, COM9, rotation9, null24',
        normalized_parameter_bounds=dict(A=[0.,1.],angle=[-1.,1.],fraction=[0.,1.],shape=[-1.,1.]),
        shape_scales=dict(v_mps=.1,omega_radps=.3,COM_m=.01,rotation_rad=.1,null_rad=.1),
        objective='Mean squared world COM acceleration over common H; no reused old objective quadratic.',
        B0_seeds=SEEDS,phase_time_limits_s=PHASE_LIMITS,maximum_active_seconds=1200,
        count_limits=GLOBAL_LIMITS,maximum_local_steps=30,
        B0_max_calls=3,B0_max_eval_each=120,B0_max_eval=360,B0_maxiter=30,
        B1_max_calls=1,B1_max_eval=1200,B1_maxiter=25,ftol=1e-8,finite_difference_step=1e-6,
        solver='SLSQP, simultaneous cached objective/constraint one-sided differences, bound-aware',
        scalar_limits='Whole-piece extrema; v in [0,current_initial_speed], a in [-1,0], jerk in [-5,5], yaw within10deg; original omega/alpha caps.',
        numerical_admission_min_g=-1e-6,geometry_gap_m=0.,nonpad_penetration_m=1e-5,
        closure_tolerance_SI=read(PRIOR/'result.json')['closure_tolerance_SI'],
        maximum_pilot_evaluations=6,maximum_coarse_constraint_rows_times=53,
        fixed_time_labels='45 uniform normalized [0,T_all] plus 0, base_tj, T_base-base_tj, T_base, object_tj, T_all-object_tj, T_all, H. No row deduplication.',
        maximum_dense_audits=2,maximum_dense_unique_IK_each=900,dense_dt_s=.001,
        scope='Limited offline kinematic reference comparison, same actual initial state/nominal command; no drive/grasp dynamics qualification, global optimum, realtime or hardware claim.',
        initial_policy='No projection or modification of the saved actual state; report existing nominal/actual velocity discrepancy.',
        failure_policy='Nonfinite/mapping/resource errors stop whole batch. Predeclared B0 local nonconvergence is retained; no extra starts. No B1 if no converged and densely admitted B0. One B1, no restart.',
        new_physics_runs=0,new_physics_steps=0,new_dependencies=0,
        source_sha256={str(path.resolve()):sha(path) for path in sorted(set(files))})
    dump(DEST/'contract.json',c); guard()
    state=dict(stage='prepared',failure=None,pilot=[],solves=[])
    budget=Budget(state,'preparepilot'); prep=Preparation(budget); budget.check()
    save_progress(state,budget)
    print(json.dumps(dict(prepared=True,sources=len(c['source_sha256']),active_s=budget.elapsed())))


def pilot(state,budget,prep):
    assert state['stage']=='prepared'
    points=[]
    for _ in range(2): points.append(np.r_[SEEDS[0],np.zeros(50)])
    x=np.zeros(53); x[:3]=[.25,-.01,.02]; points.append(x)
    x=np.zeros(53); x[:3]=[.25,0.,.1]; x[3:11]=1e-5*np.cos(np.arange(8)+1); points.append(x)
    for sign in [1.,-1.]:
        x=np.zeros(53); x[:3]=[.25,0.,.1]; x[11:]=sign*1e-5*np.cos(np.arange(42)+1); points.append(x)
    packets=[]; vectors=[]
    for i,x in enumerate(points):
        packet,g=evaluate(prep,budget,x,'P1_'+str(i)); packets.append(packet); vectors.append(g)
        state['pilot'].append(packet['number']); save_progress(state,budget)
    assert packets[0]['qualified'] and packets[1]['qualified']
    assert packets[0]['f']==packets[1]['f'] and np.array_equal(vectors[0],vectors[1])
    assert all(len(g)==len(vectors[0]) for g in vectors)
    state['stage']='pilot_done'
    dump(DEST/'pilot.json',dict(passed=True,evaluations=packets,zero_f_g_equal=True,
         finite_infeasible_probes_allowed=True,constraint_count=len(vectors[0]),new_physics_steps=0))


class Problem:
    def __init__(self,prep,budget,start,family,maxeval):
        self.prep,self.budget,self.start,self.family=prep,budget,np.array(start),family
        self.n=3 if family=='B0' else 53; self.maxeval=maxeval
        self.cache=OrderedDict(); self.records=[]; self.iterations=[]; self.jac_key=None
        self.lower=np.r_[0.,-1.,0.,-np.ones(50)][:self.n]
        self.upper=np.ones(self.n)
        self.best=None

    def expand(self,x): return np.r_[x,np.zeros(50)] if self.n==3 else np.asarray(x)

    def value(self,x):
        self.budget.stats['requests']+=1; self.budget.check()
        key=np.asarray(x,dtype=float).tobytes()
        if key in self.cache:
            self.budget.stats['cache_hits']+=1; return self.cache[key]
        if len(self.records)>=self.maxeval: raise LimitReached(self.family+' evaluation cap')
        packet,g=evaluate(self.prep,self.budget,self.expand(x),self.family)
        self.records.append(packet); self.cache[key]=(packet,g)
        if len(self.cache)>64: self.cache.popitem(last=False)
        if packet['qualified'] and (self.best is None or packet['f']<self.best['f']): self.best=packet
        dump(DEST/'optimizer_progress.json',dict(family=self.family,evals=len(self.records),
             iterations=self.iterations,best=self.best,active_s=self.budget.elapsed(),stats=self.budget.stats))
        return packet,g

    def fun(self,x): return self.value(x)[0]['f']
    def constraint(self,x): return self.value(x)[1]

    def derivatives(self,x):
        key=np.asarray(x).tobytes()
        if key==self.jac_key: return self.df,self.dg
        packet,g=self.value(x); df=np.empty(self.n); dg=np.empty((len(g),self.n))
        for j in range(self.n):
            step=1e-6 if x[j]+1e-6<=self.upper[j] else -1e-6
            y=np.array(x,copy=True); y[j]+=step
            other,gg=self.value(y); df[j]=(other['f']-packet['f'])/step; dg[:,j]=(gg-g)/step
        if not np.all(np.isfinite(df)) or not np.all(np.isfinite(dg)): raise RuntimeError('Nonfinite Jacobian')
        self.budget.stats['jacobians']+=1
        np.savez_compressed(DEST/(self.family+f'_jac_{self.budget.stats["jacobians"]:03d}.npz'),x=x,df=df,dg=dg)
        self.jac_key,self.df,self.dg=key,df,dg
        return df,dg

    def callback(self,x):
        packet,_=self.value(x)
        self.iterations.append(dict(f=packet['f'],min_g=packet['min_g'],parameters=self.expand(x).tolist()))


def solve(prep,budget,start,family,index):
    budget.check(outer=True); budget.stats['outer_calls']+=1
    problem=Problem(prep,budget,start,family,120 if family=='B0' else 1200)
    first=None; failure=None; result=None; final=None; critical=None
    try:
        first,_=problem.value(start)
        result=minimize(problem.fun,start,method='SLSQP',jac=lambda x:problem.derivatives(x)[0],
            bounds=list(zip(problem.lower,problem.upper)),
            constraints=dict(type='ineq',fun=problem.constraint,jac=lambda x:problem.derivatives(x)[1]),
            callback=problem.callback,options=dict(maxiter=30 if family=='B0' else 25,ftol=1e-8))
        final,_=problem.value(result.x)
    except Exception as e:
        failure=type(e).__name__+': '+str(e); final=None
        # A per-start evaluation limit is a retained local nonconvergence.
        if not isinstance(e,LimitReached) or str(e)!=family+' evaluation cap': critical=e
    record=dict(family=family,index=index,success=bool(result is not None and result.success),failure=failure,
        message=None if result is None else str(result.message),initial=first,final=final,
        best_feasible=problem.best,evaluations=problem.records,iterations=problem.iterations)
    dump(DEST/(family+f'_solve_{index}.json'),record)
    if critical is not None: raise critical
    return record


def dense_schedule(p):
    phases=coarse_times(p)[45:-1]
    possible=np.linspace(.04,.96,41)*p.Ta
    valid=[t for t in possible if min(abs(t-s) for s in phases)>.0011]
    assert len(valid)>=6
    centers=[canonical(valid[i],p.Ta) for i in np.linspace(0,len(valid)-1,6).round().astype(int)]
    extra=[t+d for t in centers for d in [0.,-.001,-.0005,.0005,.001]]
    times=sorted(set(canonical(t,p.Ta) for t in np.r_[np.linspace(0,p.Ta,int(np.ceil(p.Ta/.001))+1),
        phases,np.linspace(0,p.Ta,31),extra]))
    assert len(times)<=900
    return np.array(times),centers


def difference_checks(p,arrays,centers):
    ix={canonical(t,p.Ta):i for i,t in enumerate(arrays['time'])}; reports=[]
    for t in centers:
        i=ix[t]; q,v,a=arrays['q'][i],arrays['v'][i],arrays['a'][i]
        for h in [.001,.0005]:
            im,ip=ix[canonical(t-h,p.Ta)],ix[canonical(t+h,p.Ta)]
            qm,qp=arrays['q'][im],arrays['q'][ip]
            vf=(qp-qm)/(2*h); af=(qp-2*q+qm)/h**2
            _,target,closure=p.observe(t,q,vf,af)
            nm,nc,np_=arrays['nominal'][[im,i,ip],0]
            nv=float(abs(p.N.T @ (vf-(np_-nm)/(2*h))).max())
            na=float(abs(p.N.T @ (af-(np_-2*nc+nm)/h**2)).max())
            scale=64*np.finfo(float).eps*(1+max(abs(q).max(),abs(nc).max()))*np.linalg.norm(p.N.T,ord=np.inf)
            vb,ab=2e-13/h+scale/h,4e-13/h**2+scale/h**2
            def error(prefix,field,order):
                pos=arrays[prefix+'_p']; est=(pos[ip]-pos[im])/(2*h) if order==1 else (pos[ip]-2*pos[i]+pos[im])/h**2
                return float(abs(est-arrays[prefix+'_'+field][i]).max())
            bv,ba=error('base','v',1),error('base','a',2)
            ov,oa=error('object','v',1),error('object','a',2)
            Rm,R,Rp=arrays['object_R'][[im,i,ip]]; w=arrays['object_w'][i]
            we=vee((Rp-Rm)/(2*h) @ R.T); ae=vee((Rp-2*R+Rm)/h**2 @ R.T-skew(w) @ skew(w))
            ratios=(np.sqrt(3)*np.array([bv,ba,ov,oa,max(abs(we-w)),max(abs(ae-arrays['object_alpha'][i]))]) /
                    p.ctx.closure_tol[[2,4,2,4,3,5]])
            row=dict(t_s=t,h_s=h,qdot_error=float(abs(vf-v).max()),qddot_error=float(abs(af-a).max()),
                target_scaled=target,closure_scaled=closure,base_object_FD_scaled=float(max(ratios)),
                null_v=nv,null_a=na,null_v_bound=vb,null_a_bound=ab)
            row['passed']=bool(max(target,closure,max(ratios))<=1+1e-6 and nv<=vb and na<=ab)
            reports.append(row)
    return reports


def dense_audit(prep,budget,packet,family):
    budget.switch('dense'); p=GeneralMapping(packet['parameters'],prep,budget,'dense_schedule')
    times,centers=dense_schedule(p)
    dense,g=evaluate(prep,budget,packet['parameters'],family+'_dense',times)
    with np.load(dense['path']) as z: arrays=dict(z)
    p.kinematics={float(t):(arrays['q'][i],arrays['v'][i],arrays['a'][i]) for i,t in enumerate(arrays['time'])}
    count=budget.stats['IK_calls']; differences=difference_checks(p,arrays,centers)
    schema=validate(p.external,p.m,prep.q0)
    f24,_,_,_=objective(p,24)
    assert budget.stats['IK_calls']==count,'Dense audit must reuse declared nodes'
    angular=np.array([np.linalg.norm(old.rotvec(R @ arrays['object_R'][0].T)) for R in arrays['object_R']])
    report=dict(family=family,passed=bool(dense['qualified'] and all(d['passed'] for d in differences)
        and abs(f24-dense['f'])<=1e-8),packet=dense,schema=schema,differences=differences,
        quadrature_12_24_error=abs(f24-dense['f']),reference_only=True,
        metrics=dict(rms_acceleration_mps2=np.sqrt(dense['f']),
            peak_acceleration_mps2=float(np.max(np.linalg.norm(arrays['object_a'],axis=1))),
            peak_equal_TCP_force_N=float(np.max(np.linalg.norm(p.m.body_mass[p.ctx.ob]*(arrays['object_a']-p.m.opt.gravity),axis=1)))/2,
            peak_equal_TCP_moment_Nm=float(np.max(arrays['predicted_moment'])),
            base_path_m=float(np.trapezoid(np.linalg.norm(arrays['base_v'],axis=1),arrays['time'])),
            object_path_m=float(np.trapezoid(np.linalg.norm(arrays['object_v'],axis=1),arrays['time'])),
            object_max_forward_m=float(max(arrays['object_p'][:,0]-arrays['object_p'][0,0])),
            whole_front_max_forward_m=float(max(np.max(arrays['fronts'],axis=1))-np.max(arrays['fronts'][0])),
            maximum_object_angle_rad=float(max(angular)),
            peak_object_omega_radps=float(np.max(np.linalg.norm(arrays['object_w'],axis=1))),
            peak_object_alpha_radps2=float(np.max(np.linalg.norm(arrays['object_alpha'],axis=1))),
            upper_excursion_norm_rad=float(np.max(np.linalg.norm(arrays['q']-arrays['q'][0],axis=1))),
            maximum_joint_speed_fraction=dense['max_joint_speed_fraction'],minimum_gap_m=dense['min_gap_m']),
        initial_upper_velocity_vs_actual_radps=float(abs(arrays['v'][0]-prep.v0[prep.va]).max()))
    dump(DEST/(family+'_dense.json'),report); budget.check()
    if not report['passed']: raise RuntimeError(family+' final dense qualification failed')
    return report


def b0(state,budget,prep):
    assert state['stage']=='pilot_done'
    gate=read(DEST/'p1_gate.json'); assert gate['status']=='PASS' and gate['contract_sha256']==sha(DEST/'contract.json')
    assert budget.stats['outer_calls']==0
    for i,seed in enumerate(SEEDS):
        result=solve(prep,budget,seed,'B0',i); state['solves'].append('B0_solve_'+str(i)+'.json')
        save_progress(state,budget)
    candidates=[]
    for filename in state['solves']:
        result=read(DEST/filename)
        if result['success'] and result['final']['qualified']: candidates.append(result['final'])
    if not candidates:
        state.update(stage='complete',stop_reason='No normally converged qualified B0; B1 not started')
        return
    best=min(candidates,key=lambda p:p['f'])
    state['B0']=dense_audit(prep,budget,best,'B0'); state['stage']='b0_done'


def b1(state,budget,prep):
    assert state['stage']=='b0_done' and state['B0']['passed']
    assert budget.stats['outer_calls']==3
    start=state['B0']['packet']['parameters']
    result=solve(prep,budget,start,'B1',0); state['solves'].append('B1_solve_0.json')
    if result['success'] and result['final']['qualified']:
        state['B1']=dense_audit(prep,budget,result['final'],'B1')
        decrease=state['B0']['packet']['f']-state['B1']['packet']['f']
        threshold=max(1e-8,10*max(state[k]['quadrature_12_24_error'] for k in ['B0','B1']))
        state['comparison']=dict(objective_decrease=decrease,numerical_threshold=threshold,
            decrease_resolved=bool(decrease>threshold),relative_RMS_change=np.sqrt(state['B1']['packet']['f']/state['B0']['packet']['f'])-1,
            scope='This deadline, model, local search box and budget only; no overall dominance or physical improvement claim.')
    else: state['stop_reason']='B1 did not normally converge to a qualified reference; no dense or physics'
    state['stage']='complete'


def execute(action):
    guard(); state=read(DEST/'progress.json'); assert state['failure'] is None,'Failed batch cannot resume'
    budget=Budget(state,'preparepilot' if action=='pilot' else action)
    try:
        prep=Preparation(budget)
        dict(pilot=pilot,b0=b0,b1=b1)[action](state,budget,prep)
    except Exception as e:
        state.update(failure=type(e).__name__+': '+str(e),stage='stopped')
    save_progress(state,budget)
    if state['stage'] in ['complete','stopped']:
        dump(DEST/'result.json',dict(**state,contract_sha256=sha(DEST/'contract.json'),
            new_physics_runs=0,new_physics_steps=0,comparison_completed='comparison' in state))
    guard()
    print(json.dumps(dict(stage=state['stage'],failure=state['failure'],stats=budget.stats,active_s=budget.elapsed(),
                         stop_reason=state.get('stop_reason'))))
    return state['failure'] is None


def verify():
    c=guard(); state=read(DEST/'progress.json'); packets=[]
    for path in sorted(DEST.glob('eval_*.json')):
        if path.stem.endswith(('_started','_failed')): continue
        packet=read(path)
        assert sha(Path(packet['path']))==packet['sha256']
        with np.load(packet['path']) as z:
            assert np.all(np.isfinite(z['constraints']))
            f=float(z['objective_weights'] @ np.sum(z['objective_acceleration']**2,axis=1))
            assert abs(f-packet['f'])<1e-12 and float(min(z['constraints']))==packet['min_g']
            assert bool(packet['min_g']>=-1e-6)==packet['qualified']
            assert len(z['time'])==packet['queries'] and len(z['time'])<=900
        packets.append(packet)
    assert len(packets)==state['stats']['complete_completed']
    assert sum(p['queries'] for p in packets)<=state['stats']['IK_calls']
    for k,limit in GLOBAL_LIMITS.items(): assert state['stats'][k]<=limit
    assert state['active_s']<1200.01
    # Read-only analytic checks do not regenerate mapping solutions.
    budget=Budget({},'preparepilot'); prep=Preparation(budget)
    check=[p for p in packets if p['role'].startswith('P1_') or p['role'].endswith('_dense')]
    for packet in check:
        p=GeneralMapping(packet['parameters'],prep,budget,'audit')
        with np.load(packet['path']) as z: arrays=dict(z)
        for i in sorted(set([0,len(arrays['time'])//2,len(arrays['time'])-1])):
            q,v,a=arrays['q'][i],arrays['v'][i],arrays['a'][i]; t=float(arrays['time'][i])
            item,target,closure=p.observe(t,q,v,a,True)
            assert max(target,closure)<=1+1e-6
        if packet['role'].endswith('_dense'):
            family=packet['role'].split('_')[0]; saved=read(DEST/(family+'_dense.json'))
            centers=[r['t_s'] for r in saved['differences'][::2]]
            assert difference_checks(p,arrays,centers)==saved['differences']
            assert saved['passed']
        p.ref=None; p.external=None
    assert budget.stats['IK_calls']==budget.stats['QP_calls']==0
    print(json.dumps(dict(verified=True,stage=state['stage'],failure=state['failure'],
        completed_evaluations=len(packets),saved_stats=state['stats'],active_s=state['active_s'],
        additional_audit_IK=0,additional_audit_QP=0,new_physics_steps=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','pilot','b0','b1','verify'])
    action=parser.parse_args().action
    if action=='prepare': prepare()
    elif action=='verify': verify()
    else: sys.exit(0 if execute(action) else 1)
