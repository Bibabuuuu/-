"""Offline dual-hand Jacobian QP paths, with optional torso participation.

One shared rigid-object target defines both hand poses. This is a local IK
allocator and checked interpolation, not an online WBC or collision planner.
"""
import json,time
import numpy as np
import mujoco as mj
from build_model import OUT
from build_payload import rotvec
from buffer_reference import evaluate
from load_sharing import feasible_qp


def nonpad_penetration(m,d):
    ob=m.body('payload').id;peak=0.
    for c in d.contact:
        a,b=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
        if a and b and c.dist<0:
            allowed=ob in (a,b) and m.geom(c.geom2 if a==ob else c.geom1).name.endswith('_pad')
            if not allowed:peak=max(peak,-float(c.dist))
    return peak


def joint_indices(m,include_torso):
    names=([f'torso_{i}' for i in range(6)] if include_torso else [])+[f'{s}_arm_{i}' for s in ['left','right'] for i in range(7)]
    ids=np.array([m.joint(n).id for n in names]);return names,ids,m.jnt_qposadr[ids],m.jnt_dofadr[ids]


def hand_system(m,d,sites,targets,vadr):
    jp=np.zeros((3,m.nv));jr=jp.copy();J=[];errors=[]
    for s,site in sites.items():
        pos,R=targets[s];mj.mj_jacSite(m,d,jp,jr,site)
        J.append(np.vstack([jp[:,vadr],.25*jr[:,vadr]]))
        errors.append(np.r_[pos-d.site_xpos[site],.25*rotvec(R@d.site_xmat[site].reshape(3,3).T)])
    return np.vstack(J),np.concatenate(errors)


def motion_path(m,initial,extent=.09,backward_extent=.01,include_torso=True):
    started=time.perf_counter();d=mj.MjData(m);d.qpos[:]=initial;mj.mj_forward(m,d)
    names,ids,qa,va=joint_indices(m,include_torso);sites={s:m.site(f'ee_{s}_site').id for s in ['left','right']}
    ob=m.body('payload').id;pq=int(m.joint('payload_free').qposadr[0]);Robj=d.xmat[ob].reshape(3,3).copy();pobj=d.xpos[ob].copy()
    grasp={s:(Robj.T@(d.site_xpos[i]-pobj),Robj.T@d.site_xmat[i].reshape(3,3)) for s,i in sites.items()}
    vcap={c['joint']:c['velocity_limit'] for c in json.loads((OUT/'parameters.json').read_text())['capabilities']}
    penalty=np.diag([1/vcap[n]**2 for n in names]);I=np.eye(len(names));C=np.vstack([I,-I])
    stats={'qp_calls':0,'max_qp_constraint_violation':0.,'max_qp_stationarity_error':0.,'line_search_reductions':0,'max_ik_position_error_m':0.,'max_ik_angle_error_rad':0.}
    def targets(offset):return {s:(pobj+[offset,0,0]+Robj@p,Robj@R) for s,(p,R) in grasp.items()}
    def solve(offset,seed):
        d.qpos[:]=initial;d.qpos[qa]=seed;d.qpos[pq]+=offset;mj.mj_forward(m,d);task=targets(offset)
        for iteration in range(80):
            J,e=hand_system(m,d,sites,task,va)
            pos=max(np.linalg.norm(e[k:k+3]) for k in [0,6]);angle=max(np.linalg.norm(e[k+3:k+6])/.25 for k in [0,6])
            if pos<1e-7 and angle<1e-6:break
            q=d.qpos[qa].copy();lo=np.maximum(m.jnt_range[ids,0]-q,-.12);hi=np.minimum(m.jnt_range[ids,1]-q,.12)
            H=J.T@J+1e-5*penalty;g=-J.T@e
            dq,info=feasible_qp(H,g,C,np.r_[hi,-lo]);stats['qp_calls']+=1
            if not info['accepted']:raise RuntimeError(('motion QP failed',offset,info))
            stats['max_qp_constraint_violation']=max(stats['max_qp_constraint_violation'],info['max_constraint_violation'])
            stats['max_qp_stationarity_error']=max(stats['max_qp_stationarity_error'],info['stationarity_error'])
            # ponytail: local backtracking rejects collision; it cannot find a route around an obstacle.
            accepted=False
            for power in range(12):
                d.qpos[qa]=q+dq*2.**(-power);mj.mj_forward(m,d)
                _,candidate_error=hand_system(m,d,sites,task,va)
                if nonpad_penetration(m,d)<1e-5 and np.linalg.norm(candidate_error)<np.linalg.norm(e):
                    stats['line_search_reductions']+=power;accepted=True;break
            if not accepted:raise RuntimeError(('No collision-free decreasing local IK step',offset,iteration))
        else:raise RuntimeError(('Local motion IK iteration guard',offset,pos,angle))
        stats['max_ik_position_error_m']=max(stats['max_ik_position_error_m'],float(pos))
        stats['max_ik_angle_error_rad']=max(stats['max_ik_angle_error_rad'],float(angle))
        return d.qpos[qa].copy()
    points={0.:np.array(initial)[qa].copy()}
    for end in [-backward_extent,extent]:
        seed=points[0.].copy()
        for x in np.linspace(0,end,41)[1:]:seed=solve(float(x),seed);points[float(x)]=seed.copy()
    xs=np.array(sorted(points));q=np.array([points[float(x)] for x in xs]);length=extent+backward_extent
    assert length>0 and extent>0 and backward_extent>=0
    # Fit through the actual initial joint positions, so static initialization is unchanged.
    z=xs/length;delta=q-np.array(initial)[qa];powers=np.column_stack([z**i for i in range(1,8)])
    shifted=np.vstack([np.array(initial)[qa],np.linalg.lstsq(powers,delta,rcond=None)[0]])
    u0=backward_extent/length;coeff=np.zeros_like(shifted)
    for j in range(len(names)):
        p=np.polynomial.Polynomial(shifted[:,j])(np.polynomial.Polynomial([-u0,1.]))
        coeff[:len(p.coef),j]=p.coef
    path={'qadr':qa,'vadr':va,'coeff':coeff,'first':np.polynomial.polynomial.polyder(coeff),
          'second':np.polynomial.polynomial.polyder(coeff,2),'extent':extent,'backward_extent':backward_extent,'names':names}
    maxpos=maxangle=maxpen=0.;margin=float('inf')
    for x in np.linspace(-backward_extent,extent,161):
        d.qpos[:]=initial;d.qpos[qa]=evaluate(path,float(x),0.,0.)[0];d.qpos[pq]+=x;mj.mj_forward(m,d)
        task=targets(float(x))
        for s,i in sites.items():
            p,R=task[s];maxpos=max(maxpos,float(np.linalg.norm(p-d.site_xpos[i])));maxangle=max(maxangle,float(np.linalg.norm(rotvec(R@d.site_xmat[i].reshape(3,3).T))))
        margin=min(margin,float(np.min(np.minimum(d.qpos[qa]-m.jnt_range[ids,0],m.jnt_range[ids,1]-d.qpos[qa]))))
        maxpen=max(maxpen,nonpad_penetration(m,d))
    assert maxpos<1e-4 and maxangle<1e-3 and maxpen<1e-5 and margin>0,(maxpos,maxangle,maxpen,margin)
    path['audit']={**stats,'method':'shared-object Jacobian local QP, offline path, degree-7 interpolation',
                   'include_torso':include_torso,'active_joint_names':names,'extent_m':extent,'backward_extent_m':backward_extent,
                   'polynomial_max_position_error_m':maxpos,'polynomial_max_angle_error_rad':maxangle,'nonpayload_self_penetration_m':maxpen,
                   'joint_range_min_margin_rad':margin,'validation_points':161,'preparation_wall_ms':1000*(time.perf_counter()-started),
                   'grasp_transforms_object_to_hands':{s:{'position_m':p.tolist(),'rotation':R.tolist()} for s,(p,R) in grasp.items()},
                   'objective':'Task pose residual plus 1e-5 times joint-increment norm normalized by model joint-speed caps; same weights for both groups.',
                   'numerical_trust_step_rad':.12,'physical_speed_check':'Performed on the timed stop reference and actual execution, not inferred from IK iteration steps.',
                   'collision_method':'Nonlinear FK at accepted IK steps and independent interpolation grid; no continuous collision certificate or global route search.',
                   'reference_data_is_separate_from_physics':True}
    return path
