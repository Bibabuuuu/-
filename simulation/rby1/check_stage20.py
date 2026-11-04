"""Independent finite-difference, FK, closure, collision and QP checks."""
import json
import numpy as np
import mujoco as mj
from build_model import OUT,dump
from build_payload import PAY,rotvec
from buffer_reference import evaluate
from motion_distribution import motion_path,joint_indices,nonpad_penetration
from load_sharing import feasible_qp
from state_stop_reference import StateStop
from check_stage15 import observe


def main():
    dest=OUT.parent/'stage20';dest.mkdir(exist_ok=True)
    initial=np.array(json.loads((OUT.parent/'buffer_preflight/initialization.json').read_text())['qpos'])
    m=mj.MjModel.from_xml_path(str(PAY/'payload_scene.xml'));d=mj.MjData(m);d.qpos[:]=initial;mj.mj_forward(m,d)
    untouched=d.qpos.copy();pos0={s:d.site_xpos[m.site(f'ee_{s}_site').id].copy() for s in ['left','right']}
    R0={s:d.site_xmat[m.site(f'ee_{s}_site').id].reshape(3,3).copy() for s in pos0}
    q0=np.array(initial);pq=int(m.joint('payload_free').qposadr[0]);results={};admissions={}
    H=np.diag(np.linspace(1.,2.,20));g=np.linspace(-3.,3.,20);C=np.vstack([np.eye(20),-np.eye(20)]);rhs=np.ones(40)*.2
    answer,info=feasible_qp(H,g,C,rhs);error=float(max(abs(answer-np.clip(-g/np.diag(H),-.2,.2))))
    assert info['accepted'] and error<1e-9 and info['stationarity_error']<1e-7
    for include in [False,True]:
        d.qpos[:]=initial;d.qvel[:]=0.;mj.mj_forward(m,d)
        name='waist_arms' if include else 'arms';path=motion_path(m,initial,include_torso=include)
        assert np.array_equal(d.qpos,untouched),'Reference preparation must not write live data'
        assert max(abs(evaluate(path,0.,0.,0.)[0]-initial[path['qadr']]))<1e-12
        names,ids,qa,va=joint_indices(m,include);jacerr=0.;h=1e-6
        for x in [0.,.045,.08]:
            probe=mj.MjData(m);probe.qpos[:]=initial;probe.qpos[qa]=evaluate(path,x,0.,0.)[0];mj.mj_forward(m,probe)
            for s in pos0:
                site=m.site(f'ee_{s}_site').id;jp=np.zeros((3,m.nv));jr=jp.copy();mj.mj_jacSite(m,probe,jp,jr,site)
                for col,adr in enumerate(qa):
                    ps=[];Rs=[]
                    for sign in [-1,1]:
                        finite=mj.MjData(m);finite.qpos[:]=probe.qpos;finite.qpos[adr]+=sign*h;mj.mj_forward(m,finite)
                        ps.append(finite.site_xpos[site].copy());Rs.append(finite.site_xmat[site].reshape(3,3).copy())
                    numerical=np.r_[(ps[1]-ps[0])/(2*h),rotvec(Rs[1]@Rs[0].T)/(2*h)]
                    jacerr=max(jacerr,float(max(abs(numerical-np.r_[jp[:,va[col]],jr[:,va[col]]]))))
        assert jacerr<1e-7,jacerr
        poserr=angleerr=pen=closure=0.;derivative=second=0.;margin=float('inf')
        for x in np.linspace(-.01,.09,801):
            probe=mj.MjData(m);probe.qpos[:]=initial;q,v,a=evaluate(path,float(x),1.,0.);probe.qpos[qa]=q;probe.qpos[pq]+=x;mj.mj_forward(m,probe)
            margin=min(margin,float(np.min(np.minimum(q-m.jnt_range[ids,0],m.jnt_range[ids,1]-q))))
            for s in pos0:
                site=m.site(f'ee_{s}_site').id;poserr=max(poserr,float(np.linalg.norm(probe.site_xpos[site]-pos0[s]-[x,0,0])))
                angleerr=max(angleerr,float(np.linalg.norm(rotvec(probe.site_xmat[site].reshape(3,3)@R0[s].T))))
            left=m.site('ee_left_site').id;right=m.site('ee_right_site').id
            closure=max(closure,float(np.linalg.norm(probe.site_xpos[left]-probe.site_xpos[right]-(pos0['left']-pos0['right']))))
            pen=max(pen,nonpad_penetration(m,probe))
            if -.009<x<.089:
                step=1e-5;lo=evaluate(path,float(x-step),0.,0.)[0];hi=evaluate(path,float(x+step),0.,0.)[0]
                derivative=max(derivative,float(max(abs((hi-lo)/(2*step)-v))))
                second=max(second,float(max(abs((hi-2*q+lo)/step**2-a))))
        assert poserr<1e-4 and angleerr<1e-3 and pen<1e-5 and margin>0 and closure<2e-4
        assert derivative<1e-5 and second<2e-4,(derivative,second)
        results[name]={'path_audit':path['audit'],'independent_grid_points':801,'fk_max_position_error_m':poserr,'fk_max_angle_error_rad':angleerr,
                       'closure_position_error_m':closure,'nonpad_penetration_m':pen,'joint_range_margin_rad':margin,
                       'jacobian_finite_difference_max_error':jacerr,'first_derivative_error':derivative,'second_derivative_error':second}
        admissions[name]={}
        for speed in [.4,.5]:
            d.qpos[:]=initial;d.qvel[:]=0.;mj.mj_forward(m,d);ref=StateStop(speed=speed,seed_policy='command',batched=True)
            ref.prepare(m,d,path);observe(ref,m,d,initial);assert ref.audit['preferred_plan_admitted'],ref.audit
            admissions[name][str(speed)]=dict(ref.audit)
    result={'qp_20d_analytic_box_error':error,'qp_kkt':info,'kinematic_checks':results,'reference_admissions':admissions,
            'physics_state_not_written_by_path_preparation':True,'ready_for_physical_comparison':True}
    dump(dest/'independent_checks.json',result);print(json.dumps({k:{n:v[n] for n in ['fk_max_position_error_m','closure_position_error_m','jacobian_finite_difference_max_error']} for k,v in results.items()}))


if __name__=='__main__':main()
