"""One-dimensional reference admission using local IK and explicit space budgets.

This is not a global reachability solver or a dynamic safety certificate.
"""
import json
from pathlib import Path
import mujoco as mj
import numpy as np
from build_model import HERE,OUT,dump
from build_payload import PAY,solve_ik
from buffer_reference import arm_path,evaluate,references

DEST=OUT.parent/'governor'


def make_initial(x):
    m=mj.MjModel.from_xml_path(str(PAY/'payload_scene.xml'));d=mj.MjData(m)
    initial=json.loads((PAY/'initialization.json').read_text());d.qpos[:]=initial['qpos'];mj.mj_forward(m,d)
    ob=m.body('payload').id;delta=x-float(d.xpos[ob,0]);refs={s:(d.site_xpos[m.site(f'ee_{s}_site').id].copy(),d.site_xmat[m.site(f'ee_{s}_site').id].reshape(3,3).copy()) for s in ['left','right']}
    d.qpos[m.joint('payload_free').qposadr[0]]+=delta
    ik=[solve_ik(m,d,s,p+[delta,0,0],R) for s,(p,R) in refs.items()]
    assert all(r['position_error_m']<1e-5 and r['angle_error_rad']<1e-4 for r in ik)
    initial['qpos']=d.qpos.tolist();initial['ik']=ik;initial['governor_initial_payload_x_m']=x
    return m,initial


def local_reach(m,initial):
    d=mj.MjData(m);d.qpos[:]=initial['qpos'];mj.mj_forward(m,d);ob=m.body('payload').id
    sites={s:m.site(f'ee_{s}_site').id for s in ['left','right']}
    refs={s:(d.site_xpos[i].copy(),d.site_xmat[i].reshape(3,3).copy()) for s,i in sites.items()}
    pq=int(m.joint('payload_free').qposadr[0]);rows=[];last=0.
    for offset in np.linspace(0,.12,25):
        result=[solve_ik(m,d,s,p+[offset,0,0],R) for s,(p,R) in refs.items()]
        d.qpos[pq]=initial['qpos'][pq]+offset;mj.mj_forward(m,d)
        penetration=0.
        for c in d.contact:
            a,b=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
            if not a or not b:continue
            pad=ob in (a,b) and m.geom(c.geom2 if a==ob else c.geom1).name.endswith('_pad')
            if not pad:penetration=max(penetration,-float(c.dist))
        ok=all(r['position_error_m']<1e-5 and r['angle_error_rad']<1e-4 for r in result) and penetration<1e-5
        rows.append({'offset_m':float(offset),'passed':ok,'max_position_error_m':max(r['position_error_m'] for r in result),'nonpad_penetration_m':penetration})
        if not ok:break
        last=float(offset)
    return {'checked_forward_lower_bound_m':last,'rows':rows,'scope':'Continuous seeded IK branch and sampled collision check; failure is not proof of global unreachability.'}


def select(x,clearance,*,desired_object_T=1.,space_reserve=.01,reach_reserve=.005):
    m,initial=make_initial(x);reach=local_reach(m,initial)
    v=.4;Tb=.6;db=v*Tb/2;desired_s=v*(desired_object_T-Tb)/2
    reach_s=max(0.,reach['checked_forward_lower_bound_m']-reach_reserve)
    space_s=clearance-space_reserve-db
    decision={'initial_x_m':x,'front_clearance_budget_m':clearance,'space_reserve_m':space_reserve,'reach_reserve_m':reach_reserve,
        'desired_object_T_s':desired_object_T,'base_T_s':Tb,'base_reference_distance_m':db,'desired_extra_stroke_m':desired_s,
        'reach':reach,'space_limited_extra_stroke_m':space_s,'reach_limited_extra_stroke_m':reach_s,
        'qualification':'Candidate within one smooth-stop family and a local IK branch, followed by physical validation. Reserves are explicit engineering assumptions, not certified uncertainty bounds.'}
    if space_s< -1e-10:
        decision.update({'status':'reject_template','reason':'This base-stop reference already exceeds the front-space budget after the selected reserve. No global impossibility claim.'})
        return decision,initial
    stroke=max(0.,min(desired_s,reach_s,space_s));original=stroke;velocity_limits={c['joint']:c['velocity_limit'] for c in json.loads((OUT/'parameters.json').read_text())['capabilities']}
    attempts=[]
    while True:
        T=Tb+2*stroke/v;extent=max(stroke,.001)
        try:
            path=arm_path(m,initial['qpos'],extent=extent);speed_ratio=0.
            for t in np.linspace(3.5,3.5+T,301):
                b,o=references(t,v,Tb,T);q,qd,qdd=evaluate(path,o[0]-b[0],o[1]-b[1],o[2]-b[2])
                names=[m.joint(int(m.dof_jntid[vj])).name for vj in path['vadr']]
                speed_ratio=max(speed_ratio,max(abs(qd[i])/velocity_limits[n] for i,n in enumerate(names)))
            accepted=speed_ratio<=1.
            attempts.append({'stroke_m':stroke,'kinematic_passed':True,'peak_reference_joint_speed_fraction':speed_ratio})
        except AssertionError as e:
            accepted=False;attempts.append({'stroke_m':stroke,'kinematic_passed':False,'error':str(e)})
        if accepted:break
        if stroke<=1e-9:
            decision.update({'status':'reject_template','reason':'Even the zero-stroke reference failed local reference checks.','reference_attempts':attempts})
            return decision,initial
        stroke=max(0.,stroke-.005)
    decision.update({'status':'full_buffer' if stroke>=desired_s-1e-8 else 'partial_buffer' if stroke>1e-6 else 'hold',
        'selected_extra_stroke_m':stroke,'selected_object_T_s':T,'reference_peak_object_acceleration_mps2':1.5*v/T,
        'reference_object_translation_m':v*T/2,'path_extent_m':extent,'reference_speed_fraction':speed_ratio,
        'soft_performance_goal_met':T>=desired_object_T-1e-8,'pre_reference_check_stroke_m':original,'reference_attempts':attempts,
        'reference_audit':path['audit']})
    return decision,initial


def main():
    DEST.mkdir(exist_ok=True);decisions=[]
    for label,x,space in [('roomy',.36,.22),('tight',.36,.15),('extended',.44,.22),('near_limit',.46,.22),('too_tight',.36,.10)]:
        decision,initial=select(x,space);case=DEST/label;case.mkdir(exist_ok=True)
        decision['label']=label;decision['initialization_path']=str((case/'initialization.json').resolve());dump(case/'initialization.json',initial);dump(case/'decision.json',decision)
        decisions.append(decision)
        print(json.dumps({k:decision.get(k) for k in ['label','status','selected_extra_stroke_m','selected_object_T_s','reference_peak_object_acceleration_mps2','soft_performance_goal_met']}),flush=True)
    dump(DEST/'decisions.json',{'cases':decisions})


if __name__=='__main__':main()
