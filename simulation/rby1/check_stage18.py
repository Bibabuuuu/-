"""Audit a nominal combination before treating it as a new coordination problem."""
import json,math
import numpy as np
import mujoco as mj
from build_model import OUT,dump
from build_payload import PAY,solve_ik
from check_stage17 import collisions
from buffer_reference import arm_path,stop
from state_stop_reference import make_otg,LegacyStateStop as StateStop
from check_stage15 import observe

DEST=OUT.parent/'stage18'


def main():
    DEST.mkdir(exist_ok=True);src=OUT.parent/'stage17/initial_x_0.44.json';(DEST/'initialization.json').write_bytes(src.read_bytes())
    invalid=[{'object_minimum_duration_s':-1.},{'object_minimum_duration_s':float('nan')},{'strategy':'fixed','object_minimum_duration_s':0.}]
    for options in invalid:
        try:StateStop(**options)
        except AssertionError:pass
        else:raise AssertionError(options)
    initial=json.loads(src.read_text());q0=np.array(initial['qpos']);m=mj.MjModel.from_xml_path(str(PAY/'payload_scene.xml'));d=mj.MjData(m)
    d.qpos[:]=q0;mj.mj_forward(m,d);pq=int(m.joint('payload_free').qposadr[0]);px=float(d.qpos[pq])
    targets={side:(d.site_xpos[m.site(f'ee_{side}_site').id].copy(),d.site_xmat[m.site(f'ee_{side}_site').id].reshape(3,3).copy()) for side in ['left','right']}
    joints=np.array([m.joint(f'{side}_arm_{i}').id for side in targets for i in range(7)]);qa=m.jnt_qposadr[joints]
    def point(offset,seed):
        probe=mj.MjData(m);probe.qpos[:]=seed;probe.qpos[pq]=px+offset
        ik=[solve_ik(m,probe,side,p+[offset,0,0],R) for side,(p,R) in targets.items()]
        pos=max(r['position_error_m'] for r in ik);angle=max(r['angle_error_rad'] for r in ik);pen,pairs=collisions(m,probe)
        margin=float(np.min(np.minimum(probe.qpos[qa]-m.jnt_range[joints,0],m.jnt_range[joints,1]-probe.qpos[qa])))
        return {'offset_m':float(offset),'position_error_m':pos,'angle_error_rad':angle,'nonpad_penetration_m':pen,'pairs':pairs,
                'joint_min_margin_rad':margin,'passed':bool(pos<1e-5 and angle<1e-4 and pen<1e-5)},probe.qpos.copy()
    scan=[];seeds=[(0.,q0.copy())];seed=q0.copy()
    for offset in np.arange(.00025,.02501,.00025):
        row,new=point(offset,seed);scan.append(row)
        if not row['passed']:break
        seed=new;seeds.append((float(offset),seed.copy()))
    last=seeds[-1][0];specs=[]
    for minimum in [.6,0.]:
        tr=make_otg(.2,.6,minimum_duration=minimum);ts=np.linspace(0,max(.6,tr.duration),10001)
        o=np.array([np.array(tr.at_time(min(t,tr.duration)))[:,0] for t in ts]);base=np.array([stop(t,.2,.6) for t in ts]);relative=o[:,0]-base[:,0]
        peak=float(max(relative));points=[]
        for value in [peak,peak+.001]:
            point_seed=max((s for s in seeds if s[0]<=value),key=lambda s:s[0])[1];check,_=point(value,point_seed);points.append(check)
        assert max(abs(o[0,1:]-[.2,.6]))<1e-12 and max(abs(o[-1,1:]))<1e-10
        assert min(o[:,1])>-1e-10 and max(abs(o[:,2]))<=.78+1e-12
        rate=float(max(abs(np.diff(o[:,2])/np.diff(ts))));assert rate<=5.+1e-8
        specs.append({'minimum_duration_s':minimum,'object_duration_s':tr.duration,'object_distance_m':float(o[-1,0]),
                      'relative_max_m':peak,'relative_min_m':float(min(relative)),'base_distance_m':float(base[-1,0]),
                      'max_reference_acceleration_mps2':float(max(abs(o[:,2]))),'max_reference_acceleration_difference_rate_mps3':rate,'required_point_checks':points})
    required=specs[1]['relative_max_m'];extent=math.ceil((required+.001)/.00025)*.00025
    path=None;attempts=[]
    if extent<=last and all(p['passed'] for p in specs[1]['required_point_checks']):
        for backward in [.01,.005,.002]:
            try:
                path=arm_path(m,q0,extent=extent,backward_extent=backward)
                attempts.append({'forward_m':extent,'backward_m':backward,'passed':True,'audit':path['audit']});break
            except AssertionError as e:attempts.append({'forward_m':extent,'backward_m':backward,'passed':False,'error':str(e)})
    admissions=[]
    if path is not None:
        for minimum in [None,0.]:
            ref=StateStop(ramp_start=3.,seed_policy='command',batched=True,object_minimum_duration_s=minimum)
            d.qpos[:]=q0;d.qvel[:]=0.;mj.mj_forward(m,d);ref.prepare(m,d,path);observe(ref,m,d,q0)
            admissions.append(ref.audit)
        assert not admissions[0]['preferred_plan_admitted']
    ready=bool(admissions and admissions[1]['preferred_plan_admitted'])
    result={'invalid_duration_inputs_rejected':len(invalid),'fine_branch_scan':{'step_m':.00025,'last_passing_offset_m':last,'first_failure':next((r for r in scan if not r['passed']),None),'samples':scan},
            'reference_specs':specs,'path_attempts':attempts,'admissions':admissions,'ready_for_physical_test':ready,
            'selected_path':None if path is None else {'forward_m':path['extent'],'backward_m':path['backward_extent']},
            'initialization':str(DEST/'initialization.json'),
            'qualification':'One fixed-torso/hand-orientation seeded IK branch, with .005rad solver joint margin; no global infeasibility proof. Removing a timing floor is a mature-baseline setting, not a new optimizer. No physical integration or disturbance injection in this check.'}
    dump(DEST/'independent_checks.json',result)
    print(json.dumps({'last_passing_m':last,'first_failure_m':None if result['fine_branch_scan']['first_failure'] is None else result['fine_branch_scan']['first_failure']['offset_m'],
                      'references':[{k:r[k] for k in ['minimum_duration_s','object_duration_s','relative_max_m','required_point_checks']} for r in specs],
                      'path_attempts':attempts,'ready_for_physical_test':ready}))


if __name__=='__main__':main()
