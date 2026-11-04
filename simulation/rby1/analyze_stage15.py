"""Verify physical regressions, reference release timing and finite task outcomes."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from build_model import OUT,HERE,dump
from analyze_buffer import trace
from state_stop_reference import make_otg


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def prelude(t,a):
    u=t-a['ramp_start_s'];v=a['cruise_speed_mps']
    if u<=0:return np.array([0.,0.,0.])
    if u>=1:return np.array([v*(u-.5),v,0.])
    return np.array([v*(u**3-.5*u**4),v*(3*u*u-2*u**3),6*v*u*(1-u)])


def main():
    dest=OUT.parent/'stage15';runs={}
    for p in sorted((OUT.parent/'buffer_runs').glob('*s15_*/result.json')):
        r=json.loads(p.read_text());assert r['label'] not in runs;runs[r['label']]=r
    rows=[]
    for label,r in runs.items():
        p=Path(r['folder'])
        for file,key in [('scene.xml','scene_sha256'),('initialization.json','initialization_sha256'),
                         ('run_snapshot.py','runner_sha256'),('reference_snapshot.py','reference_sha256'),
                         ('wrench_snapshot.py','wrench_script_sha256'),('load_sharing_snapshot.py','load_sharing_script_sha256'),
                         ('uncertainty_snapshot.py','uncertainty_script_sha256'),('stop_profile_snapshot.py','stop_profile_script_sha256'),('static_payload_snapshot.py','static_payload_script_sha256')]:
            assert sha(p/file)==r[key],(label,key)
        assert r['equality_constraints']==0
        a=r['reference_admission'];z=trace(r);time=z['time_s'];pending_count=0;max_release_reference_error=0.
        if a:
            assert sha(p/'state_stop_snapshot.py')==r['state_stop_script_sha256']
            assert json.loads((p/'reference_admission.json').read_text())==a
            samples=np.load(p/'admission_trace.npz')['samples']
            assert np.max(np.abs(samples[:,7]-(samples[:,4]-samples[:,1])))<1e-12
            assert abs(max(samples[:,8])-a['candidate_envelope_front_travel_m'])<1e-12
            assert a['preferred_plan_admitted']
            otg=make_otg(a['otg_initial_velocity_mps'],a['otg_initial_acceleration_mps2'],minimum_duration=a['minimum_remaining_duration_s'])
            for i,t in enumerate(time):
                if t<a['trigger_s']:continue
                if t<a['result_release_s']:
                    expected=prelude(t,a)[1:];pending_count+=1
                    assert z['base_ref_v_mps'][i]<prelude(t,a)[1]
                else:
                    pva=np.array(otg.at_time(min(t-a['result_release_s'],otg.duration)))[:,0];expected=pva[1:]
                error=float(np.max(np.abs(expected-np.array([z['object_ref_v_mps'][i],z['object_ref_a_mps2'][i]]))))
                max_release_reference_error=max(max_release_reference_error,error)
            assert max_release_reference_error<1e-11,(label,max_release_reference_error)
            if a['seed_policy']=='command':assert max(abs(np.array(a['planned_handover_reference_jump'])))<1e-12
        envelope=r['max_upper_envelope_front_travel_m'];jump=None if not a else a['planned_handover_reference_jump']
        physical=all(r[k] for k in ['quality_passed','original_grasp_criterion_passed','braking_passed'])
        rows.append({'label':label,'seed_policy':None if not a else a['seed_policy'],'delay_ms':None if not a else a['planning_delay_s']*1000,
                     'dt_ms':r['config']['dt_s']*1000,'quality':r['quality_passed'],'grasp':r['original_grasp_criterion_passed'],'braking':r['braking_passed'],
                     'acceleration_mps2':r['max_brake_payload_acceleration_mps2'],'angle_deg':r['max_brake_payload_target_angle_error_deg'],
                     'base_cm':100*r['brake_distance_to_final_m'],'arm_forward_mm':1000*r['max_extra_ee_forward_m'],
                     'envelope_cm':None if envelope is None else envelope*100,'grasp_mm':r['max_relative_change_after_1s_m']*1000,
                     'reference_acceleration_difference_rate_mps3':r['peak_object_reference_jerk_mps3'],
                     'handover_position_jump_m':None if jump is None else jump[0],'handover_velocity_jump_mps':None if jump is None else jump[1],
                     'handover_acceleration_jump_mps2':None if jump is None else jump[2],
                     'sampled_pending_trace_points':pending_count,'release_reference_error':max_release_reference_error,
                     'example_motion_task_passed':bool(physical and r['max_brake_payload_acceleration_mps2']<=.8 and r['max_brake_payload_target_angle_error_deg']<=.5 and (envelope is None or envelope<=r['config']['front_clearance_budget_m'])),
                     'max_brake_sampled_velocity_tracking_error_mps':float(max(abs(z['object_com_vx_mps'][time>=3.5]-z['object_ref_v_mps'][time>=3.5]))),
                     'folder':str(p)})
    old=json.loads((OUT.parent/'stage14/summary.json').read_text());regressions=[]
    names=[('s15_regression_fixed','s14_final_regression_fixed'),('s15_regression_buffer','s14_final_regression_buffer'),('s15_fast_measured_accelerating','s14_accelerating_ruckig')]
    for new,previous in names:
        if new not in runs:continue
        a=np.load(Path(runs[new]['folder'])/'joint_trace.npz');b=np.load(Path(old['runs'][previous]['folder'])/'joint_trace.npz')
        exact=all(np.array_equal(a[k],b[k]) for k in ['time','qpos','qvel','ctrl']);assert exact
        regressions.append({'new':new,'old':previous,'time_state_control_exact':exact})
    if 's15_final_command_accelerating_45ms' in runs:
        new=runs['s15_final_command_accelerating_45ms'];previous=runs['s15_command_accelerating_45ms']
        a=np.load(Path(new['folder'])/'joint_trace.npz');b=np.load(Path(previous['folder'])/'joint_trace.npz')
        exact=all(np.array_equal(a[k],b[k]) for k in ['time','qpos','qvel','ctrl']);assert exact
        regressions.append({'new':new['label'],'old':previous['label'],'time_state_control_exact':exact})
    pairs=[]
    for case in ['nominal','accelerating']:
        base='s15_command_'+case+'_0ms'
        for label in runs:
            if label==base or '_halfdt' in label or not label.startswith('s15_command_'+case+'_'):continue
            if base not in runs:continue
            r,t=runs[base],runs[label]
            assert all(r[k]==t[k] for k in ['scene_sha256','initialization_sha256','runner_sha256','reference_sha256','state_stop_script_sha256'])
            a=trace(r);b=trace(t);pre=a['time_s']<3.5;post=~pre
            ar=np.load(Path(r['folder'])/'joint_trace.npz');br=np.load(Path(t['folder'])/'joint_trace.npz')
            exact=all(np.array_equal(ar[k][pre],br[k][pre]) for k in ['qpos','qvel','ctrl']);assert exact
            assert np.array_equal(a['base_ref_v_mps'],b['base_ref_v_mps'])
            dx=float(max(abs(a['base_x_m'][post]-b['base_x_m'][post])));dv=float(max(abs(a['base_v_mps'][post]-b['base_v_mps'][post])))
            assert dx<.002 and dv<.01
            pairs.append({'case':case,'other':label,'pretrigger_state_control_exact':exact,'base_reference_exact':True,'base_max_position_difference_m':dx,'base_max_speed_difference_mps':dv})
    half=None
    if 's15_command_accelerating_45ms_halfdt' in runs:
        a=runs['s15_command_accelerating_45ms'];b=runs['s15_command_accelerating_45ms_halfdt']
        half={'main_acceleration':a['max_brake_payload_acceleration_mps2'],'half_acceleration':b['max_brake_payload_acceleration_mps2'],
              'main_envelope_m':a['max_upper_envelope_front_travel_m'],'half_envelope_m':b['max_upper_envelope_front_travel_m'],
              'same_quality_grasp_brake':all(a[k]==b[k] for k in ['quality_passed','original_grasp_criterion_passed','braking_passed'])}
    result={'run_count':len(runs),'runs':runs,'metrics':rows,'regressions':regressions,'delay_pairs':pairs,'half_timestep':half,
            'all_snapshot_hashes_verified':True,'independent_checks':json.loads((dest/'independent_checks.json').read_text()),
            'qualification':'Command-reference P/V/A continuity plus existing measured-state feedback. Deterministic result-availability delays are explicitly simulated, not measured communication latency or asynchronous real-time implementation. Only sampled task feasibility and finite physical cases; no MPC, active torso, physical jerk guarantee or novelty claim.'}
    dump(dest/'summary.json',result)
    if rows:
        with (dest/'runs.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps([{k:r[k] for k in ['label','acceleration_mps2','arm_forward_mm','envelope_cm','example_motion_task_passed']} for r in rows]))


if __name__=='__main__':main()
