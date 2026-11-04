"""Separate execution tracking, original stop requirements and object criteria."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from build_model import OUT,HERE,dump
from analyze_buffer import trace
from buffer_reference import stop
from analyze_stage15 import prelude


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    dest=OUT.parent/'stage16';runs={}
    for p in sorted((OUT.parent/'buffer_runs').glob('*s16_*/result.json')):
        r=json.loads(p.read_text());assert r['label'] not in runs;runs[r['label']]=r
    rows=[]
    for name,r in runs.items():
        folder=Path(r['folder']);z=trace(r);post=z['time_s']>=3.5
        for file,key in [('scene.xml','scene_sha256'),('initialization.json','initialization_sha256'),('run_snapshot.py','runner_sha256'),
                         ('reference_snapshot.py','reference_sha256'),('wrench_snapshot.py','wrench_script_sha256'),
                         ('load_sharing_snapshot.py','load_sharing_script_sha256'),('uncertainty_snapshot.py','uncertainty_script_sha256'),
                         ('stop_profile_snapshot.py','stop_profile_script_sha256'),('static_payload_snapshot.py','static_payload_script_sha256')]:
            assert sha(folder/file)==r[key],(name,key)
        assert r['equality_constraints']==0
        a=r['reference_admission'];execution=None if a is None else a['base_execution_emulation']
        if a:
            assert sha(folder/'state_stop_snapshot.py')==r['state_stop_script_sha256']
            assert sha(folder/'base_execution_snapshot.py')==r['base_execution_script_sha256']==sha(HERE/'brake_prediction.py')
            assert json.loads((folder/'reference_admission.json').read_text())==a and execution==r['config']['base_execution_emulation']
            assert a['preferred_plan_admitted'] and max(abs(np.array(a['planned_handover_reference_jump'])))<1e-12
            x,v,_=prelude(a['trigger_s'],a)
            for i in np.where(post)[0]:
                t=z['time_s'][i]-a['trigger_s'];lag=execution['execution_start_delay_s']
                p,sv,acc=(v*t,v,0.) if t<=lag else stop(t-lag,v,execution['execution_T_s'])
                if t>lag:p+=v*lag
                assert max(abs(np.array([z['base_ref_x_m'][i]-z['base_ref_x_m'][0]-x,z['base_ref_v_mps'][i],z['base_ref_a_mps2'][i]])-np.array([p,sv,acc])))<1e-10
        obj_error=z['object_com_x_m']-z['object_ref_com_x_m'];base_error=z['base_x_m']-z['base_forecast_x_m']
        joint_peak=float(max(z['max_arm_joint_reference_error_rad'][post]))
        envelope=r['max_upper_envelope_front_travel_m'];space_ok=None if envelope is None else envelope<=r['config']['front_clearance_budget_m']
        acc_ok=r['max_brake_payload_acceleration_mps2']<=.8;angle_ok=r['max_brake_payload_target_angle_error_deg']<=.5
        physical=bool(r['quality_passed'] and r['original_grasp_criterion_passed'])
        caps={j['name']:j['force_cap'] for j in r['per_joint']};torques=r['phase_joint_peak_torques_Nm']['braking']
        upper=max(abs(t)/caps[n] for n,t in torques.items() if 'wheel' not in n and 'finger' not in n)
        wheel=max(abs(t)/caps[n] for n,t in torques.items() if 'wheel' in n)
        rows.append({'label':name,'execution_T_s':None if not execution else execution['execution_T_s'],'execution_start_delay_ms':None if not execution else execution['execution_start_delay_s']*1000,
                     'dt_ms':r['config']['dt_s']*1000,'quality':r['quality_passed'],'grasp':r['original_grasp_criterion_passed'],
                     'original_braking_metric_passed':r['braking_passed'],'execution_tracking_passed':r.get('execution_brake_tracking_passed'),
                     'acceleration_mps2':r['max_brake_payload_acceleration_mps2'],'acceleration_within_example_limit':acc_ok,
                     'angle_deg':r['max_brake_payload_target_angle_error_deg'],'envelope_cm':None if envelope is None else envelope*100,
                     'envelope_within_example_budget':space_ok,'base_distance_cm':100*r['brake_distance_to_final_m'],
                     'actual_arm_forward_mm':1000*r['max_extra_ee_forward_m'],'grasp_mm':1000*r['max_relative_change_after_1s_m'],
                     'object_world_reference_error_mm':1000*r['max_brake_payload_world_reference_x_error_m'],
                     'base_forecast_position_error_mm':1000*r['max_brake_base_forecast_position_error_m'],
                     'sampled_world_error_minus_base_error_max_mm':1000*float(max(abs(obj_error[post]-base_error[post]))),
                     'sampled_arm_joint_reference_error_deg':float(np.rad2deg(joint_peak)),
                     'braking_upper_torque_cap_fraction':upper,'braking_wheel_torque_cap_fraction':wheel,
                     'object_example_criteria_passed':None if a is None else bool(physical and acc_ok and angle_ok and space_ok),
                     'combined_original_task_passed':None if a is None else bool(physical and acc_ok and angle_ok and space_ok and r['braking_passed']),
                     'folder':str(folder)})
    previous=json.loads((OUT.parent/'stage15/summary.json').read_text());regressions=[]
    for new,old in [('s16_regression_fixed','s15_regression_fixed'),('s16_regression_buffer','s15_regression_buffer'),
                    ('s16_nominal_T600_lag0','s15_command_nominal_0ms'),('s16_accelerating_T600_lag0','s15_command_accelerating_0ms')]:
        if new not in runs:continue
        a=np.load(Path(runs[new]['folder'])/'joint_trace.npz');b=np.load(Path(previous['runs'][old]['folder'])/'joint_trace.npz')
        errors={k:float(max(abs(a[k]-b[k]).ravel())) for k in ['time','qpos','qvel','ctrl']}
        exact=all(np.array_equal(a[k],b[k]) for k in errors)
        if 'regression' in new:assert exact
        else:assert max(errors.values())<1e-9,(new,errors)
        regressions.append({'new':new,'old':old,'exact':exact,'max_errors':errors,'within_1e_9':True})
    matched=[]
    for case in ['nominal','accelerating']:
        baseline='s16_'+case+'_T600_lag0'
        if baseline not in runs:continue
        br=runs[baseline];bz=trace(br);bstates=np.load(Path(br['folder'])/'joint_trace.npz');samples=np.load(Path(br['folder'])/'admission_trace.npz')['samples'];pre=bz['time_s']<3.5
        for name,r in runs.items():
            if name==baseline or not name.startswith('s16_'+case+'_') or 'halfdt' in name:continue
            z=trace(r);states=np.load(Path(r['folder'])/'joint_trace.npz')
            assert all(br[k]==r[k] for k in ['scene_sha256','initialization_sha256','runner_sha256','reference_sha256','state_stop_script_sha256','base_execution_script_sha256'])
            assert all(np.array_equal(bstates[k][pre],states[k][pre]) for k in ['qpos','qvel','ctrl'])
            assert np.array_equal(samples,np.load(Path(r['folder'])/'admission_trace.npz')['samples'])
            assert all(np.array_equal(bz[k],z[k]) for k in ['object_ref_com_x_m','object_ref_v_mps','object_ref_a_mps2','base_forecast_x_m','base_forecast_v_mps','base_forecast_a_mps2','arm_relative_reference_x_m'])
            matched.append({'case':case,'other':name,'pretrigger_state_control_exact':True,'object_plan_forecast_and_arm_relative_reference_exact':True})
    half=None
    if 's16_nominal_T720_lag0_halfdt' in runs:
        a=runs['s16_nominal_T720_lag0'];b=runs['s16_nominal_T720_lag0_halfdt']
        half={'main_acceleration':a['max_brake_payload_acceleration_mps2'],'half_acceleration':b['max_brake_payload_acceleration_mps2'],
              'main_envelope_cm':100*a['max_upper_envelope_front_travel_m'],'half_envelope_cm':100*b['max_upper_envelope_front_travel_m'],
              'same_quality_grasp_original_brake_execution_tracking':all(a[k]==b[k] for k in ['quality_passed','original_grasp_criterion_passed','braking_passed','execution_brake_tracking_passed'])}
    result={'run_count':len(runs),'runs':runs,'metrics':rows,'regressions':regressions,'matched_mismatch_cases':matched,'half_timestep':half,
            'all_snapshot_hashes_verified':True,'independent_checks':json.loads((dest/'independent_checks.json').read_text()),
            'qualification':'One-shot object plan and forecast-based arm-relative tracking, with a separately prescribed base execution curve. No actual brake-fault calibration, active torso, saturation search or replanning. Original brake matching, execution tracking and object criteria are separately reported.'}
    dump(dest/'summary.json',result)
    if rows:
        with (dest/'runs.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps([{k:r[k] for k in ['label','acceleration_mps2','envelope_cm','object_world_reference_error_mm','sampled_arm_joint_reference_error_deg','execution_tracking_passed','original_braking_metric_passed']} for r in rows]))


if __name__=='__main__':main()
