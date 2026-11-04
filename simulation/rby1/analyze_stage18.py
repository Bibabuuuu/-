"""Separate reference feasibility, physical execution and the illustrative limit."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from build_model import OUT,HERE,dump
from analyze_buffer import trace


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    dest=OUT.parent/'stage18';runs={}
    for p in sorted((OUT.parent/'buffer_runs').glob('*s18_*/result.json')):
        r=json.loads(p.read_text());assert r['label'] not in runs;runs[r['label']]=r
    rows=[]
    for name,r in runs.items():
        p=Path(r['folder']);z=trace(r);post=z['time_s']>=3.5
        for file,key in [('scene.xml','scene_sha256'),('initialization.json','initialization_sha256'),('run_snapshot.py','runner_sha256'),
                         ('reference_snapshot.py','reference_sha256'),('state_stop_snapshot.py','state_stop_script_sha256'),
                         ('wrench_snapshot.py','wrench_script_sha256'),('load_sharing_snapshot.py','load_sharing_script_sha256'),
                         ('uncertainty_snapshot.py','uncertainty_script_sha256'),('stop_profile_snapshot.py','stop_profile_script_sha256'),('static_payload_snapshot.py','static_payload_script_sha256')]:
            assert sha(p/file)==r[key],(name,key)
        assert r['equality_constraints']==0 and r['reference_corridor_guard'] is None
        a=r['reference_admission'];assert a['preferred_plan_admitted']
        assert json.loads((p/'reference_admission.json').read_text())==a
        assert r['config']['world_x_feedback'] and r['config']['base_execution_emulation'] is None and a['planning_delay_s']==0
        base=z['base_x_m']-z['base_forecast_x_m'][0];obj=z['object_ref_com_x_m']-z['object_ref_com_x_m'][0]
        assert max(abs(z['arm_relative_reference_x_m']-(obj-base)))<1e-12
        caps={j['name']:j['force_cap'] for j in r['per_joint']};torque=r['phase_joint_peak_torques_Nm']['braking']
        upper=max(abs(v)/caps[k] for k,v in torque.items() if 'wheel' not in k and 'finger' not in k)
        wheel=max(abs(v)/caps[k] for k,v in torque.items() if 'wheel' in k)
        rows.append({'label':name,'dt_ms':r['config']['dt_s']*1000,'quality':r['quality_passed'],'grasp':r['original_grasp_criterion_passed'],'braking':r['braking_passed'],
                     'actual_peak_acceleration_mps2':r['max_brake_payload_acceleration_mps2'],
                     'object_reference_duration_s':a['candidate_duration_s'],'base_reference_duration_s':r['config']['base_T_s'],
                     'actual_arm_forward_mm':1000*r['max_extra_ee_forward_m'],
                     'actual_relative_reference_max_mm':1000*r['braking_relative_reference_max_m'],
                     'checked_forward_path_mm':1000*r['config']['path_extent_m'],
                     'world_reference_error_mm':1000*r['max_brake_payload_world_reference_x_error_m'],
                     'envelope_cm':100*r['max_upper_envelope_front_travel_m'],'target_angle_deg':r['max_brake_payload_target_angle_error_deg'],
                     'grasp_mm':1000*r['max_relative_change_after_1s_m'],'base_distance_cm':100*r['brake_distance_to_final_m'],
                     'sampled_arm_joint_error_deg':float(np.rad2deg(max(z['max_arm_joint_reference_error_rad'][post]))),
                     'braking_upper_torque_cap_fraction':upper,'braking_wheel_torque_cap_fraction':wheel,
                     'example_acceleration_limit_passed':r['max_brake_payload_acceleration_mps2']<=.8,
                     'example_pose_and_space_passed':bool(r['max_brake_payload_target_angle_error_deg']<=.5 and r['max_upper_envelope_front_travel_m']<=.15),
                     'folder':str(p)})
    old=json.loads((OUT.parent/'stage17/summary.json').read_text())['runs']['s17_posture_ruckig'];new=runs['s18_default_posture_regression']
    a=np.load(Path(old['folder'])/'joint_trace.npz');b=np.load(Path(new['folder'])/'joint_trace.npz')
    exact=all(np.array_equal(a[k],b[k]) for k in ['time','qpos','qvel','ctrl']);assert exact
    f=runs['s18_combined_free_duration'];h=runs['s18_combined_free_duration_halfdt']
    same_plan=np.array_equal(np.load(Path(f['folder'])/'admission_trace.npz')['samples'],np.load(Path(h['folder'])/'admission_trace.npz')['samples']);assert same_plan
    same=all(f[k]==h[k] for k in ['quality_passed','original_grasp_criterion_passed','braking_passed']);assert same
    independent=json.loads((dest/'independent_checks.json').read_text())
    assert independent['ready_for_physical_test'] and not independent['admissions'][0]['preferred_plan_admitted']
    assert f['reference_admission']['object_minimum_duration_s']==h['reference_admission']['object_minimum_duration_s']==0.
    result={'run_count':len(runs),'runs':runs,'metrics':rows,'default_exact_regression':exact,'all_snapshot_hashes_verified':True,
            'half_timestep':{'same_reference_plan':same_plan,'same_quality_grasp_braking':same,
                             'main_acceleration':f['max_brake_payload_acceleration_mps2'],'half_acceleration':h['max_brake_payload_acceleration_mps2'],
                             'both_example_acceleration_limit_failed':f['max_brake_payload_acceleration_mps2']>.8 and h['max_brake_payload_acceleration_mps2']>.8},
            'independent_checks':independent,
            'qualification':'The old candidate is rejected on the checked fixed-torso IK branch; removing an added object-duration floor gives a feasible mature reference. Physical execution/grasp/base stop pass, while actual peak acceleration exceeds the illustrative .8 limit. No new optimizer, global feasibility proof or physical damage evidence.'}
    dump(dest/'summary.json',result)
    with (dest/'runs.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(rows))


if __name__=='__main__':main()
