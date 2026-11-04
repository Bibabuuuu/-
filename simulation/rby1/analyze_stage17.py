"""Nominal paired outcomes; compensation is basic execution, not new planning."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from build_model import OUT,HERE,dump
from analyze_buffer import trace


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    dest=OUT.parent/'stage17';runs={}
    for p in sorted((OUT.parent/'buffer_runs').glob('*s17_*/result.json')):
        r=json.loads(p.read_text());assert r['label'] not in runs;runs[r['label']]=r
    rows=[]
    for name,r in runs.items():
        p=Path(r['folder']);z=trace(r);post=z['time_s']>=3.5
        for file,key in [('scene.xml','scene_sha256'),('initialization.json','initialization_sha256'),('run_snapshot.py','runner_sha256'),
                         ('reference_snapshot.py','reference_sha256'),('wrench_snapshot.py','wrench_script_sha256'),
                         ('load_sharing_snapshot.py','load_sharing_script_sha256'),('uncertainty_snapshot.py','uncertainty_script_sha256'),
                         ('stop_profile_snapshot.py','stop_profile_script_sha256'),('static_payload_snapshot.py','static_payload_script_sha256')]:
            assert sha(p/file)==r[key],(name,key)
        assert r['equality_constraints']==0
        a=r['reference_admission'];feedback=r['config']['world_x_feedback']
        if a:
            assert sha(p/'state_stop_snapshot.py')==r['state_stop_script_sha256']
            assert json.loads((p/'reference_admission.json').read_text())==a
            assert a['preferred_plan_admitted']
        object_displacement=z['object_ref_com_x_m']-z['object_ref_com_x_m'][0]
        base_position=z['base_x_m']-z['base_forecast_x_m'][0] if feedback else z['base_forecast_x_m']-z['base_forecast_x_m'][0]
        mapping_error=float(max(abs(z['arm_relative_reference_x_m']-(object_displacement-base_position))))
        assert mapping_error<1e-12
        if feedback:
            assert r['reference_corridor_guard'] is None
            assert min(z['arm_relative_reference_x_m'])>=-r['config']['path_backward_extent_m']-1e-9
            assert max(z['arm_relative_reference_x_m'])<=r['config']['path_extent_m']+1e-9
        envelope=r['max_upper_envelope_front_travel_m'];space=None if envelope is None else envelope<=r['config']['front_clearance_budget_m']
        object_ok=None if a is None else bool(r['quality_passed'] and r['original_grasp_criterion_passed'] and r['max_brake_payload_acceleration_mps2']<=.8 and r['max_brake_payload_target_angle_error_deg']<=.5 and space)
        rows.append({'label':name,'feedback':feedback,'strategy':a['strategy'] if a else r['mode'],
                     'quality':r['quality_passed'],'grasp':r['original_grasp_criterion_passed'],'braking':r['braking_passed'],
                     'acceleration_mps2':r['max_brake_payload_acceleration_mps2'],'target_angle_deg':r['max_brake_payload_target_angle_error_deg'],
                     'base_distance_cm':100*r['brake_distance_to_final_m'],'arm_forward_mm':1000*r['max_extra_ee_forward_m'],
                     'envelope_cm':None if envelope is None else 100*envelope,'clearance_cm':None if r['config']['front_clearance_budget_m'] is None else 100*r['config']['front_clearance_budget_m'],
                     'world_reference_error_mm':1000*r['max_brake_payload_world_reference_x_error_m'],
                     'grasp_mm':1000*r['max_relative_change_after_1s_m'],'mapping_check_error_m':mapping_error,
                     'max_mapping_correction_mm':1000*r['max_brake_mapping_correction_m'],
                     'relative_reference_min_mm':1000*r['braking_relative_reference_min_m'],'relative_reference_max_mm':1000*r['braking_relative_reference_max_m'],
                     'checked_path_forward_mm':1000*r['config']['path_extent_m'],'checked_path_backward_mm':1000*r['config']['path_backward_extent_m'],
                     'reference_acceleration_difference_rate_mps3':r['peak_object_reference_jerk_mps3'],
                     'sampled_arm_joint_error_deg':float(np.rad2deg(max(z['max_arm_joint_reference_error_rad'][post]))),
                     'object_example_criteria_passed':object_ok,'folder':str(p)})
    previous=json.loads((OUT.parent/'stage15/summary.json').read_text());regressions=[]
    for new,old in [('s17_regression_fixed','s15_regression_fixed'),('s17_regression_buffer','s15_regression_buffer'),('s17_legacy_nominal_ruckig','s15_command_nominal_0ms')]:
        if new not in runs:continue
        a=np.load(Path(runs[new]['folder'])/'joint_trace.npz');b=np.load(Path(previous['runs'][old]['folder'])/'joint_trace.npz')
        exact=all(np.array_equal(a[k],b[k]) for k in ['time','qpos','qvel','ctrl']);assert exact
        regressions.append({'new':new,'old':old,'exact_time_state_control':exact})
    pairs=[]
    for case in ['nominal','accelerating','faster','posture']:
        fn='s17_'+case+'_fixed';bn='s17_'+case+'_ruckig'
        if fn not in runs or bn not in runs:continue
        f,b=runs[fn],runs[bn];ft,bt=trace(f),trace(b);pre=ft['time_s']<3.5;post=~pre
        assert all(f[k]==b[k] for k in ['scene_sha256','initialization_sha256','runner_sha256','reference_sha256','state_stop_script_sha256'])
        fs=np.load(Path(f['folder'])/'joint_trace.npz');bs=np.load(Path(b['folder'])/'joint_trace.npz')
        assert all(np.array_equal(fs[k][pre],bs[k][pre]) for k in ['qpos','qvel','ctrl'])
        assert np.array_equal(ft['base_ref_v_mps'],bt['base_ref_v_mps'])
        dx=float(max(abs(ft['base_x_m'][post]-bt['base_x_m'][post])));dv=float(max(abs(ft['base_v_mps'][post]-bt['base_v_mps'][post])))
        assert dx<.002 and dv<.01
        pairs.append({'case':case,'pretrigger_state_control_exact':True,'base_reference_exact':True,'base_motion_comparable':True,
                      'base_position_difference_m':dx,'base_speed_difference_mps':dv,
                      'synchronous_acceleration':f['max_brake_payload_acceleration_mps2'],'buffer_acceleration':b['max_brake_payload_acceleration_mps2'],
                      'peak_acceleration_reduction_percent':100*(1-b['max_brake_payload_acceleration_mps2']/f['max_brake_payload_acceleration_mps2']),
                      'extra_arm_forward_mm':1000*(b['max_extra_ee_forward_m']-f['max_extra_ee_forward_m']),
                      'extra_envelope_mm':1000*(b['max_upper_envelope_front_travel_m']-f['max_upper_envelope_front_travel_m'])})
    feedback_comparison=None
    if 's17_nominal_ruckig' in runs and 's17_legacy_nominal_ruckig' in runs:
        a,b=runs['s17_legacy_nominal_ruckig'],runs['s17_nominal_ruckig']
        same=np.array_equal(np.load(Path(a['folder'])/'admission_trace.npz')['samples'],np.load(Path(b['folder'])/'admission_trace.npz')['samples']);assert same
        feedback_comparison={'object_stop_plan_exactly_unchanged':same,'legacy_world_reference_error_mm':1000*a['max_brake_payload_world_reference_x_error_m'],
                             'feedback_world_reference_error_mm':1000*b['max_brake_payload_world_reference_x_error_m'],
                             'legacy_acceleration':a['max_brake_payload_acceleration_mps2'],'feedback_acceleration':b['max_brake_payload_acceleration_mps2'],
                             'note':'Feedback active from initialization; physical pretrigger states need not be identical between the old/new mappings. Paired synchronous/buffer tests use the same new mapping.'}
    half=None
    if 's17_posture_ruckig_halfdt' in runs:
        a,b=runs['s17_posture_ruckig'],runs['s17_posture_ruckig_halfdt']
        half={'main_acceleration':a['max_brake_payload_acceleration_mps2'],'half_acceleration':b['max_brake_payload_acceleration_mps2'],
              'main_world_error_mm':1000*a['max_brake_payload_world_reference_x_error_m'],'half_world_error_mm':1000*b['max_brake_payload_world_reference_x_error_m'],
              'same_quality_grasp_braking':all(a[k]==b[k] for k in ['quality_passed','original_grasp_criterion_passed','braking_passed'])}
    result={'run_count':len(runs),'runs':runs,'metrics':rows,'regressions':regressions,'pairs':pairs,'feedback_comparison':feedback_comparison,
            'half_timestep':half,'all_snapshot_hashes_verified':True,'independent_checks':json.loads((dest/'independent_checks.json').read_text()),
            'offline_reference_admissions':json.loads((dest/'reference_admission_checks.json').read_text()),
            'qualification':'Nominal straight-line cases with basic measured-base forward compensation and one-shot mature stop references. No injected brake mismatch, wait, noise, new optimizer, active torso or novelty evidence. Seeded workspace scans are branch-local, not global reachability or hardware limits.'}
    dump(dest/'summary.json',result)
    if rows:
        with (dest/'runs.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps([{k:r[k] for k in ['label','acceleration_mps2','arm_forward_mm','world_reference_error_mm','relative_reference_max_mm']} for r in rows]))


if __name__=='__main__':main()
