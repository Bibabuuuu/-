"""Verify cleanup and report continuous metrics without invented payload pass limits."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from build_model import OUT,HERE,dump
from analyze_buffer import trace


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    dest=OUT.parent/'stage19';runs={}
    for p in sorted((OUT.parent/'buffer_runs').glob('*s19_*/result.json')):
        r=json.loads(p.read_text());assert r['label'] not in runs;runs[r['label']]=r
    assert set(runs)=={'s19_legacy','s19_nominal','s19_combined','s19_fast'}
    rows=[]
    for name,r in runs.items():
        p=Path(r['folder']);z=trace(r);a=r['reference_admission']
        for filename,key in [('scene.xml','scene_sha256'),('initialization.json','initialization_sha256'),
                             ('run_snapshot.py','runner_sha256'),('reference_snapshot.py','reference_sha256'),
                             ('state_stop_snapshot.py','state_stop_script_sha256'),('wrench_snapshot.py','wrench_script_sha256'),
                             ('load_sharing_snapshot.py','load_sharing_script_sha256'),('uncertainty_snapshot.py','uncertainty_script_sha256'),
                             ('stop_profile_snapshot.py','stop_profile_script_sha256'),('static_payload_snapshot.py','static_payload_script_sha256')]:
            assert sha(p/filename)==r[key],(name,key)
        # Historical evidence is verified against its own snapshots, not later source revisions.
        assert r['equality_constraints']==0 and r['config']['world_x_feedback']
        if name!='s19_legacy':
            assert r['config']['admission']['requested_s']==r['config']['base_T_s']==.6
            assert r['config']['admission']['explicit_reference_acceleration_cap_mps2'] is None
            assert not r['config']['admission']['legacy_time_stretch_enabled']
            assert a['clearance_m'] is None and a['space_reserve_m']==a['stroke_reserve_m']==a['minimum_remaining_duration_s']==0.
            assert not a['legacy_base_duration_floor'] and r['front_clearance_passed'] is None
        post=z['time_s']>=3.5;times=z['time_s'][post];speeds=abs(z['base_v_mps'][post])
        settled=np.maximum.accumulate(speeds[::-1])[::-1]<.01
        approximate_stop=float(times[np.flatnonzero(settled)[0]]-3.5) if any(settled) else None
        caps={j['name']:j['force_cap'] for j in r['per_joint']};torque=r['phase_joint_peak_torques_Nm']['braking']
        wheel=max(abs(v)/caps[k] for k,v in torque.items() if 'wheel' in k)
        rows.append({'label':name,'completed':r['status']=='completed','model_checks':r['quality_passed'],
                     'requested_base_time_s':r['config']['admission']['requested_s'],'base_reference_time_s':r['config']['base_T_s'],
                     'base_tracking_check':r['braking_passed'],'sampled_settling_time_below_0_01mps_s':approximate_stop,
                     'base_distance_cm':100*r['brake_distance_to_final_m'],'reference_admitted':a['preferred_plan_admitted'],
                     'object_reference_duration_s':a['candidate_duration_s'],'peak_object_acceleration_mps2':r['max_brake_payload_acceleration_mps2'],
                     'object_angle_error_deg':r['max_brake_payload_target_angle_error_deg'],
                     'grasp_relative_change_mm':1000*r['max_relative_change_after_1s_m'],
                     'legacy_grasp_diagnostic':r['original_grasp_criterion_passed'],
                     'arm_forward_mm':1000*r['max_extra_ee_forward_m'],'upper_envelope_cm':100*r['max_upper_envelope_front_travel_m'],
                     'braking_wheel_torque_cap_fraction':wheel,'reference_guard_triggered':r['reference_corridor_guard'] is not None})
    prior=json.loads((OUT.parent/'stage18/summary.json').read_text())['runs']['s18_default_posture_regression']
    old=np.load(Path(prior['folder'])/'joint_trace.npz');new=np.load(Path(runs['s19_legacy']['folder'])/'joint_trace.npz')
    exact=all(np.array_equal(old[k],new[k]) for k in ['time','qpos','qvel','ctrl']);assert exact
    frozen={str(OUT/'research_model.xml'):'a66346c0839622d469dd4739d5f2a172ab8a1424d86537b43a571925c3c0195e',
            str(OUT.parent/'payload_model/payload_scene.xml'):'d6701100d6fd49deb2541093ca6fca63acf4ffa676d8cbc8e3a3fe29bb824154',
            str(HERE/'run_payload.py'):'0d426bd07b83fa7d9e9aa55be086826dd5cbfef845aee8a89e706febeeb15906'}
    for filename,expected in frozen.items():assert sha(Path(filename))==expected,filename
    result={'run_count':len(runs),'runs':runs,'metrics':rows,'legacy_exact_state_control_regression':exact,
            'all_snapshots_verified':True,'frozen_models_unchanged':True,
            'no_universal_payload_pass_limit':True,'task_payload_limits':None,
            'qualification':'Nominal uncalibrated model, straight motion, fixed torso and hand orientation, .5kg rigid payload and 6s observation. A/J are shaping settings. No general object safety, hardware or novelty claim.'}
    dump(dest/'summary.json',result)
    with (dest/'runs.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump(dest/'delivery_check.json',{'complete_runs':len(runs),'legacy_exact_regression':exact,'all_run_snapshots_verified':True,
                                   'frozen_models_unchanged':True,'future_default_constraints_removed':True})
    print(json.dumps(rows))


if __name__=='__main__':main()
