"""Physical outcomes, sampled admission and transition constraints are separate."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from build_model import OUT,HERE,dump
from analyze_buffer import trace


def main():
    dest=OUT.parent/'stage14';runs={}
    for p in sorted((OUT.parent/'buffer_runs').glob('*s14_*/result.json')):
        r=json.loads(p.read_text());runs[r['label']]=r
    rows=[]
    for name,r in runs.items():
        p=Path(r['folder']);z=trace(r)
        for file,key in [('scene.xml','scene_sha256'),('initialization.json','initialization_sha256'),
                         ('run_snapshot.py','runner_sha256'),('reference_snapshot.py','reference_sha256'),
                         ('wrench_snapshot.py','wrench_script_sha256'),('load_sharing_snapshot.py','load_sharing_script_sha256'),
                         ('uncertainty_snapshot.py','uncertainty_script_sha256'),('stop_profile_snapshot.py','stop_profile_script_sha256'),
                         ('static_payload_snapshot.py','static_payload_script_sha256')]:
            assert hashlib.sha256((p/file).read_bytes()).hexdigest()==r[key],(name,key)
        assert r['equality_constraints']==0
        a=r['reference_admission'];envelope=r['max_upper_envelope_front_travel_m']
        envelope_ok=None if a is None else envelope<=r['config']['front_clearance_budget_m']
        if a is not None:
            assert hashlib.sha256((p/'state_stop_snapshot.py').read_bytes()).hexdigest()==r['state_stop_script_sha256']
            assert json.loads((p/'reference_admission.json').read_text())==a
            if (p/'admission_trace.npz').is_file():
                saved=np.load(p/'admission_trace.npz')['samples'];assert np.max(np.abs(saved[:,7]-(saved[:,4]-saved[:,1])))<1e-12
                assert abs(np.max(saved[:,8])-a['candidate_envelope_front_travel_m'])<1e-12
            assert a['preferred_plan_admitted']==(len(a['reasons'])==0)
        # The saved initial acceleration mismatch is against the observed state,
        # not the left limit of the old commanded reference. Audit both separately.
        handover_v=handover_a=None
        if a is not None:
            u=a['trigger_s']-a['ramp_start_s'];v=a['cruise_speed_mps']
            old_v=v if u>=1 else v*(3*u*u-2*u**3) if u>0 else 0.
            old_a=0. if u<=0 or u>=1 else 6*v*u*(1-u)
            if a['applied_reference'] in ['fixed','fixed_upper_fallback']:
                new_v=old_v;new_a=0.
            else:
                new_v=a['measured_object_velocity_mps']
                new_a=a['measured_object_acceleration_mps2'] if a['strategy']=='ruckig' else 0.
            handover_v=new_v-old_v;handover_a=new_a-old_a
        qualified=bool(r['quality_passed'] and r['original_grasp_criterion_passed'] and r['braking_passed'])
        rows.append({'label':name,'strategy':a['strategy'] if a else r['mode'],'spatial_kinematic_admitted':a['preferred_plan_admitted'] if a else None,
                     'reference_applied':a['applied_reference'] if a else r['mode'],
                     'quality':r['quality_passed'],'grasp':r['original_grasp_criterion_passed'],'braking':r['braking_passed'],
                     'acceleration_mps2':r['max_brake_payload_acceleration_mps2'],
                     'target_angle_deg':r['max_brake_payload_target_angle_error_deg'],
                     'payload_front_cm':100*r['max_payload_front_travel_m'],'upper_envelope_cm':None if envelope is None else 100*envelope,
                     'clearance_cm':100*r['config']['front_clearance_budget_m'],'envelope_within_budget':envelope_ok,
                     'base_cm':100*r['brake_distance_to_final_m'],'grasp_mm':1000*r['max_relative_change_after_1s_m'],
                     'reference_jerk_mps3':r['peak_object_reference_jerk_mps3'],
                     'reference_jerk_within_5':bool(r['peak_object_reference_jerk_mps3']<=5.+1e-6),
                     'initial_accel_minus_measured_mps2':None if a is None else a['initial_command_acceleration_jump_mps2'],
                     'reference_velocity_jump_at_trigger_mps':handover_v,
                     'reference_acceleration_jump_at_trigger_mps2':handover_a,
                     'example_motion_task_passed':bool(qualified and r['max_brake_payload_acceleration_mps2']<=.8 and r['max_brake_payload_target_angle_error_deg']<=.5 and (envelope_ok if a else r['max_payload_front_travel_m']<=.15)),
                     'admission_wall_ms':None if a is None else a['admission_wall_ms'],'folder':r['folder']})
    old=json.loads((OUT.parent/'stage12/summary.json').read_text());regression=[]
    for version in ['regression','final_regression']:
        for mode in ['fixed','buffer']:
            label='s14_'+version+'_'+mode
            if label not in runs:continue
            a=np.load(Path(old['runs']['s12_regression_'+mode]['folder'])/'joint_trace.npz');b=np.load(Path(runs[label]['folder'])/'joint_trace.npz')
            same=all(np.array_equal(a[k],b[k]) for k in ['time','qpos','qvel','ctrl']);assert same
            regression.append({'version':version,'mode':mode,'exact':same})
    pairs=[]
    for case in ['nominal','accelerating','faster','accelerating_halfdt']:
        if case=='accelerating_halfdt':a='s14_accelerating_scurve_halfdt';b='s14_accelerating_ruckig_halfdt'
        else:a='s14_'+case+'_scurve';b='s14_'+case+'_ruckig'
        if a not in runs or b not in runs:continue
        f,g=runs[a],runs[b]
        assert all(f[k]==g[k] for k in ['scene_sha256','initialization_sha256','runner_sha256','reference_sha256','state_stop_script_sha256'])
        ft,gt=trace(f),trace(g);assert np.array_equal(ft['time_s'],gt['time_s']);pre=ft['time_s']<3.5;post=~pre
        fs=np.load(Path(f['folder'])/'joint_trace.npz');gs=np.load(Path(g['folder'])/'joint_trace.npz')
        same=all(np.array_equal(fs[k][pre],gs[k][pre]) for k in ['qpos','qvel','ctrl']);assert same
        dv=float(max(abs(ft['base_v_mps'][post]-gt['base_v_mps'][post])));dx=float(max(abs(ft['base_x_m'][post]-gt['base_x_m'][post])))
        pairs.append({'case':case,'pretrigger_exact_state_control':same,'base_speed_difference_mps':dv,'base_position_difference_m':dx,
                      'base_motion_comparable':dv<.01 and dx<.002,
                      'both_preferred_references_applied':all(r['reference_admission']['preferred_plan_admitted'] for r in [f,g]),
                      'scurve_acceleration':f['max_brake_payload_acceleration_mps2'],'ruckig_acceleration':g['max_brake_payload_acceleration_mps2'],
                      'scurve_reference_jerk':f['peak_object_reference_jerk_mps3'],'ruckig_reference_jerk':g['peak_object_reference_jerk_mps3']})
    half=[]
    for strategy in ['scurve','ruckig']:
        a='s14_accelerating_'+strategy;b=a+'_halfdt'
        if a not in runs or b not in runs:continue
        f,g=runs[a],runs[b]
        half.append({'strategy':strategy,'main_acceleration':f['max_brake_payload_acceleration_mps2'],'half_acceleration':g['max_brake_payload_acceleration_mps2'],
                     'main_reference_jerk':f['peak_object_reference_jerk_mps3'],'half_reference_jerk':g['peak_object_reference_jerk_mps3'],
                     'same_quality_grasp_brake':all(f[k]==g[k] for k in ['quality_passed','original_grasp_criterion_passed','braking_passed'])})
    result={'run_count':len(runs),'all_snapshot_hashes_verified':True,'default_exact_regression':regression,'pairs':pairs,'half_timestep':half,'metrics':rows,'runs':runs,
            'independent_checks':json.loads((dest/'independent_checks.json').read_text()),'offline_admission_checks':json.loads((dest/'admission_checks.json').read_text()),
            'qualification':'One-shot state-informed reference plus existing high-rate joint/attitude feedback, not MPC or novelty evidence. Spatial admission is sampled. Reference jerk is the physics-step acceleration finite difference; the small command-velocity handover jump is separately reported, so no globally C2 reference or physical jerk guarantee is established. Computation wall time is not injected as simulated command latency.'}
    dump(dest/'summary.json',result)
    if rows:
        with (dest/'runs.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps([{k:r[k] for k in ['label','reference_applied','quality','grasp','braking','acceleration_mps2','reference_jerk_mps3','upper_envelope_cm']} for r in rows]))


if __name__=='__main__':main()
