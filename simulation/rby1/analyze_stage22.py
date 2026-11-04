"""Recorded-state and reference checks for stopping during independent delivery."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import mujoco as mj
from build_model import OUT,HERE,dump
from analyze_buffer import trace


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    dest=OUT.parent/'stage22';independent=json.loads((dest/'independent_checks.json').read_text());runs={};rows=[]
    for p in sorted((OUT.parent/'buffer_runs').glob('*s22v2_*/result.json')):
        r=json.loads(p.read_text());assert r['label'] not in runs;runs[r['label']]=r
    assert set(runs)=={'s22v2_'+x for x in ['regression','early','middle','late','halfdt']}
    for name,r in runs.items():
        p=Path(r['folder']);a=r['reference_admission'];z=trace(r);joint=np.load(p/'joint_trace.npz');samples=np.load(p/'admission_trace.npz')['samples']
        for file,key in [('scene.xml','scene_sha256'),('initialization.json','initialization_sha256'),('run_snapshot.py','runner_sha256'),
                         ('reference_snapshot.py','reference_sha256'),('state_stop_snapshot.py','state_stop_script_sha256'),
                         ('motion_distribution_snapshot.py','motion_distribution_script_sha256'),('wrench_snapshot.py','wrench_script_sha256'),
                         ('load_sharing_snapshot.py','load_sharing_script_sha256'),('uncertainty_snapshot.py','uncertainty_script_sha256'),
                         ('stop_profile_snapshot.py','stop_profile_script_sha256'),('static_payload_snapshot.py','static_payload_script_sha256')]:assert sha(p/file)==r[key],(name,key)
        for filename,key in [('run_buffer.py','runner_sha256'),('state_stop_reference.py','state_stop_script_sha256'),('motion_distribution.py','motion_distribution_script_sha256')]:
            assert r[key]==independent['source_sha256'][filename],(name,'preflight code version',filename)
        assert r['config']['base_T_s']==r['config']['admission']['requested_s']==.6
        assert r['config']['world_x_feedback'] and r['equality_constraints']==0 and a['clearance_m'] is None and a['planning_delay_s']==0
        assert max(abs(np.array(a['planned_handover_reference_jump'])))<1e-12
        assert max(abs(samples[0,4:7]-np.array(a['handover_left_reference'])))<1e-12
        assert abs(samples[0,-1])<1e-12 and a['reference_acceleration_mps2']==.78 and a['reference_jerk_mps3']==5.
        assert r['peak_object_reference_jerk_mps3']<=5.+1e-7
        task=a.get('delivery_task');phase=None;rel=[0.,0.,0.]
        if task:
            assert sha(p/'task_reference_snapshot.py')==r['task_reference_script_sha256']==independent['source_sha256']['delivery_stop.py']
            phase=task['trigger_progress'];rel=task['relative_reference_at_trigger']
            assert abs(a['initial_command_relative_displacement_m']-rel[0])<1e-12
            assert abs(samples[0,7]-rel[0])<1e-12
            assert np.allclose(a['handover_left_reference'],independent['cases'][str(phase)]['object_pva_at_trigger'],atol=1e-12,rtol=0)
        m=mj.MjModel.from_xml_path(str(p/'scene.xml'));names=[f'torso_{i}' for i in range(6)]+[f'{s}_arm_{i}' for s in ['left','right'] for i in range(7)]
        qa=np.array([m.joint(n).qposadr[0] for n in names]);va=np.array([m.joint(n).dofadr[0] for n in names]);post=joint['time']>=3.5
        trigger=int(np.argmin(abs(joint['time']-3.5)));peak_upper_v_at_trigger=float(max(abs(joint['qvel'][trigger,va])))
        if task:assert peak_upper_v_at_trigger>1e-4,'Upper body must actually be moving at handover'
        measured=a['measured_object_velocity_mps'];command=a['otg_initial_velocity_mps']
        rows.append({'label':name,'phase':phase,'dt_ms':1000*r['config']['dt_s'],'status':r['status'],
                     'model_checks':r['quality_passed'],'base_tracking':r['braking_passed'],'legacy_grasp_diagnostic':r['original_grasp_criterion_passed'],
                     'reference_admitted':a['preferred_plan_admitted'],'delivery_position_at_trigger_mm':1000*rel[0],
                     'delivery_velocity_at_trigger_mps':rel[1],'delivery_acceleration_at_trigger_mps2':rel[2],
                     'object_command_velocity_at_trigger_mps':command,'object_measured_velocity_at_trigger_mps':measured,
                     'object_velocity_handover_tracking_error_mps':measured-command,'sampled_peak_upper_joint_speed_at_trigger_radps':peak_upper_v_at_trigger,
                     'object_stop_reference_duration_s':a['candidate_duration_s'],'peak_reference_jerk_mps3':r['peak_object_reference_jerk_mps3'],
                     'actual_peak_object_acceleration_mps2':r['max_brake_payload_acceleration_mps2'],
                     'object_angle_error_deg':r['max_brake_payload_target_angle_error_deg'],'grasp_change_mm':1000*r['max_relative_change_after_1s_m'],
                     'extra_ee_forward_after_trigger_cm':100*r['max_extra_ee_forward_m'],'upper_envelope_after_trigger_cm':100*r['max_upper_envelope_front_travel_m'],
                     'world_reference_error_mm':1000*r['max_brake_payload_world_reference_x_error_m'],'base_distance_cm':100*r['brake_distance_to_final_m'],
                     'sampled_max_joint_reference_error_deg':float(np.rad2deg(max(z['max_motion_joint_reference_error_rad'][post]))),
                     'reference_guard_triggered':r['reference_corridor_guard'] is not None,'admission_wall_ms':a['admission_wall_ms']})
    prior=json.loads((OUT.parent/'stage21/summary.json').read_text())['runs']['s21_normal_fixed']
    old=np.load(Path(prior['folder'])/'joint_trace.npz');new=np.load(Path(runs['s22v2_regression']['folder'])/'joint_trace.npz')
    exact=all(np.array_equal(old[k],new[k]) for k in ['time','qpos','qvel','ctrl']);assert exact
    baseline=trace(runs['s22v2_regression']);base_comparisons=[]
    for case in ['early','middle','late']:
        r=runs['s22v2_'+case];z=trace(r)
        for key in ['base_ref_x_m','base_ref_v_mps','base_ref_a_mps2']:assert np.array_equal(z[key],baseline[key])
        base_comparisons.append({'case':case,'same_base_reference':True,'actual_base_position_max_difference_m':float(max(abs(z['base_x_m']-baseline['base_x_m']))),
                                 'actual_base_speed_max_difference_mps':float(max(abs(z['base_v_mps']-baseline['base_v_mps'])))})
    mainrun,half=runs['s22v2_middle'],runs['s22v2_halfdt']
    same_plan=np.array_equal(np.load(Path(mainrun['folder'])/'admission_trace.npz')['samples'],np.load(Path(half['folder'])/'admission_trace.npz')['samples']);assert same_plan
    step_check={'reference_plan_exactly_equal':same_plan,'main_acceleration_mps2':mainrun['max_brake_payload_acceleration_mps2'],
                'half_acceleration_mps2':half['max_brake_payload_acceleration_mps2'],'same_model_grasp_base_checks':all(mainrun[k]==half[k] for k in ['quality_passed','original_grasp_criterion_passed','braking_passed'])}
    first=json.loads((dest/'first_pass_summary.json').read_text());boundary_regressions=[]
    for case in ['regression','early','middle','late','halfdt']:
        before=first['runs']['s22_'+case];after=runs['s22v2_'+case]
        old_state=np.load(Path(before['folder'])/'joint_trace.npz');new_state=np.load(Path(after['folder'])/'joint_trace.npz')
        unchanged=all(np.array_equal(old_state[k],new_state[k]) for k in ['time','qpos','qvel','ctrl']);assert unchanged
        boundary_regressions.append({'case':case,'physical_state_and_control_exactly_unchanged':unchanged,
                                     'first_pass_reported_world_error_mm':1000*before['max_brake_payload_world_reference_x_error_m'],
                                     'corrected_world_error_mm':1000*after['max_brake_payload_world_reference_x_error_m']})
    frozen={OUT/'research_model.xml':'a66346c0839622d469dd4739d5f2a172ab8a1424d86537b43a571925c3c0195e',OUT.parent/'payload_model/payload_scene.xml':'d6701100d6fd49deb2541093ca6fca63acf4ffa676d8cbc8e3a3fe29bb824154',HERE/'run_payload.py':'0d426bd07b83fa7d9e9aa55be086826dd5cbfef845aee8a89e706febeeb15906'}
    for p,h in frozen.items():assert sha(p)==h
    result={'run_count':len(runs),'runs':runs,'metrics':rows,'independent_checks':independent,'legacy_exact_state_control_regression':exact,
            'half_timestep':step_check,'base_comparisons':base_comparisons,'boundary_fix_regressions':boundary_regressions,'all_snapshots_verified':True,'frozen_models_unchanged':True,
            'qualification':'Continuous command-state handover during one-dimensional delivery, offline prepared waist-arm path, nominal .5kg model and 6s window. No physical state resets, online replan, global stability, damage or novelty guarantee.'}
    dump(dest/'summary.json',result)
    with (dest/'runs.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dump(dest/'delivery_check.json',{'complete_runs':len(runs),'exact_legacy_regression':exact,'all_snapshots_and_frozen_models_verified':True,
                                   'same_plan_half_timestep':same_plan,'source_versions_match_preflight':True,
                                   'first_pass_complete_runs_retained':first['run_count'],'boundary_fix_exact_state_control_regressions':len(boundary_regressions),
                                   'physical_failures':[r['label'] for r in rows if r['status']!='completed' or not r['model_checks']]})
    print(json.dumps({'metrics':rows,'half_timestep':step_check}))


if __name__=='__main__':main()
