"""Same stop task, independently measured joint motion and physical outcomes."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import mujoco as mj
from build_model import OUT,HERE,dump
from analyze_buffer import trace


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    dest=OUT.parent/'stage20';runs={};rows=[];extra_checks={}
    for p in sorted((OUT.parent/'buffer_runs').glob('*s20_*/result.json')):
        r=json.loads(p.read_text());assert r['label'] not in runs;runs[r['label']]=r
    assert set(runs)=={'s20_regression','s20_normal_arms','s20_normal_waist','s20_fast_arms','s20_fast_waist'}
    for name,r in runs.items():
        folder=Path(r['folder']);a=r['reference_admission'];z=trace(r)
        for file,key in [('scene.xml','scene_sha256'),('initialization.json','initialization_sha256'),('run_snapshot.py','runner_sha256'),
                         ('reference_snapshot.py','reference_sha256'),('state_stop_snapshot.py','state_stop_script_sha256'),
                         ('wrench_snapshot.py','wrench_script_sha256'),('load_sharing_snapshot.py','load_sharing_script_sha256'),
                         ('uncertainty_snapshot.py','uncertainty_script_sha256'),('stop_profile_snapshot.py','stop_profile_script_sha256'),
                         ('static_payload_snapshot.py','static_payload_script_sha256')]:
            assert sha(folder/file)==r[key],(name,key)
        # Compare historical runs with their recorded snapshots, not later source revisions.
        assert r['config']['admission']['requested_s']==r['config']['base_T_s']==.6
        assert a['clearance_m'] is None and a['space_reserve_m']==a['stroke_reserve_m']==0.
        assert r['config']['world_x_feedback'] and r['equality_constraints']==0
        if r['config']['motion_distribution']!='legacy_arms':
            assert sha(folder/'motion_distribution_snapshot.py')==r['motion_distribution_script_sha256']
            extra_checks[name]={'joint_reference_path_sha256':sha(folder/'joint_reference_path.npz')}
        m=mj.MjModel.from_xml_path(str(folder/'scene.xml'));joint=np.load(folder/'joint_trace.npz');post=joint['time']>=3.5
        jnames=[j['name'] for j in r['per_joint']];caps={j['name']:j['speed_cap'] for j in r['per_joint']}
        groups={g:[n for n in jnames if (n.startswith('torso_') if g=='torso' else '_arm_' in n)] for g in ['torso','arms']}
        measures={}
        for group,names in groups.items():
            ids=[m.joint(n).id for n in names];qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids]
            q=joint['qpos'][post][:,qa];v=joint['qvel'][post][:,va];t=joint['time'][post]
            variation=np.sum(np.abs(np.diff(q,axis=0)),axis=0);excursion=np.max(np.abs(q-q[0]),axis=0)
            measures[group]={'total_variation_rad':float(sum(variation)),'max_excursion_deg':float(np.rad2deg(max(excursion))),
                             'sampled_peak_speed_fraction':float(np.max(np.abs(v)/np.array([caps[n] for n in names]))),
                             'normalized_speed_square_integral_s':float(np.trapezoid(np.sum((v/np.array([caps[n] for n in names]))**2,axis=1),t)),
                             'minimum_range_margin_rad':float(np.min(np.minimum(q-m.jnt_range[ids,0],m.jnt_range[ids,1]-q))),
                             'per_joint_excursion_deg':dict(zip(names,np.rad2deg(excursion).tolist()))}
        extra_checks.setdefault(name,{})['joint_measures']=measures
        rows.append({'label':name,'status':r['status'],'model_checks':r['quality_passed'],'base_tracking':r['braking_passed'],
                     'legacy_grasp_diagnostic':r['original_grasp_criterion_passed'],'reference_admitted':a['preferred_plan_admitted'],
                     'acceleration_mps2':r['max_brake_payload_acceleration_mps2'],'angle_error_deg':r['max_brake_payload_target_angle_error_deg'],
                     'grasp_change_mm':1000*r['max_relative_change_after_1s_m'],'world_reference_error_mm':1000*r['max_brake_payload_world_reference_x_error_m'],
                     'arm_total_variation_rad':measures['arms']['total_variation_rad'],'arm_max_excursion_deg':measures['arms']['max_excursion_deg'],
                     'torso_max_excursion_deg':measures['torso']['max_excursion_deg'],
                     'arm_sampled_peak_speed_fraction':measures['arms']['sampled_peak_speed_fraction'],
                     'arm_normalized_speed_square_integral_s':measures['arms']['normalized_speed_square_integral_s'],
                     'torso_normalized_speed_square_integral_s':measures['torso']['normalized_speed_square_integral_s'],
                     'arm_range_margin_rad':measures['arms']['minimum_range_margin_rad'],'upper_envelope_cm':100*r['max_upper_envelope_front_travel_m'],
                     'base_distance_cm':100*r['brake_distance_to_final_m'],'sampled_motion_joint_error_deg':float(np.rad2deg(max(z['max_motion_joint_reference_error_rad'][post]))),
                     'reference_guard_triggered':r['reference_corridor_guard'] is not None})
    prior=json.loads((OUT.parent/'stage19/summary.json').read_text())['runs']['s19_nominal']
    old=np.load(Path(prior['folder'])/'joint_trace.npz');new=np.load(Path(runs['s20_regression']['folder'])/'joint_trace.npz')
    exact=all(np.array_equal(old[k],new[k]) for k in ['time','qpos','qvel','ctrl']);assert exact
    metrics={r['label']:r for r in rows};pairs=[]
    for case in ['normal','fast']:
        left=runs[f's20_{case}_arms'];right=runs[f's20_{case}_waist'];f,w=metrics[left['label']],metrics[right['label']]
        assert all(left[k]==right[k] for k in ['scene_sha256','initialization_sha256','runner_sha256','reference_sha256','state_stop_script_sha256'])
        fl=np.load(Path(left['folder'])/'admission_trace.npz')['samples'];wl=np.load(Path(right['folder'])/'admission_trace.npz')['samples']
        # Last column is the geometry-dependent envelope and is expected to differ.
        same_reference=np.array_equal(fl[:,:8],wl[:,:8]);assert same_reference
        ft,wt=trace(left),trace(right);post=ft['time_s']>=3.5;pre=~post
        assert np.array_equal(ft['time_s'],wt['time_s'])
        pairs.append({'case':case,'same_base_object_reference_and_forecast_relative_motion':same_reference,
                      'arm_total_variation_reduction_percent':100*(1-w['arm_total_variation_rad']/f['arm_total_variation_rad']),
                      'arm_speed_square_integral_reduction_percent':100*(1-w['arm_normalized_speed_square_integral_s']/f['arm_normalized_speed_square_integral_s']),
                      'torso_max_excursion_deg':w['torso_max_excursion_deg'],'acceleration_arms':f['acceleration_mps2'],'acceleration_waist':w['acceleration_mps2'],
                      'angle_arms_deg':f['angle_error_deg'],'angle_waist_deg':w['angle_error_deg'],
                      'grasp_arms_mm':f['grasp_change_mm'],'grasp_waist_mm':w['grasp_change_mm'],
                      'envelope_difference_mm':10*(w['upper_envelope_cm']-f['upper_envelope_cm']),
                      'max_actual_braking_base_position_difference_m':float(max(abs(ft['base_x_m'][post]-wt['base_x_m'][post]))),
                      'max_actual_braking_base_speed_difference_mps':float(max(abs(ft['base_v_mps'][post]-wt['base_v_mps'][post]))),
                      'max_pretrigger_object_position_difference_m':float(max(abs(ft['object_com_x_m'][pre]-wt['object_com_x_m'][pre]))),
                      'qualification':'Both mappings active from initialization; pretrigger physical states need not be identical. Joint motion metrics use stored 100Hz samples, not mechanical work/energy.'})
    frozen={OUT/'research_model.xml':'a66346c0839622d469dd4739d5f2a172ab8a1424d86537b43a571925c3c0195e',
            OUT.parent/'payload_model/payload_scene.xml':'d6701100d6fd49deb2541093ca6fca63acf4ffa676d8cbc8e3a3fe29bb824154',
            HERE/'run_payload.py':'0d426bd07b83fa7d9e9aa55be086826dd5cbfef845aee8a89e706febeeb15906'}
    for p,h in frozen.items():assert sha(p)==h
    result={'run_count':len(runs),'runs':runs,'metrics':rows,'pairs':pairs,'independent_checks':json.loads((dest/'independent_checks.json').read_text()),
            'legacy_exact_state_control_regression':exact,'all_snapshots_verified':True,'frozen_models_unchanged':True,'extra_checks':extra_checks,
            'qualification':'Offline local Jacobian QP and checked polynomial paths; nominal uncalibrated .5kg straight-line simulation, fixed object orientation. No online QP/WBC, torque saturation research, arbitrary-object safety or novelty claim.'}
    dump(dest/'summary.json',result)
    with (dest/'runs.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dump(dest/'delivery_check.json',{'complete_runs':len(runs),'legacy_exact_regression':exact,'same_reference_pairs':len(pairs),
                                   'snapshots_and_frozen_models_verified':True,'physical_failures':[r['label'] for r in rows if r['status']!='completed' or not r['model_checks']]})
    print(json.dumps({'metrics':rows,'pairs':pairs}))


if __name__=='__main__':main()
