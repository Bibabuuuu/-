"""Validate finite reference tradeoffs against nominal physical execution."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import mujoco as mj
from build_model import OUT,HERE,dump
from analyze_buffer import trace


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    dest=OUT.parent/'stage21';runs={};rows=[];candidate_rows=[]
    for p in sorted((OUT.parent/'buffer_runs').glob('*s21_*/result.json')):
        r=json.loads(p.read_text());assert r['label'] not in runs;runs[r['label']]=r
    assert set(runs)=={'s21_regression'}|{'s21_'+s+'_'+p for s in ['normal','forward'] for p in ['fixed','acceleration','motion']}
    independent=json.loads((dest/'independent_checks.json').read_text())
    for scenario,outputs in independent['cases'].items():
        selected=outputs['acceleration'];front=selected['frontier_indices']
        for i,row in enumerate(selected['candidate_rows']):candidate_rows.append({'scenario':scenario,'frontier':i in front,**row,'reasons':' | '.join(row['reasons'])})
    for name,r in runs.items():
        p=Path(r['folder']);a=r['reference_admission'];z=trace(r);joint=np.load(p/'joint_trace.npz');post=joint['time']>=3.5
        for file,key in [('scene.xml','scene_sha256'),('initialization.json','initialization_sha256'),('run_snapshot.py','runner_sha256'),
                         ('reference_snapshot.py','reference_sha256'),('state_stop_snapshot.py','state_stop_script_sha256'),
                         ('motion_distribution_snapshot.py','motion_distribution_script_sha256'),('wrench_snapshot.py','wrench_script_sha256'),
                         ('load_sharing_snapshot.py','load_sharing_script_sha256'),('uncertainty_snapshot.py','uncertainty_script_sha256'),
                         ('stop_profile_snapshot.py','stop_profile_script_sha256'),('static_payload_snapshot.py','static_payload_script_sha256')]:assert sha(p/file)==r[key],(name,key)
        # Historical source is verified using that run's own snapshots and hashes.
        assert r['config']['base_T_s']==r['config']['admission']['requested_s']==.6 and r['config']['world_x_feedback']
        assert r['equality_constraints']==0 and a['clearance_m'] is None and a['planning_delay_s']==0
        m=mj.MjModel.from_xml_path(str(p/'scene.xml'));path=np.load(p/'joint_reference_path.npz');names=path['names'].tolist()
        vc={j['name']:j['speed_cap'] for j in r['per_joint']};caps=np.array([vc[n] for n in names]);v=joint['qvel'][post][:,path['vadr']];q=joint['qpos'][post][:,path['qadr']]
        actual_cost=float(np.trapezoid(np.sum((v/caps)**2,axis=1),joint['time'][post]))
        reference=np.load(p/'admission_trace.npz')['samples'];length=r['config']['path_extent_m']+r['config']['path_backward_extent_m']
        u=(reference[:,7]+r['config']['path_backward_extent_m'])/length
        qv=np.polynomial.polynomial.polyval(u,path['first'])*((reference[:,5]-reference[:,2])/length)
        predicted_cost=float(np.trapezoid(np.sum((qv/caps[:,None])**2,axis=0),reference[:,0]))
        selected_A=a['reference_acceleration_mps2'];policy='fixed';front=None
        if 'stop_candidates_script_sha256' in r:
            assert sha(p/'stop_candidates_snapshot.py')==r['stop_candidates_script_sha256']
            selection=json.loads((p/'candidate_selection.json').read_text());assert selection==a['candidate_selection']
            chosen=selection['candidate_rows'][selection['selected_index']];policy=selection['priority'];front=selection['frontier_indices']
            assert selected_A==selection['selected_A_mps2'] and abs(predicted_cost-chosen['normalized_upper_speed_square_integral_s'])<1e-14
            source_key='default_cruise_04' if name.startswith('s21_normal') else 'forward_cruise_05'
            assert selection['candidate_rows']==independent['cases'][source_key][policy]['candidate_rows']
        torso=np.array([n.startswith('torso_') for n in names]);variation=np.sum(np.abs(np.diff(q,axis=0)),axis=0);excursion=np.max(np.abs(q-q[0]),axis=0)
        rows.append({'label':name,'scenario':'regression' if name=='s21_regression' else name.split('_')[1],'policy':policy,'A_setting_mps2':selected_A,
                     'status':r['status'],'model_checks':r['quality_passed'],'base_tracking':r['braking_passed'],'legacy_grasp_diagnostic':r['original_grasp_criterion_passed'],
                     'reference_admitted':a['preferred_plan_admitted'],'object_reference_duration_s':a['candidate_duration_s'],
                     'predicted_peak_acceleration_mps2':float(max(abs(reference[:,6]))),'actual_peak_acceleration_mps2':r['max_brake_payload_acceleration_mps2'],
                     'predicted_upper_motion_cost_s':predicted_cost,'actual_upper_motion_cost_s':actual_cost,
                     'upper_total_variation_rad':float(sum(variation)),'arm_total_variation_rad':float(sum(variation[~torso])),
                     'torso_max_joint_excursion_deg':float(np.rad2deg(max(excursion[torso]))),
                     'angle_error_deg':r['max_brake_payload_target_angle_error_deg'],'grasp_change_mm':1000*r['max_relative_change_after_1s_m'],
                     'upper_envelope_cm':100*r['max_upper_envelope_front_travel_m'],'base_distance_cm':100*r['brake_distance_to_final_m'],
                     'ee_relative_forward_cm':100*r['max_extra_ee_forward_m'],
                     'reference_guard_triggered':r['reference_corridor_guard'] is not None})
    oldr=json.loads((OUT.parent/'stage20/summary.json').read_text())['runs']['s20_normal_waist'];newr=runs['s21_regression']
    old=np.load(Path(oldr['folder'])/'joint_trace.npz');new=np.load(Path(newr['folder'])/'joint_trace.npz')
    exact=all(np.array_equal(old[k],new[k]) for k in ['time','qpos','qvel','ctrl']);assert exact
    comparisons=[];lookup={r['label']:r for r in rows}
    for scenario in ['normal','forward']:
        fixed=runs[f's21_{scenario}_fixed'];fz=trace(fixed)
        for policy in ['acceleration','motion']:
            adaptive=runs[f's21_{scenario}_{policy}'];az=trace(adaptive)
            assert all(fixed[k]==adaptive[k] for k in ['scene_sha256','initialization_sha256','runner_sha256','reference_sha256','motion_distribution_script_sha256'])
            for key in ['base_ref_x_m','base_ref_v_mps','base_ref_a_mps2']:assert np.array_equal(fz[key],az[key])
            f,w=lookup[fixed['label']],lookup[adaptive['label']];post=fz['time_s']>=3.5
            comparisons.append({'scenario':scenario,'policy':policy,'same_base_reference':True,
                                'actual_acceleration_change_percent':100*(w['actual_peak_acceleration_mps2']/f['actual_peak_acceleration_mps2']-1),
                                'actual_upper_motion_cost_change_percent':100*(w['actual_upper_motion_cost_s']/f['actual_upper_motion_cost_s']-1),
                                'max_actual_base_position_difference_m':float(max(abs(fz['base_x_m'][post]-az['base_x_m'][post]))),
                                'max_actual_base_speed_difference_mps':float(max(abs(fz['base_v_mps'][post]-az['base_v_mps'][post])))})
    frozen={OUT/'research_model.xml':'a66346c0839622d469dd4739d5f2a172ab8a1424d86537b43a571925c3c0195e',OUT.parent/'payload_model/payload_scene.xml':'d6701100d6fd49deb2541093ca6fca63acf4ffa676d8cbc8e3a3fe29bb824154',HERE/'run_payload.py':'0d426bd07b83fa7d9e9aa55be086826dd5cbfef845aee8a89e706febeeb15906'}
    for p,h in frozen.items():assert sha(p)==h
    for filename,data in [('runs.csv',rows),('all_candidates.csv',candidate_rows)]:
        with (dest/filename).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    result={'run_count':len(runs),'runs':runs,'metrics':rows,'comparisons':comparisons,'independent_checks':independent,
            'legacy_exact_state_control_regression':exact,'all_snapshots_verified':True,'frozen_models_unchanged':True,
            'qualification':'Finite-family command-state evaluation on offline prepared pose paths; two transparent example priorities, no default preference, global optimum or live-posture replanning. One deterministic run per condition; no statistical generalization.'}
    dump(dest/'summary.json',result)
    dump(dest/'delivery_check.json',{'complete_runs':len(runs),'legacy_exact_regression':exact,'all_snapshots_and_frozen_models_verified':True,
                                   'base_reference_checks':len(comparisons),'physical_failures':[r['label'] for r in rows if r['status']!='completed' or not r['model_checks']]})
    print(json.dumps({'metrics':rows,'comparisons':comparisons}))


if __name__=='__main__':main()
