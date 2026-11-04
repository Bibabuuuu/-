"""Audit full-load-path and reference-family tests with explicit task contracts."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from build_model import OUT, HERE, dump
from analyze_stage7 import pair
from task_constraints import assess, reference_options
from stop_profiles import check_profile

DEST=OUT.parent/'stage8'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    all_runs={}
    for p in sorted((OUT.parent/'buffer_runs').glob('*/result.json')):
        r=json.loads(p.read_text());all_runs[r['label']]=r
    categories=['arms','upper','mass_under20','mass_over20','com_back2cm','com_front2cm','delay20ms','profile']
    labels=[f's8_{c}_{m}' for c in categories for m in ['fixed','buffer']]
    labels += [f's8_duration_{d}ms' for d in [700,800,900,1050]]
    labels += ['s8_profile_'+c for c in ['halfdt','mass_under20','delay20ms','gentle']]
    runs={k:all_runs[k] for k in labels}
    for r in runs.values():
        p=Path(r['folder'])
        for filename,key in [('scene.xml','scene_sha256'),('initialization.json','initialization_sha256'),
                             ('run_snapshot.py','runner_sha256'),('reference_snapshot.py','reference_sha256'),
                             ('wrench_snapshot.py','wrench_script_sha256'),('uncertainty_snapshot.py','uncertainty_script_sha256')]:
            assert sha(p/filename)==r[key],(r['label'],filename)
        if 'stop_profile_script_sha256' in r:
            assert sha(p/'stop_profile_snapshot.py')==r['stop_profile_script_sha256']
        assert r['equality_constraints']==0
        assert r['config']['mass_kg']==1.
        if r['config']['upper_delay_steps']:
            z=np.load(p/'delay_trace.npz');n=r['config']['upper_delay_steps']
            expected=z['computed_upper_ctrl'][np.maximum(0,np.arange(len(z['computed_upper_ctrl']))-n)]
            assert np.array_equal(expected,z['applied_upper_ctrl'])
    assert len({r['scene_sha256'] for k,r in runs.items() if k!='s8_profile_halfdt'})==1
    assert len({r['initialization_sha256'] for r in runs.values()})==1
    assert len({r['wrench_script_sha256'] for r in runs.values()})==1
    paired=[]
    for c in categories:
        f,b=runs[f's8_{c}_fixed'],runs[f's8_{c}_buffer']
        assert f['config']['payload_wrench_scope']==b['config']['payload_wrench_scope']
        paired.append(pair(f,b,c))
    # Duration probes share one fixed run with matching runner/model/initial state.
    duration_pairs=[pair(runs['s8_profile_fixed'],runs[f's8_duration_{d}ms'],f'duration_{d}ms') for d in [700,800,900,1050]]
    assert all(p['qualified_performance_comparison'] for p in paired+duration_pairs)
    regression=[]
    for mode in ['fixed','buffer']:
        a=np.load(Path(all_runs['s7_nominal_'+mode]['folder'])/'joint_trace.npz')
        b=np.load(Path(runs['s8_arms_'+mode]['folder'])/'joint_trace.npz')
        passed=all(np.array_equal(a[k],b[k]) for k in ['time','qpos','qvel','ctrl']);assert passed
        regression.append({'mode':mode,'historical_default_exact_match':passed})
    # Adding the optional profile must leave the corrected cubic fixed baseline unchanged.
    a=np.load(Path(runs['s8_upper_fixed']['folder'])/'joint_trace.npz')
    b=np.load(Path(runs['s8_profile_fixed']['folder'])/'joint_trace.npz')
    optional_match=all(np.array_equal(a[k],b[k]) for k in ['time','qpos','qvel','ctrl']);assert optional_match
    tasks={
        'tight':{'acceleration_limit':.8,'angle_limit_deg':.5,'clearance_m':.15},
        'gentle':{'acceleration_limit':.6,'angle_limit_deg':.5,'clearance_m':.18},
        'roomy':{'acceleration_limit':.6,'angle_limit_deg':.5,'clearance_m':.22},
    }
    task_results={name:{label:assess(r,**lim) for label,r in runs.items() if r['config']['payload_wrench_scope']=='upper'} for name,lim in tasks.items()}
    rows=[]
    for r in runs.values():
        rows.append({'label':r['label'],'quality':r['quality_passed'],'grasp':r['original_grasp_criterion_passed'],
                     'braking':r['braking_passed'],'wrench_scope':r['config']['payload_wrench_scope'],
                     'object_T_s':r['config']['object_T_s'],'profile':r['config'].get('object_profile','cubic'),
                     'acceleration_mps2':r['max_brake_payload_acceleration_mps2'],
                     'target_angle_deg':r['max_brake_payload_target_angle_error_deg'],
                     'prebrake_target_angle_deg':r['max_prebrake_payload_target_angle_error_deg'],
                     'relative_grasp_mm':1000*r['max_relative_change_after_1s_m'],
                     'front_travel_cm':100*r['max_payload_front_travel_m'],
                     'arm_extra_cm':100*r['max_extra_ee_forward_m'],
                     'base_stop_cm':100*r['brake_distance_to_final_m'],'folder':r['folder']})
    reference_windows={k:reference_options(v['acceleration_limit'],v['clearance_m']) for k,v in tasks.items()}
    result={'run_count':len(runs),'quality_grasp_braking_pass_count':sum(r['quality_passed'] and r['original_grasp_criterion_passed'] and r['braking_passed'] for r in runs.values()),
            'all_snapshot_hashes_verified':True,'main_scene_hash_identical':True,'delay_sequences_verified':True,
            'regression':regression,'optional_profile_fixed_baseline_exact_match':optional_match,
            'pairs':paired,'duration_pairs':duration_pairs,'tasks':tasks,'task_results':task_results,'reference_windows':reference_windows,
            'profile_check':check_profile(),'runs':runs,'metrics':rows,
            'scripts':{n:sha(HERE/n) for n in ['analyze_stage8.py','run_stage8.py','task_constraints.py','check_load_path.py','stop_profiles.py']},
            'scope':'All limits are illustrative. Offline physics confirms selected candidates only; neither profile rejection nor success is a global or robust safety claim.'}
    dump(DEST/'summary.json',result)
    for name,entries in [('runs.csv',rows),('pairs.csv',paired+duration_pairs)]:
        with (DEST/name).open('w',encoding='utf-8-sig',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=list(entries[0]));w.writeheader();w.writerows(entries)
    print(json.dumps({'runs':len(runs),'quality_grasp_braking_pass_count':result['quality_grasp_braking_pass_count'],
                      'paired_comparisons':len(paired)+len(duration_pairs),
                      'tight_probe':{k:{'passed':v['passed'],'failed':v['failed_checks']} for k,v in task_results['tight'].items() if k.startswith('s8_profile_')},
                      'gentle_probe':task_results['gentle']['s8_profile_gentle']},ensure_ascii=False))


if __name__=='__main__':main()
