"""Audit calibration observations, historical controls and unchanged task limits."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from build_model import OUT,HERE,dump
from analyze_stage7 import pair
from static_payload import fit_static
from task_constraints import assess

DEST=OUT.parent/'stage9'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    old=json.loads((OUT.parent/'stage8/summary.json').read_text())
    all_runs={}
    for p in sorted((OUT.parent/'buffer_runs').glob('*/result.json')):
        r=json.loads(p.read_text());all_runs[r['label']]=r
    groups=['nominal','margin_delay20ms','cal_nominal','cal_mass_under20','cal_com_back2cm','cal_com_front2cm','cal_bias','combined']
    labels=[f's9_{g}_{m}' for g in groups for m in ['fixed','buffer']]
    labels += ['s9_mass_under20_halfdt','s9_delay20ms_halfdt','s9_cal_mass_under20_halfdt','s9_margin_delay20ms_halfdt']
    optional=['s9_margin68_delay20ms_buffer','s9_margin68_delay20ms_halfdt','s9_combined_halfdt']
    labels += [k for k in optional if k in all_runs]
    runs={k:all_runs[k] for k in labels};fits=[];rows=[]
    for r in runs.values():
        p=Path(r['folder'])
        for file,key in [('scene.xml','scene_sha256'),('initialization.json','initialization_sha256'),
            ('run_snapshot.py','runner_sha256'),('reference_snapshot.py','reference_sha256'),
            ('wrench_snapshot.py','wrench_script_sha256'),('uncertainty_snapshot.py','uncertainty_script_sha256'),
            ('stop_profile_snapshot.py','stop_profile_script_sha256'),('static_payload_snapshot.py','static_payload_script_sha256')]:
            assert sha(p/file)==r[key],(r['label'],key)
        assert r['equality_constraints']==0 and r['config']['mass_kg']==1.
        if r['config']['upper_delay_steps']:
            z=np.load(p/'delay_trace.npz');n=r['config']['upper_delay_steps']
            assert np.array_equal(z['applied_upper_ctrl'],z['computed_upper_ctrl'][np.maximum(0,np.arange(len(z['computed_upper_ctrl']))-n)])
        if r['config']['static_calibration']:
            z=np.load(p/'calibration_trace.npz');assert min(z['time'])>=1.-1e-12 and max(z['time'])<=1.35+1e-12
            samples=list(zip(z['rotation_body_to_world'],z['world_force_N'],z['world_moment_at_origin_Nm']))
            fitted=fit_static(samples,r['config']['vertical_com_prior_m'])
            assert all(fitted[k]==r['static_calibration_result'][k] for k in fitted)
            assert r['static_calibration_result']['fit_at_s']>=max(z['time'])
            fits.append({'label':r['label'],**r['static_calibration_result']})
        task=assess(r,acceleration_limit=.8,angle_limit_deg=.5,clearance_m=.15)
        rows.append({'label':r['label'],'quality':r['quality_passed'],'grasp':r['original_grasp_criterion_passed'],
                     'braking':r['braking_passed'],'task_passed':task['passed'],'failed_checks':','.join(task['failed_checks']),
                     'acceleration_mps2':r['max_brake_payload_acceleration_mps2'],
                     'angle_deg':r['max_brake_payload_target_angle_error_deg'],
                     'pre_angle_deg':r['max_prebrake_payload_target_angle_error_deg'],
                     'front_cm':100*r['max_payload_front_travel_m'],'base_cm':100*r['brake_distance_to_final_m'],
                     'grasp_mm':1000*r['max_relative_change_after_1s_m'],'folder':r['folder']})
    pairs=[pair(runs[f's9_{g}_fixed'],runs[f's9_{g}_buffer'],g) for g in groups]
    regression=[]
    for mode in ['fixed','buffer']:
        a=np.load(Path(old['runs']['s8_profile_'+mode]['folder'])/'joint_trace.npz')
        b=np.load(Path(runs['s9_nominal_'+mode]['folder'])/'joint_trace.npz')
        match=all(np.array_equal(a[k],b[k]) for k in ['time','qpos','qvel','ctrl']);assert match
        regression.append({'mode':mode,'exact':match})
    assert len({r['scene_sha256'] for r in runs.values() if r['config']['dt_s']==.001})==1
    # Uncalibrated reference uses the initial controller estimate. Fitting is a pre-braking intervention.
    comparison=[{'condition':name,'uncalibrated_label':'s8_profile_'+name,
                 'uncalibrated':old['runs']['s8_profile_'+name],
                 'calibrated':runs['s9_cal_'+name+'_buffer']} for name in ['mass_under20']]
    result={'run_count':len(runs),'original_quality_grasp_braking_count':sum(r['quality'] and r['grasp'] and r['braking'] for r in rows),
            'task_passed_count':sum(r['task_passed'] for r in rows),'all_snapshot_hashes_verified':True,
            'nominal_exact_regression':regression,'actual_delay_arrays_verified':True,'calibration_recomputed_from_observations':True,
            'pairs':pairs,'calibration_fits':fits,'runs':runs,'metrics':rows,'mass_calibration_comparison':comparison,
            'illustrative_task':{'acceleration_mps2':.8,'angle_deg':.5,'clearance_cm':15},
            'scripts':{n:sha(HERE/n) for n in ['run_buffer.py','static_payload.py','run_stage9.py','analyze_stage9.py']},
            'scope':'Finite deterministic checks with ideal contact-wrench proxies and known vertical COM/inertia priors. No hardware or novelty claim.'}
    dump(DEST/'summary.json',result)
    for name,entries in [('runs.csv',rows),('pairs.csv',pairs)]:
        with (DEST/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(entries[0]));w.writeheader();w.writerows(entries)
    print(json.dumps({k:result[k] for k in ['run_count','original_quality_grasp_braking_count','task_passed_count']},ensure_ascii=False))
    print(json.dumps(rows,ensure_ascii=False))


if __name__=='__main__':main()
