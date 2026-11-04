"""Audit physical results separately from allocator predictions and static failures."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from build_model import OUT,HERE,dump
from analyze_stage7 import pair
from task_constraints import assess


def main():
    dest=OUT.parent/'stage11';runs={}
    for p in sorted((OUT.parent/'buffer_runs').glob('*/result.json')):
        r=json.loads(p.read_text())
        if r['label'].startswith('s11_'):runs[r['label']]=r
    assert len(runs)==21,len(runs)
    rows=[];details={}
    for name,r in runs.items():
        folder=Path(r['folder'])
        for file,key in [('scene.xml','scene_sha256'),('initialization.json','initialization_sha256'),
                         ('run_snapshot.py','runner_sha256'),('reference_snapshot.py','reference_sha256'),
                         ('wrench_snapshot.py','wrench_script_sha256'),('uncertainty_snapshot.py','uncertainty_script_sha256'),
                         ('stop_profile_snapshot.py','stop_profile_script_sha256'),('static_payload_snapshot.py','static_payload_script_sha256'),
                         ('load_sharing_snapshot.py','load_sharing_script_sha256')]:
            assert hashlib.sha256((folder/file).read_bytes()).hexdigest()==r[key],(name,key)
        assert r['equality_constraints']==0 and r['status']=='completed'
        with (folder/'trace.csv').open(encoding='utf-8') as f:trace=list(csv.DictReader(f))
        t=np.array([float(z['time_s']) for z in trace]);hold=r['config']['hold_only']
        caps={z['name']:z['force_cap'] for z in r['per_joint']}
        phase='static' if hold else 'braking';torques=r['phase_joint_peak_torques_Nm'][phase]
        def peak(prefix):
            eligible={n:v/caps[n] for n,v in torques.items() if n.startswith(prefix)}
            n=max(eligible,key=eligible.get);return eligible[n],n,torques[n]
        arms=max([peak(s+'_arm_') for s in ['left','right']]);torso=peak('torso_')
        sel=t>=5.5 if hold else t>=3.5
        def mean(key):return float(np.mean([float(z[key]) for i,z in enumerate(trace) if sel[i]]))
        rmse=float(np.sqrt(np.mean([(float(z[s+'_vertical_N'])-float(z['desired_'+s+'_vertical_N']))**2
                                   for i,z in enumerate(trace) if sel[i] for s in ['left','right']])))
        detail={'arm_peak_normalized':arms[0],'arm_joint':arms[1],'arm_torque_Nm':arms[2],
                'torso_peak_normalized':torso[0],'torso_joint':torso[1],'torso_torque_Nm':torso[2],
                'desired_left_vertical_mean_N':mean('desired_left_vertical_N'),'actual_left_vertical_mean_N':mean('left_vertical_N'),
                'desired_right_vertical_mean_N':mean('desired_right_vertical_N'),'actual_right_vertical_mean_N':mean('right_vertical_N'),
                'force_tracking_rmse_N':rmse,'trace_window':'5.5-6s' if hold else '3.5-6s',
                'minimum_normal_sum_N':min(float(z['normal_sum_N']) for i,z in enumerate(trace) if sel[i]),
                'maximum_local_friction_use':max(float(z['max_contact_friction_use']) for i,z in enumerate(trace) if sel[i])}
        if r['config']['load_allocation']!='equal':
            a=r['allocator'];assert a['accepted']==a['calls'] and a['fallbacks']==0
            assert a['max_balance_error']<1e-10 and a['max_constraint_violation']<1e-7 and a['max_stationarity_error']<1e-7
            detail['allocator_objective_reduction_percent_mean']=float(np.mean([100*(1-float(z['allocator_objective_after'])/float(z['allocator_objective_before'])) for i,z in enumerate(trace) if sel[i]]))
        details[name]=detail
        task=assess(r,acceleration_limit=.8,angle_limit_deg=.5,clearance_m=.15) if not hold else None
        rows.append({'label':name,'static_only':hold,'quality':r['quality_passed'],'grasp':r['original_grasp_criterion_passed'],
                     'braking':None if hold else r['braking_passed'],'task_passed':None if hold else task['passed'],
                     'acceleration_mps2':None if hold else r['max_brake_payload_acceleration_mps2'],
                     'target_angle_deg':r['max_brake_payload_target_angle_error_deg'],
                     'front_cm':None if hold else 100*r['max_payload_front_travel_m'],
                     'grasp_mm':1000*r['max_relative_change_after_1s_m'],
                     'first_grasp_violation_s':r['first_grasp_threshold_violation_s'],
                     'arm_torque_fraction':arms[0],'torso_torque_fraction':torso[0],
                     'peak_friction_use':detail['maximum_local_friction_use'],'folder':r['folder']})
    old=json.loads((OUT.parent/'stage9/summary.json').read_text());regression=[]
    for version in ['nominal','final']:
        for mode in ['fixed','buffer']:
            a=np.load(Path(old['runs']['s9_nominal_'+mode]['folder'])/'joint_trace.npz')
            b=np.load(Path(runs[f's11_{version}_equal_'+mode]['folder'])/'joint_trace.npz')
            exact=all(np.array_equal(a[k],b[k]) for k in ['time','qpos','qvel','ctrl']);assert exact
            regression.append({'version':version,'mode':mode,'exact':exact})
    pairs=[]
    for case in ['nominal','offset','posture']:
        for allocation in ['equal','qp']:
            label=f's11_{case}_{allocation}';p=pair(runs[label+'_fixed'],runs[label+'_buffer'],case+'_'+allocation)
            p['static_hold_qualified']=case!='offset' or runs[f's11_offset_{allocation}_hold']['original_grasp_criterion_passed']
            p['emergency_specific_evidence']=p['qualified_performance_comparison'] and p['static_hold_qualified']
            pairs.append(p)
    half=[]
    for allocation in ['equal','qp','lever']:
        a=runs[f's11_offset_{allocation}_buffer'];b=runs[f's11_offset_{allocation}_halfdt']
        half.append({'allocation':allocation,'dt_milli_acceleration':a['max_brake_payload_acceleration_mps2'],
                     'dt_half_acceleration':b['max_brake_payload_acceleration_mps2'],
                     'dt_milli_grasp_mm':1000*a['max_relative_change_after_1s_m'],'dt_half_grasp_mm':1000*b['max_relative_change_after_1s_m'],
                     'same_quality_grasp_classification':all(a[k]==b[k] for k in ['quality_passed','original_grasp_criterion_passed','braking_passed'])})
    result={'run_count':len(runs),'all_snapshot_hashes_verified':True,'nominal_exact_regression':regression,
            'quality_count':sum(r['quality'] for r in rows),'static_hold_runs':sum(r['static_only'] for r in rows),
            'grasp_pass_count':sum(r['grasp'] for r in rows),'pairs':pairs,'half_timestep':half,'metrics':rows,'details':details,'runs':runs,
            'qp_validation':json.loads((dest/'qp_validation.json').read_text()),
            'posture_screen':json.loads((dest/'posture_screen.json').read_text()),
            'scripts':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['load_sharing.py','run_buffer.py','run_stage11.py','prepare_stage11.py','check_stage11_qp.py','analyze_stage11.py']},
            'limits':'Small contact simulation; fixed torso per run; torso posture chosen offline; predicted wrench allocation is not measured-force closed-loop control. Offset equal-allocation already fails static hold; not solely a braking failure.'}
    dump(dest/'summary.json',result)
    with (dest/'runs.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({k:result[k] for k in ['run_count','quality_count','grasp_pass_count','nominal_exact_regression','half_timestep']}))
    print(json.dumps([{k:v for k,v in r.items() if k!='folder'} for r in rows]))


if __name__=='__main__':main()
