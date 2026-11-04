"""Keep persistence, normal motion and braking qualifications separate."""
import csv,hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from build_model import OUT,HERE,dump
from analyze_buffer import trace
from analyze_stage7 import pair
from task_constraints import assess


DEST=OUT.parent/'stage12'


def main():
    DEST.mkdir(exist_ok=True);runs={};failures=[]
    for p in sorted((OUT.parent/'buffer_runs').glob('*s12_*')):
        if (p/'result.json').is_file():
            r=json.loads((p/'result.json').read_text());runs[r['label']]=r
        elif (p/'attempt_failure.json').is_file():failures.append({'folder':str(p),**json.loads((p/'attempt_failure.json').read_text())})
    rows=[];traces={}
    for label,r in runs.items():
        p=Path(r['folder']);z=trace(r);traces[label]=z;t=z['time_s'];keep=t>=1.;tail=t>=t[-1]-2
        for file,key in [('scene.xml','scene_sha256'),('initialization.json','initialization_sha256'),
                         ('run_snapshot.py','runner_sha256'),('reference_snapshot.py','reference_sha256'),
                         ('load_sharing_snapshot.py','load_sharing_script_sha256'),('wrench_snapshot.py','wrench_script_sha256'),
                         ('uncertainty_snapshot.py','uncertainty_script_sha256'),('stop_profile_snapshot.py','stop_profile_script_sha256'),
                         ('static_payload_snapshot.py','static_payload_script_sha256')]:
            assert hashlib.sha256((p/file).read_bytes()).hexdigest()==r[key],(label,key)
        assert r['equality_constraints']==0
        slope=float(np.polyfit(t[tail],1000*z['relative_change_m'][tail],1)[0])
        case=r['config']['active_motion_case'];task=assess(r,acceleration_limit=.8,angle_limit_deg=.5,clearance_m=.15) if case=='brake' else None
        qualified=r['quality_passed'] and r['original_grasp_criterion_passed'] and (case!='cruise' or r['max_motion_base_speed_error_mps']<.01)
        caps={j['name']:j['force_cap'] for j in r['per_joint']};phase='static' if case=='hold' else ('cruise' if case=='cruise' else 'braking')
        tq=r['phase_joint_peak_torques_Nm'][phase]
        arm=max(tq[n]/caps[n] for n in tq if '_arm_' in n);torso=max(tq[n]/caps[n] for n in tq if n.startswith('torso_'))
        row={'label':label,'case':case,'mass_kg':r['config']['mass_kg'],'duration_s':r['config']['duration_s'],
             'dt_ms':1000*r['config']['dt_s'],'impratio':r['config']['impratio'],'noslip_iterations':r['config']['noslip_iterations'],
             'status':r['status'],'quality':r['quality_passed'],'grasp':r['original_grasp_criterion_passed'],'qualified_case':bool(qualified),
             'braking':r['braking_passed'] if case=='brake' else None,'task':task['passed'] if task else None,
             'max_relative_mm':1000*r['max_relative_change_after_1s_m'],'tail_relative_drift_mmps':slope,
             'first_grasp_violation_s':r['first_grasp_threshold_violation_s'],
             'max_motion_base_speed_error_mps':r['max_motion_base_speed_error_mps'],
             'max_target_angle_deg':float(np.max(z['object_target_angle_error_deg'][keep])),
             'contact_friction_peak':float(np.max(z['max_contact_friction_use'][keep])),
             'normal_force_mean_N':float(np.mean(z['normal_sum_N'][tail])),
             'vertical_force_mean_N':float(np.mean(z['contact_force_z_N'][tail])),
             'arm_torque_fraction':arm,'torso_torque_fraction':torso,
             'brake_acceleration_mps2':r['max_brake_payload_acceleration_mps2'],
             'front_cm':100*r['max_payload_front_travel_m'] if case=='brake' else None,'folder':r['folder']}
        rows.append(row)
    old=json.loads((OUT.parent/'stage11/summary.json').read_text());regression=[]
    for mode in ['fixed','buffer']:
        label='s12_regression_'+mode
        if label not in runs:continue
        a=np.load(Path(old['runs']['s11_final_equal_'+mode]['folder'])/'joint_trace.npz');b=np.load(Path(runs[label]['folder'])/'joint_trace.npz')
        exact=all(np.array_equal(a[k],b[k]) for k in ['time','qpos','qvel','ctrl']);assert exact
        regression.append({'mode':mode,'exact':exact})
    pairs=[];prefix=[]
    for family in ['light05','center1']:
        f='s12_'+family+'_fixed';b='s12_'+family+'_buffer';normal='s12_'+family+'_normal'
        if f in runs and b in runs:pairs.append(pair(runs[f],runs[b],family))
        if f in runs and normal in runs:
            a=np.load(Path(runs[f]['folder'])/'joint_trace.npz');n=np.load(Path(runs[normal]['folder'])/'joint_trace.npz')
            mask=a['time']<3.5;assert np.array_equal(a['time'],n['time'])
            same=all(np.array_equal(a[k][mask],n[k][mask]) for k in ['qpos','qvel','ctrl']);assert same
            prefix.append({'family':family,'normal_and_brake_exact_common_prefix':same})
    indexed={z['label']:z for z in rows};gates=[];diagnostics=[];half=[];reference_checks=[]
    for family in ['light05','center1']:
        hold='s12_'+family+'_hold';normal='s12_'+family+'_normal'
        if hold in indexed and normal in indexed:
            gate=bool(indexed[hold]['qualified_case'] and indexed[normal]['qualified_case'])
            gates.append({'family':family,'static_and_normal_passed':gate,
                          'braking_only_failure_observed':any(not indexed['s12_'+family+'_'+mode]['qualified_case'] for mode in ['fixed','buffer'])
                               if gate and all('s12_'+family+'_'+mode in indexed for mode in ['fixed','buffer']) else None})
        if normal in runs:
            z=traces[normal];post=z['time_s']>=3.5
            error=float(np.max(np.abs(z['base_ref_x_m'][post]-z['base_ref_x_m'][0]-.4*(z['time_s'][post]-2.))))
            assert error<1e-12 and np.all(z['base_ref_v_mps'][post]==.4) and np.all(z['base_ref_a_mps2'][post]==0.)
            reference_checks.append({'family':family,'normal_reference_continuation_error_m':error,'no_hidden_braking':True})
    if 's12_center1_hold' in indexed:
        for suffix,changed in [('halfdt','timestep'),('impratio100','impratio'),('noslip3','noslip_iterations')]:
            label='s12_center1_hold_'+suffix
            if label not in indexed:continue
            base=ET.parse(Path(runs['s12_center1_hold']['folder'])/'scene.xml').getroot()
            other=ET.parse(Path(runs[label]['folder'])/'scene.xml').getroot()
            a=base.find('option').attrib;b=other.find('option').attrib
            differing=[k for k in set(a)|set(b) if a.get(k)!=b.get(k)];assert differing==[changed],differing
            base.remove(base.find('option'));other.remove(other.find('option'));assert ET.tostring(base)==ET.tostring(other)
            diagnostics.append({'label':label,'only_scene_attribute_changed':changed,
                                'relative_mm':indexed[label]['max_relative_mm'],
                                'drift_rate_ratio_to_nominal':indexed[label]['tail_relative_drift_mmps']/indexed['s12_center1_hold']['tail_relative_drift_mmps']})
    for base,other in [('s12_center1_hold','s12_center1_hold_halfdt'),('s12_light05_buffer','s12_light05_buffer_halfdt')]:
        if base not in indexed or other not in indexed:continue
        half.append({'base':base,'half':other,'same_quality_and_grasp':all(indexed[base][k]==indexed[other][k] for k in ['quality','grasp']),
                     'absolute_relative_change_difference_mm':abs(indexed[base]['max_relative_mm']-indexed[other]['max_relative_mm'])})
    summary={'completed_runs':len(runs),'failed_attempts':failures,'all_snapshot_hashes_verified':True,
             'nominal_regression':regression,'pairs':pairs,'normal_brake_prefix':prefix,'metrics':rows,'runs':runs,
             'eligibility_gates':gates,'normal_reference_checks':reference_checks,'solver_diagnostics':diagnostics,'half_timestep_checks':half,
             'scope':'Finite-duration qualification. Solver variations are diagnostic, not a new default contact model. Original physical parameters retained for light-load tests.'}
    dump(DEST/'summary.json',summary)
    if rows:
        with (DEST/'runs.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps([{k:r[k] for k in ['label','quality','grasp','max_relative_mm','tail_relative_drift_mmps','contact_friction_peak','brake_acceleration_mps2']} for r in rows]))


if __name__=='__main__':main()
