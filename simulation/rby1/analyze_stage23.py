"""Phase-aware grasp observations, independently checked against legacy logs."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from build_model import OUT,dump
from build_payload import rotvec
from analyze_buffer import trace


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def integral_interval(t,value,start,end):
    cumulative=np.cumsum(np.nan_to_num(value,nan=0.)*np.diff(t,prepend=t[0]))
    return float(np.interp(end,t,cumulative)-np.interp(start,t,cumulative))


def main():
    dest=OUT.parent/'stage23';checks=json.loads((dest/'independent_checks.json').read_text());old=json.loads((OUT.parent/'stage22/summary.json').read_text())
    frozen={'outputs/rby1-research/model/research_model.xml':'a66346c0839622d469dd4739d5f2a172ab8a1424d86537b43a571925c3c0195e',
            'outputs/rby1-research/payload_model/payload_scene.xml':'d6701100d6fd49deb2541093ca6fca63acf4ffa676d8cbc8e3a3fe29bb824154',
            'simulation/rby1/run_payload.py':'0d426bd07b83fa7d9e9aa55be086826dd5cbfef845aee8a89e706febeeb15906'}
    assert all(sha(Path(path))==expected for path,expected in frozen.items())
    runs={};rows=[];verifications=[]
    for p in sorted((OUT.parent/'buffer_runs').glob('*s23_*/result.json')):
        r=json.loads(p.read_text());assert r['label'] not in runs;runs[r['label']]=r
    assert set(runs)=={'s23_'+k for k in ['regression','early','middle','late','halfdt']}
    for label,r in runs.items():
        case=label.removeprefix('s23_');p=Path(r['folder']);prior=old['runs']['s22v2_'+case]
        for file,key in [('scene.xml','scene_sha256'),('initialization.json','initialization_sha256'),('run_snapshot.py','runner_sha256'),
                         ('state_stop_snapshot.py','state_stop_script_sha256'),('motion_distribution_snapshot.py','motion_distribution_script_sha256'),
                         ('reference_snapshot.py','reference_sha256'),('wrench_snapshot.py','wrench_script_sha256'),('load_sharing_snapshot.py','load_sharing_script_sha256'),
                         ('uncertainty_snapshot.py','uncertainty_script_sha256'),('stop_profile_snapshot.py','stop_profile_script_sha256'),
                         ('static_payload_snapshot.py','static_payload_script_sha256'),('grasp_observation_snapshot.py','grasp_observation_script_sha256')]:assert sha(p/file)==r[key]
        for source,key in [('run_buffer.py','runner_sha256'),('grasp_observation.py','grasp_observation_script_sha256'),('state_stop_reference.py','state_stop_script_sha256'),('motion_distribution.py','motion_distribution_script_sha256')]:
            assert checks['source_sha256'][source]==r[key]
        if 'task_reference_script_sha256' in r:assert sha(p/'task_reference_snapshot.py')==r['task_reference_script_sha256']==checks['source_sha256']['delivery_stop.py']
        before=np.load(Path(prior['folder'])/'joint_trace.npz');after=np.load(p/'joint_trace.npz')
        exact=all(np.array_equal(before[k],after[k]) for k in ['time','qpos','qvel','ctrl']);assert exact
        assert r['scene_sha256']==prior['scene_sha256'] and r['initialization_sha256']==prior['initialization_sha256']
        with np.load(p/'grasp_trace.npz') as archive:g={key:archive[key] for key in archive.files}
        metadata=json.loads((p/'grasp_metadata.json').read_text());t=g['time'];anchor=metadata['trigger_sample_index']
        assert anchor is not None and t[anchor]==3.5 and len(t)==round(6/r['config']['dt_s'])+1
        z=trace(r);index=np.rint(z['time_s']/r['config']['dt_s']).astype(int);assert max(abs(t[index]-z['time_s']))<1e-12
        F=np.sum(g['force_on_object_world'],axis=1);M=np.sum(g['moment_on_object_about_com_world'],axis=1)
        force_error=max(float(max(abs(F[index,j]-z['contact_force_'+axis+'_N']))) for j,axis in enumerate('xyz'))
        moment_error=max(float(max(abs(M[index,j]-z['contact_moment_'+axis+'_Nm']))) for j,axis in enumerate('xyz'))
        assert force_error<1e-10 and moment_error<1e-10
        assert max(abs(np.sum(g['hand_normal_force'][index],axis=1)-z['normal_sum_N']))<1e-10
        assert np.array_equal(np.sum(g['hand_contact_count'][index],axis=1),z['pad_contacts'])
        rho=np.max(np.nan_to_num(g['hand_friction_utilization'][index],nan=0.),axis=1)
        assert max(abs(rho-z['max_contact_friction_use']))<1e-10
        joint_peak=np.max(abs(g['joint_generalized_force'][1:]),axis=0)
        assert max(abs(joint_peak-np.array([j['peak_force'] for j in r['per_joint']])))<1e-8
        verifications.append({'case':case,'state_and_control_exactly_unchanged':exact,'contact_force_sum_error_N':force_error,
                              'contact_moment_sum_error_Nm':moment_error,'high_rate_samples':len(t),'contact_rows':metadata['contact_rows']})
        stop=3.5+r['reference_admission']['candidate_duration_s'];end=float(t[-1]);phase_intervals={'braking':(3.5,stop),'transition':(stop,stop+.1),'hold':(stop+.1,end)}
        delta=g['relative_position']-g['relative_position'][anchor];distance=np.linalg.norm(delta,axis=2)
        angles=np.array([[np.linalg.norm(rotvec(g['relative_rotation'][i,s]@g['relative_rotation'][anchor,s].T)) for s in [0,1]] for i in range(len(t))])
        csv_rows=[]
        for i,time in enumerate(t):
            for side,hand in enumerate(metadata['hands']):
                load=g['contact_load_tcp_hand'][i,side]
                row={'time_s':float(time),'time_from_trigger_s':float(time-3.5),'hand':hand,
                     'relative_change_from_trigger_mm':float(1000*distance[i,side]),'relative_rotation_from_trigger_deg':float(np.rad2deg(angles[i,side])),
                     'normal_force_sum_N':float(g['hand_normal_force'][i,side]),'contact_count':int(g['hand_contact_count'][i,side]),
                     'loaded_count_above_epsilon':int(g['hand_loaded_count'][i,side]),'friction_utilization':float(g['hand_friction_utilization'][i,side]),
                     'friction_margin_N':float(g['hand_friction_margin'][i,side]),'tangent_speed_proxy_mmps':float(1000*g['hand_tangent_speed'][i,side]),
                     'tangent_travel_since_trigger_mm':float(1000*(g['hand_tangent_travel'][i,side]-g['hand_tangent_travel'][anchor,side])) if i>=anchor else None}
                row.update({name:float(load[j]) for j,name in enumerate(['contact_load_Fx_N','contact_load_Fy_N','contact_load_Fz_N','contact_load_Mx_Nm','contact_load_My_Nm','contact_load_Mz_Nm'])})
                row.update({f'relative_position_{axis}_m':float(g['relative_position'][i,side,j]) for j,axis in enumerate('xyz')})
                csv_rows.append(row)
        with (p/'grasp_hand_trace.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
        for phase,(start,finish) in phase_intervals.items():
            mask=(t>=start)&(t<=finish);first=int(np.flatnonzero(t>=start)[0])
            for side,hand in enumerate(metadata['hands']):
                load=g['contact_load_tcp_hand'][:,side];rho=g['hand_friction_utilization'][mask,side];rho=rho[np.isfinite(rho)]
                relative_phase=g['relative_position'][mask,side]-g['relative_position'][first,side]
                rows.append({'label':label,'case':case,'hand':hand,'phase':phase,'start_s':start,'end_s':finish,
                             'peak_relative_change_from_trigger_mm':float(1000*max(distance[mask,side])),
                             'peak_relative_change_from_phase_start_mm':float(1000*max(np.linalg.norm(relative_phase,axis=1))),
                             'peak_relative_rotation_from_trigger_deg':float(np.rad2deg(max(angles[mask,side]))),
                             'tangent_travel_proxy_mm':1000*float(np.interp(finish,t,g['hand_tangent_travel'][:,side])-np.interp(start,t,g['hand_tangent_travel'][:,side])),
                             'peak_weighted_tangent_speed_mmps':float(1000*np.nanmax(g['hand_tangent_speed'][mask,side])),
                             'peak_contact_force_N':float(max(np.linalg.norm(load[mask,:3],axis=1))),
                             'peak_contact_moment_about_tcp_Nm':float(max(np.linalg.norm(load[mask,3:],axis=1))),
                             'peak_friction_utilization':float(max(rho)) if len(rho) else None,
                             'min_hand_normal_force_sum_N':float(min(g['hand_normal_force'][mask,side])),
                             'no_pad_contact_duration_s':integral_interval(t,(g['hand_contact_count'][:,side]==0).astype(float),start,finish),
                             'normal_force_below_recording_epsilon_duration_s':integral_interval(t,(g['hand_normal_force'][:,side]<=metadata['force_recording_epsilon_N']).astype(float),start,finish)})
    with (dest/'phase_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={'run_count':len(runs),'runs':runs,'phase_metrics':rows,'verifications':verifications,'independent_checks':checks,
            'qualification':'Relative-pose change and contact tangential-travel proxy are different observables. Contact wrench is about the EE reference, not a compensated sensor reading. Nominal soft-contact model; no new safety threshold or control change.'}
    dump(dest/'summary.json',result);dump(dest/'delivery_check.json',{'complete_runs':len(runs),'exact_observer_on_off_state_control_regressions':len(verifications),
        'all_snapshots_and_preflight_versions_verified':True,'frozen_models_and_payload_runner_unchanged':True,
        'aggregate_contact_force_and_moment_match_legacy':True,'high_rate_samples_total':sum(v['high_rate_samples'] for v in verifications),
        'raw_contact_rows_total':sum(v['contact_rows'] for v in verifications),
        'physical_failures':[label for label,r in runs.items() if r['status']!='completed' or not all(r[k] for k in ['quality_passed','braking_passed','original_grasp_criterion_passed'])]})
    print(json.dumps([r for r in rows if r['case'] in ['middle','halfdt'] and r['hand']=='left' and r['phase'] in ['braking','hold']]))


if __name__=='__main__':main()
