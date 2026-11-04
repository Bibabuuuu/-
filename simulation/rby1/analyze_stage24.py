"""Matched stop-action comparison with raw pose change and a separate pretrend."""
import csv,json
from pathlib import Path
import numpy as np
from build_model import OUT,dump
from build_payload import rotvec
from analyze_stage23 import sha
from analyze_buffer import trace


def pretrend(t,p,start=3.,end=3.5):
    mask=(t>=start)&(t<=end);x=t[mask]-end;X=np.c_[np.ones(sum(mask)),x]
    coef=np.linalg.lstsq(X,p[mask],rcond=None)[0]
    return coef[1],float(np.max(np.linalg.norm(p[mask]-X@coef,axis=1)))


def interval_travel(t,cumulative,start,end):
    return float(np.interp(end,t,cumulative)-np.interp(start,t,cumulative))


def raw_and_trend(t,p,anchor,velocity):
    raw=p-p[anchor];reference=(t-t[anchor])[:,None]*velocity
    return raw,raw-reference


def position_window(t,p,start,end):
    first=np.array([np.interp(start,t,p[:,k]) for k in range(3)])
    last=np.array([np.interp(end,t,p[:,k]) for k in range(3)])
    return np.vstack([first,p[(t>=start)&(t<=end)],last])


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    dest=OUT.parent/'stage24';checks=json.loads((dest/'independent_checks.json').read_text());old=json.loads((OUT.parent/'stage21/summary.json').read_text())
    runs={}
    for p in sorted((OUT.parent/'buffer_runs').glob('*s24_*/result.json')):
        r=json.loads(p.read_text());assert r['label'] not in runs;runs[r['label']]=r
    expected={'s24_'+s+'_'+p for s in ['normal','forward'] for p in ['fixed','acceleration','motion']}
    assert expected.issubset(runs) and set(runs)-expected<={'s24_'+s+'_'+p+'_halfdt' for s in ['normal','forward'] for p in ['fixed','acceleration','motion']}
    summaries=[];hand_rows=[];phase_rows=[];verified=[];traces={};data={}
    for label,r in runs.items():
        case=label.removeprefix('s24_');half=case.endswith('_halfdt');basecase=case.removesuffix('_halfdt');scenario,policy=basecase.split('_');folder=Path(r['folder'])
        for filename,key in [('run_snapshot.py','runner_sha256'),('state_stop_snapshot.py','state_stop_script_sha256'),('grasp_observation_snapshot.py','grasp_observation_script_sha256'),
                             ('motion_distribution_snapshot.py','motion_distribution_script_sha256'),('reference_snapshot.py','reference_sha256'),('load_sharing_snapshot.py','load_sharing_script_sha256'),
                             ('wrench_snapshot.py','wrench_script_sha256'),('uncertainty_snapshot.py','uncertainty_script_sha256'),('stop_profile_snapshot.py','stop_profile_script_sha256'),('static_payload_snapshot.py','static_payload_script_sha256'),
                             ('scene.xml','scene_sha256'),('initialization.json','initialization_sha256')]:assert sha(folder/filename)==r[key]
        keys={'run_buffer.py':'runner_sha256','state_stop_reference.py':'state_stop_script_sha256','grasp_observation.py':'grasp_observation_script_sha256','motion_distribution.py':'motion_distribution_script_sha256','buffer_reference.py':'reference_sha256','payload_wrench.py':'wrench_script_sha256','load_sharing.py':'load_sharing_script_sha256','control_uncertainty.py':'uncertainty_script_sha256','stop_profiles.py':'stop_profile_script_sha256','static_payload.py':'static_payload_script_sha256'}
        assert all(r[k]==checks['physical_source_sha256'][f] for f,k in keys.items())
        if policy!='fixed':assert sha(folder/'stop_candidates_snapshot.py')==r['stop_candidates_script_sha256']==checks['physical_source_sha256']['stop_candidates.py']
        prior=old['runs']['s21_'+basecase]
        with np.load(folder/'joint_trace.npz') as a:joint={k:a[k] for k in a.files}
        if not half:
            with np.load(Path(prior['folder'])/'joint_trace.npz') as a:exact=all(np.array_equal(a[k],joint[k]) for k in ['time','qpos','qvel','ctrl'])
            assert exact and r['scene_sha256']==prior['scene_sha256'] and r['initialization_sha256']==prior['initialization_sha256']
        else:exact=None
        with np.load(folder/'grasp_trace.npz') as a:g={k:a[k] for k in a.files}
        t=g['time'];anchor=int(np.flatnonzero(t==3.5)[0]);assert len(t)==round(6/r['config']['dt_s'])+1
        z=trace(r);j=np.rint(z['time_s']/r['config']['dt_s']).astype(int);assert max(abs(t[j]-z['time_s']))<1e-12
        F=np.sum(g['force_on_object_world'],axis=1);M=np.sum(g['moment_on_object_about_com_world'],axis=1)
        force_error=max(float(max(abs(F[j,k]-z['contact_force_'+axis+'_N']))) for k,axis in enumerate('xyz'))
        moment_error=max(float(max(abs(M[j,k]-z['contact_moment_'+axis+'_Nm']))) for k,axis in enumerate('xyz'))
        assert force_error<1e-10 and moment_error<1e-10
        stop=3.5+r['reference_admission']['candidate_duration_s'];holdstart=stop+.1;holdend=holdstart+1.;assert holdend<=t[-1]
        phases={'braking':(3.5,stop),'transition':(stop,holdstart),'hold_1s':(holdstart,holdend)}
        local=[];series=[]
        for side,hand in enumerate(['left','right']):
            p=g['relative_position'][:,side];velocity,fit_error=pretrend(t,p);raw,residual=raw_and_trend(t,p,anchor,velocity)
            angles=np.array([np.linalg.norm(rotvec(R@g['relative_rotation'][anchor,side].T)) for R in g['relative_rotation'][:,side]])
            load=g['contact_load_tcp_hand'][:,side];trend=1000*velocity
            for phase,(start,end) in phases.items():
                mask=(t>=start)&(t<=end);window=position_window(t,p,start,end);relative_phase=window-window[0]
                raw_window=position_window(t,raw,start,end);residual_window=position_window(t,residual,start,end)
                row={'label':label,'scenario':scenario,'policy':policy,'halfdt':half,'hand':hand,'phase':phase,'start_s':start,'end_s':end,'duration_s':end-start,
                     'raw_pose_peak_from_trigger_mm':1000*float(max(np.linalg.norm(raw_window,axis=1))),
                     'raw_pose_peak_from_phase_start_mm':1000*float(max(np.linalg.norm(relative_phase,axis=1))),
                     'pretrend_residual_peak_mm':1000*float(max(np.linalg.norm(residual_window,axis=1))),
                     'raw_end_dz_mm':float(1000*raw_window[-1,2]),
                     'pretrend_reference_end_dz_mm':float(1000*velocity[2]*(end-3.5)),
                     'pretrend_residual_end_dz_mm':float(1000*residual_window[-1,2]),
                     'relative_rotation_peak_from_trigger_deg':float(np.rad2deg(max(angles[mask]))),
                     'tangent_travel_proxy_mm':1000*interval_travel(t,g['hand_tangent_travel'][:,side],start,end),
                     'mean_tangent_speed_proxy_mmps':1000*interval_travel(t,g['hand_tangent_travel'][:,side],start,end)/(end-start),
                     'peak_tangent_speed_proxy_mmps':1000*float(np.nanmax(g['hand_tangent_speed'][mask,side])),
                     'peak_contact_force_N':float(max(np.linalg.norm(load[mask,:3],axis=1))),
                     'peak_contact_moment_tcp_Nm':float(max(np.linalg.norm(load[mask,3:],axis=1))),
                     'peak_friction_utilization':float(np.nanmax(g['hand_friction_utilization'][mask,side])),
                     'min_normal_force_sum_N':float(min(g['hand_normal_force'][mask,side])),
                     'no_pad_contact_samples':int(sum(g['hand_contact_count'][mask,side]==0)),
                     'pretrend_vx_mmps':float(trend[0]),'pretrend_vy_mmps':float(trend[1]),'pretrend_vz_mmps':float(trend[2]),'pretrend_fit_max_error_mm':1000*fit_error}
                phase_rows.append(row);local.append(row)
            for i,time in enumerate(t):
                series.append({'time_s':float(time),'hand':hand,'raw_change_mm':1000*float(np.linalg.norm(raw[i])),
                               'raw_dx_mm':1000*float(raw[i,0]),'raw_dy_mm':1000*float(raw[i,1]),'raw_dz_mm':1000*float(raw[i,2]),
                               'pretrend_residual_mm':1000*float(np.linalg.norm(residual[i])),
                               'contact_moment_tcp_Nm':float(np.linalg.norm(load[i,3:])),
                               'contact_tangent_speed_mmps':1000*float(g['hand_tangent_speed'][i,side]),'friction_utilization':float(g['hand_friction_utilization'][i,side])})
            hand_rows.append({'label':label,'hand':hand,'pretrend_velocity_mmps':trend.tolist(),'pretrend_fit_error_mm':1000*fit_error})
        write_csv(folder/'grasp_comparison_trace.csv',series)
        path=np.load(folder/'joint_reference_path.npz');caps={x['name']:x['speed_cap'] for x in r['per_joint']};post=joint['time']>=3.5
        v=joint['qvel'][post][:,path['vadr']];vcaps=np.array([caps[x] for x in path['names'].tolist()]);cost=float(np.trapezoid(np.sum((v/vcaps)**2,axis=1),joint['time'][post]))
        braking=[x for x in local if x['phase']=='braking'];holding=[x for x in local if x['phase']=='hold_1s']
        summary={'label':label,'scenario':scenario,'policy':policy,'halfdt':half,'A_setting_mps2':r['reference_admission']['reference_acceleration_mps2'],
                 'object_stop_duration_s':stop-3.5,'hold_start_s':holdstart,'hold_end_s':holdend,'actual_peak_acceleration_mps2':r['max_brake_payload_acceleration_mps2'],
                 'actual_upper_motion_cost_s':cost,'extra_ee_forward_cm':100*r['max_extra_ee_forward_m'],'base_stop_distance_cm':100*r['brake_distance_to_final_m'],
                 **{k:max(x[k] for x in braking) for k in ['raw_pose_peak_from_trigger_mm','pretrend_residual_peak_mm','relative_rotation_peak_from_trigger_deg','tangent_travel_proxy_mm','mean_tangent_speed_proxy_mmps','peak_contact_moment_tcp_Nm','peak_friction_utilization']},
                 'brake_end_downward_change_mm':max(-x['raw_end_dz_mm'] for x in braking),
                 'pretrend_expected_downward_change_mm':max(-x['pretrend_reference_end_dz_mm'] for x in braking),
                 'hold_1s_pose_change_mm':max(x['raw_pose_peak_from_phase_start_mm'] for x in holding),
                 'hold_1s_tangent_travel_mm':max(x['tangent_travel_proxy_mm'] for x in holding)}
        summaries.append(summary);verified.append({'label':label,'legacy_state_control_exact':exact,'force_sum_error_N':force_error,'moment_sum_error_Nm':moment_error,'samples':len(t)})
        traces[label]=z;data[label]=g
    for scenario in ['normal','forward']:
        base='s24_'+scenario+'_fixed';common=data[base]['time']<=3.5
        for policy in ['acceleration','motion']:
            label='s24_'+scenario+'_'+policy
            assert all(np.array_equal(traces[base][k],traces[label][k]) for k in ['base_ref_x_m','base_ref_v_mps','base_ref_a_mps2'])
            assert all(np.array_equal(data[base][k][common],data[label][k][common]) for k in ['relative_position','relative_rotation','contact_load_tcp_hand','hand_tangent_speed'])
    comparisons=[]
    for scenario in ['normal','forward']:
        ref=next(x for x in summaries if x['label']=='s24_'+scenario+'_fixed')
        for policy in ['acceleration','motion']:
            row=next(x for x in summaries if x['label']=='s24_'+scenario+'_'+policy)
            keys=['actual_peak_acceleration_mps2','actual_upper_motion_cost_s','extra_ee_forward_cm','raw_pose_peak_from_trigger_mm','pretrend_residual_peak_mm','relative_rotation_peak_from_trigger_deg','tangent_travel_proxy_mm','mean_tangent_speed_proxy_mmps','peak_contact_moment_tcp_Nm','peak_friction_utilization','hold_1s_pose_change_mm']
            a=traces['s24_'+scenario+'_fixed'];b=traces['s24_'+scenario+'_'+policy];post=a['time_s']>=3.5
            comparisons.append({'scenario':scenario,'policy':policy,'same_base_reference':True,
                'max_actual_base_position_difference_m':float(max(abs(a['base_x_m'][post]-b['base_x_m'][post]))),
                'max_actual_base_speed_difference_mps':float(max(abs(a['base_v_mps'][post]-b['base_v_mps'][post]))),
                **{k+'_change_percent':100*(row[k]/ref[k]-1) for k in keys}})
    for path,expected_hash in checks['frozen_sha256'].items():assert sha(Path(path))==expected_hash
    write_csv(dest/'runs.csv',summaries);write_csv(dest/'phase_metrics.csv',phase_rows);write_csv(dest/'paired_changes.csv',comparisons)
    result={'runs':runs,'metrics':summaries,'phase_metrics':phase_rows,'pretrend_fits':hand_rows,'comparisons':comparisons,'verifications':verified,
            'hold_duration_s':1.,'pretrend_window_s':[3.,3.5],'qualification':'Raw data and pre-event linear-trend residual remain separate. A nonzero residual can mean LESS downward drift and is not an error score. No causal subtraction, new controller, universal safety threshold or real-gripper claim.'}
    dump(dest/'summary.json',result);dump(dest/'delivery_check.json',{'complete_runs':len(runs),'legacy_exact_regressions':sum(v['legacy_state_control_exact'] is True for v in verified),
        'within_scenario_pretrigger_telemetry_exact':True,'within_scenario_base_reference_exact':True,'snapshot_and_preflight_hashes_verified':True,'frozen_models_verified':True,
        'physical_failures':[k for k,r in runs.items() if r['status']!='completed' or not all(r[x] for x in ['quality_passed','braking_passed','original_grasp_criterion_passed'])]})
    print(json.dumps(summaries))


if __name__=='__main__':main()
